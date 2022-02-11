# train/test scene graph based image retrieval

from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
import time
import datetime
import json
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import torch.utils.data as data
from torch.nn.utils import weight_norm
from tqdm import tqdm

from maskrcnn_benchmark.image_retrieval.evaluation import evaluator
from maskrcnn_benchmark.image_retrieval.modelv2 import SGEncode
from maskrcnn_benchmark.image_retrieval.dataloader import get_loader
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.engine.trainer import reduce_loss_dict
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.checkpoint import clip_grad_norm
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank, all_gather
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import setup_logger, debug_print
from maskrcnn_benchmark.utils.miscellaneous import mkdir, save_config
from maskrcnn_benchmark.utils.metric_logger import MetricLogger

# See if we can use apex.DistributedDataParallel instead of the torch default,
# and enable mixed-precision via apex.amp
try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for multi-precision via apex.amp')

sg_model_name = 'motif'
sg_fusion_name = 'rubi'
sg_type_name = 'origin'

#sg_train_path = '/data1/image_retrieval/causal_{}_sgdet_{}_{}_train.pytorch'.format(sg_model_name, sg_fusion_name, sg_type_name)
#sg_test_path = '/data1/image_retrieval/causal_{}_sgdet_{}_{}_test.pytorch'.format(sg_model_name, sg_fusion_name, sg_type_name)
#output_path = '/data1/image_retrieval_model/causal_'+sg_model_name+'_sgdet_'+sg_fusion_name+'_'+sg_type_name+'_output_%s_%d.pytorch'

#Make results reproducibles
torch.manual_seed(0)
import random
random.seed(0)

output_path = '/media/rafi/Samsung_T5/_DATASETS/vg/new_model/results/sg_of_causal_sgdet_ctx_only_%s_%d.pytorch'
sg_train_path = '/media/rafi/Samsung_T5/_DATASETS/vg/new_model/sgg/train_sg_of_causal_sgdet_ctx_only.json'
sg_test_path = '/media/rafi/Samsung_T5/_DATASETS/vg/new_model/sgg/test_sg_of_causal_sgdet_ctx_only.json'
sg_val_path = '/media/rafi/Samsung_T5/_DATASETS/vg/new_model/sgg/val_sg_of_causal_sgdet_ctx_only.json'

def get_dataset():
    """
    Returns the ids of training samples, testing samples, and all scene graph relevant data for training
    :return:
    """
    print("Loading samples. This can take a while.")
    sg_data_train = json.load(open(sg_train_path))
    sg_data_val = json.load(open(sg_val_path))
    sg_data_test = json.load(open(sg_test_path))
    #sg_data = torch.load(sg_train_path)
    #sg_data.update(torch.load(sg_test_path))
    #Merge the val sample to the training data, it would be a waste...
    sg_data_train.update(sg_data_val)
    sg_data = sg_data_train.copy()
    sg_data.update(sg_data_test)
    train_ids = list(sg_data_train.keys())
    print("Number of Training Samples", len(train_ids))
    test_ids = list(sg_data_test.keys())
    print("Number of Testing Samples", len(test_ids))
    return train_ids, test_ids, sg_data

def train(cfg, local_rank, distributed, logger):
    model = SGEncode()
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    num_batch = cfg.SOLVER.IMS_PER_BATCH
    optimizer = make_optimizer(cfg, model, logger, rl_factor=float(num_batch))
    scheduler = make_lr_scheduler(cfg, optimizer, logger)
    debug_print(logger, 'end optimizer and shcedule')
    # Initialize mixed-precision training
    use_mixed_precision = cfg.DTYPE == "float16"
    amp_opt_level = 'O1' if use_mixed_precision else 'O0'
    model, optimizer = amp.initialize(model, optimizer, opt_level=amp_opt_level)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
    debug_print(logger, 'end distributed')

    train_ids, test_ids, sg_data = get_dataset()

    train_data_loader = get_loader(cfg, train_ids, test_ids, sg_data=sg_data, test_on=False, val_on=False, num_test=5000, num_val=1000)
    val_data_loader = get_loader(cfg, train_ids, test_ids, sg_data=sg_data, test_on=False, val_on=True, num_test=5000, num_val=1000)
    test_data_loader = get_loader(cfg, train_ids, test_ids, sg_data=sg_data, test_on=True, val_on=False, num_test=5000, num_val=1000)

    debug_print(logger, 'end dataloader')
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    if os.path.exists(cfg.MODEL.PRETRAINED_DETECTOR_CKPT):
        checkpoint = torch.load(cfg.MODEL.PRETRAINED_DETECTOR_CKPT, map_location=torch.device("cpu"))
        print("Loading pretrained model:", cfg.MODEL.PRETRAINED_DETECTOR_CKPT)
        model.load_state_dict(checkpoint)

    if cfg.SOLVER.PRE_VAL:
        logger.info("Validate before training")
        val_result = run_test(cfg, model, val_data_loader, distributed, logger)
        evaluator(logger, val_result)

    logger.info("Start training")
    max_iter = len(train_data_loader)
    start_training_time = time.time()
    end = time.time()

    test_result = run_test(cfg, model, test_data_loader, distributed, logger)
    evaluator(logger, test_result)
    torch.save(test_result, output_path % ('test', 0))

    print_first_grad = True
    for epoch in tqdm(range(cfg.SOLVER.MAX_ITER)):
        epoch_loss = []
        bad_sample_list = []
        for iteration, (fg_imgs, fg_txts, bg_imgs, bg_txts) in enumerate(tqdm(train_data_loader)):
            data_time = time.time() - end
            iteration = iteration + 1
            model.train()

            for sub_iteration, entry in enumerate(zip(fg_imgs, fg_txts, bg_imgs, bg_txts)):
                fg_img, fg_txt, bg_img, bg_txt = entry
                # If no relationship is captured, ignore the whole sample (positive, negative)
                if len(fg_img['entities']) < 2 \
                        or len(fg_txt['entities']) < 2 \
                        or len(fg_img['relations']) < 1 \
                        or len(fg_txt['relations']) < 1 \
                        or len(bg_img['entities']) < 2 \
                        or len(bg_txt['entities']) < 2 \
                        or len(bg_img['relations']) < 1 \
                        or len(bg_txt['relations']) < 1:
                    bad_sample_list.append(sub_iteration)
                    next
                fg_img['entities'] = fg_img['entities'].to(device)
                fg_img['relations'] = fg_img['relations'].to(device)
                fg_img['graph'] = fg_img['graph'].to(device)
                fg_txt['entities'] = fg_txt['entities'].to(device)
                fg_txt['relations'] = fg_txt['relations'].to(device)
                fg_txt['graph'] = fg_txt['graph'].to(device)
                bg_img['entities'] = bg_img['entities'].to(device)
                bg_img['relations'] = bg_img['relations'].to(device)
                bg_img['graph'] = bg_img['graph'].to(device)
                bg_txt['entities'] = bg_txt['entities'].to(device)
                bg_txt['relations'] = bg_txt['relations'].to(device)
                bg_txt['graph'] = bg_txt['graph'].to(device)

            for i in reversed(bad_sample_list):
                del (fg_imgs[i])
                del (fg_txts[i])
                del (bg_imgs[i])
                del (bg_txts[i])
            bad_sample_list.clear()

            if len(fg_imgs) > 0:
                loss_list = model(fg_imgs, fg_txts, bg_imgs, bg_txts)

                losses = sum(loss_list) / (len(loss_list) + 1e-9)
                epoch_loss.append(float(losses))
                #print("batch loss; ", float(losses))
                optimizer.zero_grad()
                # Note: If mixed precision is not used, this ends up doing nothing
                # Otherwise apply loss scaling for mixed-precision recipe
                with amp.scale_loss(losses, optimizer) as scaled_losses:
                    scaled_losses.backward()

                # add clip_grad_norm from MOTIFS, used for debug
                verbose = (iteration % cfg.SOLVER.PRINT_GRAD_FREQ) == 0 or print_first_grad # print grad or not
                print_first_grad = False
                clip_grad_norm([(n, p) for n, p in model.named_parameters() if p.requires_grad], max_norm=cfg.SOLVER.GRAD_NORM_CLIP, logger=logger, verbose=verbose, clip=True)

                optimizer.step()
                # scheduler should be called after optimizer.step() in pytorch>=1.1.0
                assert cfg.SOLVER.SCHEDULE.TYPE != "WarmupReduceLROnPlateau"
                scheduler.step()

            batch_time = time.time() - end
            end = time.time()

        logger.info("epoch: {epoch} loss: {loss:.6f} lr: {lr:.6f}".format(epoch=epoch, loss=float(sum(epoch_loss) / len(epoch_loss)), lr=optimizer.param_groups[-1]["lr"]))
        
        if epoch % checkpoint_period == 0:
            save_path = os.path.join(cfg.OUTPUT_DIR, "model_{}.pytorch".format(str(epoch)))
            logger.info(f"Saving model {save_path}")
            torch.save(model.state_dict(), save_path)
        if epoch == max_iter:
            save_path =  os.path.join(cfg.OUTPUT_DIR, "model_final.pytorch")
            logger.info(f"Saving model {save_path}")
            torch.save(model.state_dict(), save_path)

        val_result = None # used for scheduler updating
        if cfg.SOLVER.TO_VAL and epoch % cfg.SOLVER.VAL_PERIOD == 0:
            logger.info("Start testing")
            test_result = run_test(cfg, model, test_data_loader, distributed, logger)
            test_similarity = evaluator(logger, test_result)
            torch.save({'result' : test_result, 'similarity' : test_similarity}, output_path % ('test', epoch))
            logger.info("Start validating")
            val_result = run_test(cfg, model, val_data_loader, distributed, logger)
            val_similarity = evaluator(logger, val_result)
            torch.save({'result' : val_result, 'similarity' : val_similarity}, output_path % ('val', epoch))
        


    
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )

    return model, test_result

def run_val(cfg, model, val_data_loader, distributed, logger):
    if distributed:
        model = model.module
    torch.cuda.empty_cache()
    device = torch.device(cfg.MODEL.DEVICE)
    model.eval()

    val_result = []
    bad_sample_list = []
    logger.info('START VALIDATION with size: ' + str(len(val_data_loader)))
    for iteration, (fg_imgs, fg_txts, bg_imgs, bg_txts) in enumerate(tqdm(val_data_loader)):
        for sub_iteration, entry in enumerate(zip(fg_imgs, fg_txts, bg_imgs, bg_txts)):
            fg_img, fg_txt, bg_img, bg_txt = entry
            # If no relationship is captured, ignore the whole sample (positive, negative)
            if len(fg_img['entities']) < 2\
                    or len(fg_txt['entities']) < 2\
                    or len(fg_img['relations']) < 1\
                    or len(fg_txt['relations']) < 1 \
                    or len(bg_img['entities']) < 2 \
                    or len(bg_txt['entities']) < 2 \
                    or len(bg_img['relations']) < 1 \
                    or len(bg_txt['relations']) < 1:
                bad_sample_list.append(sub_iteration)
                next
            fg_img['entities'] = fg_img['entities'].to(device)
            fg_img['relations'] = fg_img['relations'].to(device)
            fg_img['graph'] = fg_img['graph'].to(device)
            fg_txt['entities'] = fg_txt['entities'].to(device)
            fg_txt['relations'] = fg_txt['relations'].to(device)
            fg_txt['graph'] = fg_txt['graph'].to(device)
            bg_img['entities'] = bg_img['entities'].to(device)
            bg_img['relations'] = bg_img['relations'].to(device)
            bg_img['graph'] = bg_img['graph'].to(device)
            bg_txt['entities'] = bg_txt['entities'].to(device)
            bg_txt['relations'] = bg_txt['relations'].to(device)
            bg_txt['graph'] = bg_txt['graph'].to(device)

        for i in reversed(bad_sample_list):
            del(fg_imgs[i])
            del(fg_txts[i])
            del(bg_imgs[i])
            del(bg_txts[i])
        bad_sample_list.clear()
        if len(fg_imgs) > 0:
            loss_list = model(fg_imgs, fg_txts, bg_imgs, bg_txts)

            losses = sum(loss_list)

            synchronize()
            val_result.append(float(losses))
    # support for multi gpu distributed testing
    gathered_result = all_gather(torch.tensor(val_result).cpu())
    gathered_result = [t.view(-1) for t in gathered_result]
    gathered_result = torch.cat(gathered_result, dim=-1).view(-1)
    valid_result = gathered_result[gathered_result>=0]
    val_result = float(valid_result.mean())
    del gathered_result, valid_result
    torch.cuda.empty_cache()
    return val_result

def to_cpu(inp_list):
    cpu_output = []
    for item in inp_list:
        cpu_output.append(torch.stack([item[0].detach().cpu(), item[1].detach().cpu()], dim=0))
    return torch.stack(cpu_output, dim=0)

def run_test(cfg, model, test_data_loader, distributed, logger):
    if distributed:
        model = model.module
    torch.cuda.empty_cache()
    device = torch.device(cfg.MODEL.DEVICE)
    model.eval()

    test_result = []
    logger.info('START TEST with size: ' + str(len(test_data_loader)))
    bad_sample_list = []
    for iteration, (fg_imgs, fg_txts, bg_imgs, bg_txts) in enumerate(tqdm(test_data_loader)):
        for sub_iteration, entry in enumerate(zip(fg_imgs, fg_txts, bg_imgs, bg_txts)):
            fg_img, fg_txt, bg_img, bg_txt = entry
            # If no relationship is captured, ignore the whole sample (positive, negative)
            if len(fg_img['entities']) < 2\
                    or len(fg_txt['entities']) < 2\
                    or len(fg_img['relations']) < 1\
                    or len(fg_txt['relations']) < 1 \
                    or len(bg_img['entities']) < 2 \
                    or len(bg_txt['entities']) < 2 \
                    or len(bg_img['relations']) < 1 \
                    or len(bg_txt['relations']) < 1:
                bad_sample_list.append(sub_iteration)
                continue

            fg_img['entities'] = fg_img['entities'].to(device)
            fg_img['relations'] = fg_img['relations'].to(device)
            fg_img['graph'] = fg_img['graph'].to(device)
            fg_txt['entities'] = fg_txt['entities'].to(device)
            fg_txt['relations'] = fg_txt['relations'].to(device)
            fg_txt['graph'] = fg_txt['graph'].to(device)
            bg_img['entities'] = bg_img['entities'].to(device)
            bg_img['relations'] = bg_img['relations'].to(device)
            bg_img['graph'] = bg_img['graph'].to(device)
            bg_txt['entities'] = bg_txt['entities'].to(device)
            bg_txt['relations'] = bg_txt['relations'].to(device)
            bg_txt['graph'] = bg_txt['graph'].to(device)

        synchronize()
        for i in reversed(bad_sample_list):
            del(fg_imgs[i])
            del(fg_txts[i])
            del(bg_imgs[i])
            del(bg_txts[i])
        bad_sample_list.clear()
        if len(fg_imgs) > 0:
            test_output = model(fg_imgs, fg_txts, bg_imgs, bg_txts, is_test=True)
            gathered_result = all_gather(to_cpu(test_output).cpu())
            test_result.append(gathered_result)
    return test_result

def main():
    parser = argparse.ArgumentParser(description="PyTorch Relation Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("image_retrieval_using_sg", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    output_config_path = os.path.join(cfg.OUTPUT_DIR, 'config.yml')
    logger.info("Saving config into: {}".format(output_config_path))
    # save overloaded model config in the output directory
    save_config(cfg, output_config_path)

    model, test_result = train(cfg, args.local_rank, args.distributed, logger)
    evaluator(logger, test_result)

if __name__ == "__main__":
    main()
