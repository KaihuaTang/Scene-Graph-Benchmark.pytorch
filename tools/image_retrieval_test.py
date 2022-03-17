# train/test scene graph based image retrieval



import argparse
import os
import torch

from maskrcnn_benchmark.image_retrieval.evaluation import run_evaluation
from maskrcnn_benchmark.image_retrieval.modelv2 import SGEncode
from maskrcnn_benchmark.image_retrieval.dataloader import get_loader
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.logger import setup_logger, debug_print
from maskrcnn_benchmark.utils.miscellaneous import mkdir, save_config
from tools.image_retrieval_main import get_dataset, run_test
import numpy as np
# See if we can use apex.DistributedDataParallel instead of the torch default,
# and enable mixed-precision via apex.amp
try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for multi-precision via apex.amp')

# Do Not set it above 5000, otherwise you will start to run tests on the validation data...
GALLERY_SIZE = 150
OUTPUT_DIR = "/media/rafi/Samsung_T5/_DATASETS/vg/new_model/"


def execute_test(cfg, local_rank, distributed, logger, gallery_size):
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

    test_data_loader = get_loader(cfg, train_ids, test_ids, sg_data=sg_data, test_on=True, val_on=False, num_test=gallery_size, num_val=1000)

    debug_print(logger, 'end dataloader')
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    if os.path.exists(cfg.MODEL.PRETRAINED_DETECTOR_CKPT):
        checkpoint = torch.load(cfg.MODEL.PRETRAINED_DETECTOR_CKPT, map_location=torch.device("cpu"))
        print("Loading pretrained model:", cfg.MODEL.PRETRAINED_DETECTOR_CKPT)
        model.load_state_dict(checkpoint)

    test_result = run_test(cfg, model, test_data_loader, distributed, logger)

    cat_data = []
    for item in test_result:
        cat_data.append(item[0])
    # shape [num_image, 2, hidden_dim]
    stacked_vectors = torch.cat(cat_data, dim=0).squeeze(2)

    return model, stacked_vectors


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
    parser.add_argument("--gallery-size", type=int, default=GALLERY_SIZE)


    args = parser.parse_args()
    #Work Around due to cfg.merge_from_list()
    gallery_size = args.gallery_size
    # Do Not set it above 5000, otherwise you will start to run tests on the validation data...
    assert gallery_size <= 5000
    del(args.gallery_size)

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
    else:
        output_dir = OUTPUT_DIR
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

    _, test_result = execute_test(cfg, args.local_rank, args.distributed, logger, gallery_size)

    threshold_list = [None]
    # This range has been chosen because the mean of the diagonal on the dev set was around 0.9X
    threshold_list.extend(np.linspace(0.80, 0.99, 15))

    fei_fei_recall = "feifei_johnson_recall"
    ir_type = f"vg_{gallery_size}_graph_query"

    eval_name = lambda caption_type, recall_type: f"{caption_type}_{recall_type}"

    name = eval_name(ir_type, fei_fei_recall)



    run_evaluation(name, test_result, threshold_list, output_dir)

if __name__ == "__main__":
    main()
