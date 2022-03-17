# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os
import bisect
import copy
import logging

import json
import torch
import torch.utils.data
from maskrcnn_benchmark.utils.comm import get_world_size
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.miscellaneous import save_labels

from . import datasets as D
from . import samplers

from .collate_batch import BatchCollator, BBoxAugCollator
from .transforms import build_transforms

# by Jiaxin
def get_dataset_statistics(cfg):
    """
    get dataset statistics (e.g., frequency bias) from training data
    will be called to help construct FrequencyBias module
    """
    logger = logging.getLogger(__name__)
    logger.info('-'*100)
    logger.info('get dataset statistics...')
    paths_catalog = import_file(
        "maskrcnn_benchmark.config.paths_catalog", cfg.PATHS_CATALOG, True
    )
    DatasetCatalog = paths_catalog.DatasetCatalog
    dataset_names = cfg.DATASETS.TRAIN

    data_statistics_name = ''.join(dataset_names) + '_statistics'
    save_file = os.path.join(cfg.OUTPUT_DIR, "{}.cache".format(data_statistics_name))
    
    if os.path.exists(save_file):
        logger.info('Loading data statistics from: ' + str(save_file))
        logger.info('-'*100)
        return torch.load(save_file, map_location=torch.device("cpu"))
    else:
        logger.info('Unable to load data statistics from: ' + str(save_file))

    statistics = []
    for dataset_name in dataset_names:
        data = DatasetCatalog.get(dataset_name, cfg)
        factory = getattr(D, data["factory"])
        args = data["args"]
        # Remove it because not part of the original repo (factory cant deal with additional parameters...).
        if "capgraphs_file" in args.keys():
            del args["capgraphs_file"]
        dataset = factory(**args)
        statistics.append(dataset.get_statistics())
    logger.info('finish')

    assert len(statistics) == 1
    result = {
        'fg_matrix': statistics[0]['fg_matrix'],
        'pred_dist': statistics[0]['pred_dist'],
        'obj_classes': statistics[0]['obj_classes'], # must be exactly same for multiple datasets
        'rel_classes': statistics[0]['rel_classes'],
        'att_classes': statistics[0]['att_classes'],
    }
    logger.info('Save data statistics to: ' + str(save_file))
    logger.info('-'*100)
    torch.save(result, save_file)
    return result


def build_dataset(cfg, dataset_list, transforms, dataset_catalog, is_train=True):
    """
    Arguments:
        dataset_list (list[str]): Contains the names of the datasets, i.e.,
            coco_2014_trian, coco_2014_val, etc
        transforms (callable): transforms to apply to each (image, target) sample
        dataset_catalog (DatasetCatalog): contains the information on how to
            construct a dataset.
        is_train (bool): whether to setup the dataset for training or testing
    """
    if not isinstance(dataset_list, (list, tuple)):
        raise RuntimeError(
            "dataset_list should be a list of strings, got {}".format(dataset_list)
        )
    datasets = []
    for dataset_name in dataset_list:
        data = dataset_catalog.get(dataset_name, cfg)
        factory = getattr(D, data["factory"])
        args = data["args"]
        # for COCODataset, we want to remove images without annotations
        # during training
        if data["factory"] == "COCODataset":
            args["remove_images_without_annotations"] = is_train
        if data["factory"] == "PascalVOCDataset":
            args["use_difficult"] = not is_train
        args["transforms"] = transforms

        #Remove it because not part of the original repo (factory cant deal with additional parameters...).
        if "capgraphs_file" in args.keys():
            del args["capgraphs_file"]

        # make dataset from factory
        dataset = factory(**args)
        datasets.append(dataset)

    # for testing, return a list of datasets
    if not is_train:
        return datasets

    # for training, concatenate all datasets into a single one
    dataset = datasets[0]
    if len(datasets) > 1:
        dataset = D.ConcatDataset(datasets)

    return [dataset]


def make_data_sampler(dataset, shuffle, distributed):
    if distributed:
        return samplers.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def _quantize(x, bins):
    bins = copy.copy(bins)
    bins = sorted(bins)
    quantized = list(map(lambda y: bisect.bisect_right(bins, y), x))
    return quantized


def _compute_aspect_ratios(dataset):
    aspect_ratios = []
    for i in range(len(dataset)):
        img_info = dataset.get_img_info(i)
        aspect_ratio = float(img_info["height"]) / float(img_info["width"])
        aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def make_batch_data_sampler(
    dataset, sampler, aspect_grouping, images_per_batch, num_iters=None, start_iter=0
):
    if aspect_grouping:
        if not isinstance(aspect_grouping, (list, tuple)):
            aspect_grouping = [aspect_grouping]
        aspect_ratios = _compute_aspect_ratios(dataset)
        group_ids = _quantize(aspect_ratios, aspect_grouping)
        batch_sampler = samplers.GroupedBatchSampler(
            sampler, group_ids, images_per_batch, drop_uneven=False
        )
    else:
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, images_per_batch, drop_last=False
        )
    if num_iters is not None:
        batch_sampler = samplers.IterationBasedBatchSampler(
            batch_sampler, num_iters, start_iter
        )
    return batch_sampler


def make_data_loader(cfg, mode='train', is_distributed=False, start_iter=0, dataset_to_test=None):
    assert mode in {'train', 'val', 'test'}
    assert dataset_to_test in {'train', 'val', 'test', None}
    # this variable enable to run a test on any data split, even on the training dataset
    # without actually flagging it for training....
    if dataset_to_test is None:
        dataset_to_test = mode

    num_gpus = get_world_size()
    is_train = mode == 'train'
    if is_train:
        images_per_batch = cfg.SOLVER.IMS_PER_BATCH
        assert (
            images_per_batch % num_gpus == 0
        ), "SOLVER.IMS_PER_BATCH ({}) must be divisible by the number of GPUs ({}) used.".format(
            images_per_batch, num_gpus)
        images_per_gpu = images_per_batch // num_gpus
        shuffle = True
        num_iters = cfg.SOLVER.MAX_ITER
    else:
        images_per_batch = cfg.TEST.IMS_PER_BATCH
        assert (
            images_per_batch % num_gpus == 0
        ), "TEST.IMS_PER_BATCH ({}) must be divisible by the number of GPUs ({}) used.".format(
            images_per_batch, num_gpus)
        images_per_gpu = images_per_batch // num_gpus
        shuffle = False if not is_distributed else True
        num_iters = None
        start_iter = 0

    if images_per_gpu > 1:
        logger = logging.getLogger(__name__)
        logger.warning(
            "When using more than one image per GPU you may encounter "
            "an out-of-memory (OOM) error if your GPU does not have "
            "sufficient memory. If this happens, you can reduce "
            "SOLVER.IMS_PER_BATCH (for training) or "
            "TEST.IMS_PER_BATCH (for inference). For training, you must "
            "also adjust the learning rate and schedule length according "
            "to the linear scaling rule. See for example: "
            "https://github.com/facebookresearch/Detectron/blob/master/configs/getting_started/tutorial_1gpu_e2e_faster_rcnn_R-50-FPN.yaml#L14"
        )

    # group images which have similar aspect ratio. In this case, we only
    # group in two cases: those with width / height > 1, and the other way around,
    # but the code supports more general grouping strategy
    aspect_grouping = [1] if cfg.DATALOADER.ASPECT_RATIO_GROUPING else []

    paths_catalog = import_file(
        "maskrcnn_benchmark.config.paths_catalog", cfg.PATHS_CATALOG, True
    )
    DatasetCatalog = paths_catalog.DatasetCatalog
    if dataset_to_test == 'train':
        dataset_list = cfg.DATASETS.TRAIN
    elif dataset_to_test == 'val':
        dataset_list = cfg.DATASETS.VAL
    else:
        dataset_list = cfg.DATASETS.TEST

    # If bbox aug is enabled in testing, simply set transforms to None and we will apply transforms later
    transforms = None if not is_train and cfg.TEST.BBOX_AUG.ENABLED else build_transforms(cfg, is_train)
    datasets = build_dataset(cfg, dataset_list, transforms, DatasetCatalog, is_train)

    if is_train:
        # save category_id to label name mapping
        save_labels(datasets, cfg.OUTPUT_DIR)

    data_loaders = []
    for dataset in datasets:
        # print('============')
        # print(len(dataset))
        # print(images_per_gpu)
        # print('============')
        sampler = make_data_sampler(dataset, shuffle, is_distributed)
        batch_sampler = make_batch_data_sampler(
            dataset, sampler, aspect_grouping, images_per_gpu, num_iters, start_iter
        )
        collator = BBoxAugCollator() if not is_train and cfg.TEST.BBOX_AUG.ENABLED else \
            BatchCollator(cfg.DATALOADER.SIZE_DIVISIBILITY)
        num_workers = cfg.DATALOADER.NUM_WORKERS
        data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            collate_fn=collator,
        )
        # the dataset information used for scene graph detection on customized images
        if cfg.TEST.CUSTUM_EVAL:
            custom_data_info = {}
            custom_data_info['idx_to_files'] = dataset.custom_files
            custom_data_info['ind_to_classes'] = dataset.ind_to_classes
            custom_data_info['ind_to_predicates'] = dataset.ind_to_predicates

            if not os.path.exists(cfg.DETECTED_SGG_DIR):
                os.makedirs(cfg.DETECTED_SGG_DIR)

            with open(os.path.join(cfg.DETECTED_SGG_DIR, 'custom_data_info.json'), 'w') as outfile:  
                json.dump(custom_data_info, outfile)
            print('=====> ' + str(os.path.join(cfg.DETECTED_SGG_DIR, 'custom_data_info.json')) + ' SAVED !')
        data_loaders.append(data_loader)
    if is_train:
        # during training, a single (possibly concatenated) data_loader is returned
        assert len(data_loaders) == 1
        return data_loaders[0]
    return data_loaders
