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

def evaluator(logger, input_lists):
    cat_data = []
    for item in input_lists:
        cat_data.append(item[0])
    # shape [num_image, 2, hidden_dim]
    cat_data = torch.cat(cat_data, dim=0).squeeze(2)

    similarity = cat_data[:, 0, :] @ (cat_data[:, 1, :].transpose(0,1))   # img to txt
    similarity = similarity.transpose(0,1)                                # txt to img

    pred_rank = (similarity > similarity.diag().view(-1, 1)).sum(-1)

    num_sample = pred_rank.shape[0]
    thres = [1, 5, 10, 20, 50, 100]
    for k in thres:
        logger.info('Recall @ %d: %.4f; ' % (k, float((pred_rank<k).sum()) / num_sample))

    return similarity