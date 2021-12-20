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
from maskrcnn_benchmark.config.paths_catalog import DatasetCatalog
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
import os

class SGEncoding(data.Dataset):
    """ SGEncoding dataset """
    def __init__(self, train_ids, test_ids, sg_data, test_on=False, val_on=False, num_test=5000, num_val=5000):
        super(SGEncoding, self).__init__()

        data_dir = DatasetCatalog.DATA_DIR
        attrs = DatasetCatalog.DATASETS["VG_stanford_filtered_with_attribute"]
        cap_graph_file = os.path.join(data_dir, attrs["capgraphs_file"])
        vg_dict_file =  os.path.join(data_dir, attrs["dict_file"])

        cap_graph = json.load(open(cap_graph_file))
        vg_dict = json.load(open(vg_dict_file))
        self.img_txt_sg = sg_data
        self.key_list = list(self.img_txt_sg.keys())
        self.key_list.sort()
        self.train_ids = train_ids
        self.test_ids = test_ids
        if test_on:
            self.key_list = self.test_ids[:num_test]
        elif val_on:
            self.key_list = self.test_ids[num_test:num_test+num_val]
        else:
            self.key_list = self.test_ids[num_test+num_val:] + self.train_ids

        # generate union predicate vocabulary
        self.sgg_rel_vocab = list(set(cap_graph['idx_to_meta_predicate'].values()))
        self.txt_rel_vocab = list(set(cap_graph['cap_predicate'].keys()))

        # generate union object vocabulary
        self.sgg_obj_vocab = list(set(vg_dict['idx_to_label'].values()))
        self.txt_obj_vocab = list(set(cap_graph['cap_category'].keys()))

        # vocabulary length
        self.num_sgg_rel = len(self.sgg_rel_vocab)
        self.num_txt_rel = len(self.txt_rel_vocab)
        self.num_sgg_obj = len(self.sgg_obj_vocab)
        self.num_txt_obj = len(self.txt_obj_vocab)

    def _to_tensor(self, inp_dict):
        return {'entities': torch.LongTensor(inp_dict['entities']), 
                'relations': torch.LongTensor(inp_dict['relations'])}

    def _generate_tensor_by_idx(self, idx):
        img = self._to_tensor(self.img_txt_sg[self.key_list[idx]]['img'])
        img_graph = torch.FloatTensor(self.img_txt_sg[self.key_list[idx]]['image_graph'])
        txt = self._to_tensor(self.img_txt_sg[self.key_list[idx]]['txt'])
        txt_graph = torch.FloatTensor(self.img_txt_sg[self.key_list[idx]]['text_graph'])
        img['graph'] = img_graph
        txt['graph'] = txt_graph
        return img, txt

    def __getitem__(self, item):
        fg_img, fg_txt = self._generate_tensor_by_idx(item)
        # generate negative sample
        bg_idx = item
        while(bg_idx == item):
            bg_idx = int(random.random() * len(self.key_list))
        bg_img, bg_txt = self._generate_tensor_by_idx(bg_idx)
        return fg_img, fg_txt, bg_img, bg_txt

    def __len__(self):
        return len(self.key_list)
        
class SimpleCollator(object):
    def __call__(self, batch):
        return list(zip(*batch))

def get_loader(cfg, train_ids, test_ids, sg_data, test_on=False, val_on=False, num_test=5000, num_val=1000):
    """ Returns a data loader for the desired split """
    split = SGEncoding(train_ids, test_ids, sg_data=sg_data, test_on=test_on, val_on=val_on, num_test=num_test, num_val=num_val)

    loader = torch.utils.data.DataLoader(split,
        batch_size=cfg.SOLVER.IMS_PER_BATCH,
        shuffle=not (test_on or val_on),  # only shuffle the data in training
        pin_memory=True,
        num_workers=4,
        collate_fn=SimpleCollator(),
    )
    return loader
