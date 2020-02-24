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
from maskrcnn_benchmark.layers import smooth_l1_loss

class FCNet(nn.Module):
    def __init__(self, in_size, out_size, activate=None, drop=0.0):
        super(FCNet, self).__init__()
        self.lin = weight_norm(nn.Linear(in_size, out_size), dim=None)
        self.drop_value = drop
        self.drop = nn.Dropout(drop)
        # in case of using upper character by mistake
        self.activate = activate.lower() if (activate is not None) else None 
        if activate == 'relu':
            self.ac_fn = nn.ReLU()
        elif activate == 'sigmoid':
            self.ac_fn = nn.Sigmoid()
        elif activate == 'tanh':
            self.ac_fn = nn.Tanh()

    def forward(self, x):
        if self.drop_value > 0:
            x = self.drop(x)
        x = self.lin(x)
        if self.activate is not None:
            x = self.ac_fn(x)
        return x


class ApplyAttention(nn.Module):
    def __init__(self, v_features, q_features, mid_features, glimpses, drop=0.0):
        super(ApplyAttention, self).__init__()
        self.glimpses = glimpses
        layers = []
        for g in range(self.glimpses):
            layers.append(ApplySingleAttention(v_features, q_features, mid_features, drop))
        self.glimpse_layers = nn.ModuleList(layers)
    
    def forward(self, v, q, atten):
        """
        v = batch, num_obj, dim
        q = batch, que_len, dim
        atten:  batch x glimpses x v_num x q_num
        """
        for g in range(self.glimpses):
            atten_h = self.glimpse_layers[g](v, q, atten)
            q = q + atten_h
        #q = q * q_mask.unsqueeze(2)
        return q.sum(1)

class ApplySingleAttention(nn.Module):
    def __init__(self, v_features, q_features, mid_features, drop=0.0):
        super(ApplySingleAttention, self).__init__()
        self.lin_v = FCNet(v_features, mid_features, activate='relu', drop=drop)  # let self.lin take care of bias
        self.lin_q = FCNet(q_features, mid_features, activate='relu', drop=drop)
        self.lin_atten = FCNet(mid_features, mid_features, drop=drop)
        
    def forward(self, v, q, atten):
        """
        v = batch, num_obj, dim
        q = batch, que_len, dim
        atten:  batch x v_num x q_num
        """

        # apply single glimpse attention
        v_ = self.lin_v(v).transpose(1,2).unsqueeze(2) # batch, dim, 1, num_obj
        q_ = self.lin_q(q).transpose(1,2).unsqueeze(3) # batch, dim, que_len, 1
        v_ = torch.matmul(v_, atten.unsqueeze(1)) # batch, dim, 1, que_len
        h_ = torch.matmul(v_, q_) # batch, dim, 1, 1
        h_ = h_.squeeze(3).squeeze(2) # batch, dim
        
        atten_h = self.lin_atten(h_.unsqueeze(1))

        return atten_h


class SGEncode(nn.Module):
    def __init__(self, img_num_obj=151, img_num_rel=51, txt_num_obj=4460, txt_num_rel=646):
        super(SGEncode, self).__init__()
        self.embed_dim = 512
        self.hidden_dim = 512
        self.final_dim = 1024
        self.num_layer = 2
        self.margin = 1.0
        self.img_num_obj = img_num_obj
        self.img_num_rel = img_num_rel
        self.txt_num_obj = txt_num_obj
        self.txt_num_rel = txt_num_rel

        self.img_obj_embed = nn.Embedding(self.img_num_obj, self.embed_dim)
        self.img_rel_head_embed = nn.Embedding(self.img_num_obj, self.embed_dim)
        self.img_rel_tail_embed = nn.Embedding(self.img_num_obj, self.embed_dim)
        self.img_rel_pred_embed = nn.Embedding(self.img_num_rel, self.embed_dim)
        self.txt_obj_embed = nn.Embedding(self.txt_num_obj, self.embed_dim)
        self.txt_rel_head_embed = nn.Embedding(self.txt_num_obj, self.embed_dim)
        self.txt_rel_tail_embed = nn.Embedding(self.txt_num_obj, self.embed_dim)
        self.txt_rel_pred_embed = nn.Embedding(self.txt_num_rel, self.embed_dim)

        self.apply_attention = ApplyAttention(
            v_features=self.embed_dim*3,
            q_features=self.embed_dim,
            mid_features=self.hidden_dim,
            glimpses=self.num_layer,
            drop=0.2,)

        self.final_fc = nn.Sequential(*[nn.Linear(self.hidden_dim, self.hidden_dim), 
                                            nn.ReLU(inplace=True),
                                            nn.Linear(self.hidden_dim, self.final_dim),
                                            nn.ReLU(inplace=True)
                                        ])

    def encode(self, inp_dict, is_img=False, is_txt=False):
        assert is_img + is_txt
        if len(inp_dict['relations'].shape) == 1:
            inp_dict['relations'] = torch.zeros(1,3).to(inp_dict['entities'].device).long()
            inp_dict['graph'] = torch.zeros(len(inp_dict['entities']), 1).to(inp_dict['entities'].device).float()

        if is_img:
            obj_encode = self.img_obj_embed(inp_dict['entities'])
            rel_head_encode = self.img_rel_head_embed(inp_dict['relations'][:,0])
            rel_tail_encode = self.img_rel_tail_embed(inp_dict['relations'][:,1])
            rel_pred_encode = self.img_rel_pred_embed(inp_dict['relations'][:,2])
        elif is_txt:
            obj_encode = self.txt_obj_embed(inp_dict['entities'])
            rel_head_encode = self.txt_rel_head_embed(inp_dict['relations'][:,0])
            rel_tail_encode = self.txt_rel_tail_embed(inp_dict['relations'][:,1])
            rel_pred_encode = self.txt_rel_pred_embed(inp_dict['relations'][:,2])
        else:
            print('ERROR')

        rel_encode = torch.cat((rel_head_encode, rel_tail_encode, rel_pred_encode), dim=-1)

        atten = inp_dict['graph'].transpose(0, 1)  # num_rel, num_obj
        atten = atten / (atten.sum(0).view(1, -1) + 1e-9)

        sg_encode = self.apply_attention(rel_encode.unsqueeze(0), obj_encode.unsqueeze(0), atten.unsqueeze(0))

        return self.final_fc(sg_encode).sum(0).view(1, -1)

    def forward(self, fg_imgs, fg_txts, bg_imgs, bg_txts, is_test=False):
        loss = []
        encode_list = []
        for fg_img, fg_txt, bg_img, bg_txt in zip(fg_imgs, fg_txts, bg_imgs, bg_txts):
            fg_img_encode = self.encode(fg_img, is_img=True)
            fg_txt_encode = self.encode(fg_txt, is_txt=True)
            bg_img_encode = self.encode(bg_img, is_img=True)
            bg_txt_encode = self.encode(bg_txt, is_txt=True)

            fg_intra = smooth_l1_loss(fg_img_encode, fg_txt_encode)
            fg_inter = smooth_l1_loss(fg_img_encode, bg_txt_encode)
            triplet_fg = fg_intra + self.margin - fg_inter
            triplet_fg = triplet_fg * (triplet_fg >= 0).float()
            loss.append(triplet_fg.sum())

            bg_intra = smooth_l1_loss(bg_txt_encode, bg_img_encode)
            bg_inter = smooth_l1_loss(fg_txt_encode, bg_img_encode)
            triplet_bg = bg_intra + self.margin - bg_inter
            triplet_bg = triplet_bg * (triplet_bg >= 0).float()

            loss.append(triplet_bg.sum())
            encode_list.append([fg_img_encode, fg_txt_encode])
        
        if is_test:
            return encode_list
        else:
            return loss