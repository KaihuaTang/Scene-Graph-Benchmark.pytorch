from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
import time
import datetime
import json
import random

import h5py
import numpy as np

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
from maskrcnn_benchmark.config.paths_catalog import DatasetCatalog

def preprocess_scene_graphs_output( detected_path, output_file_name):
    data_dir = DatasetCatalog.DATA_DIR
    attrs = DatasetCatalog.DATASETS["VG_stanford_filtered_with_attribute"]
    cap_graph_file = os.path.join(data_dir, attrs["capgraphs_file"])
    vg_dict_file = os.path.join(data_dir, attrs["dict_file"])
    image_file = os.path.join(data_dir, attrs["image_file"])
    roidb_file = os.path.join(data_dir, attrs["roidb_file"])
    # where to load detected scene graph
    #detected_path = '/home/kaihua/checkpoints/causal_sgdet_ctx_only/inference/VG_stanford_filtered_wth_attribute_test/'
    #detected_path = '/media/rafi/Samsung_T5/_DATASETS/vg/'
    #detected_path = '/home/users/alatif/data/ImageCorpora/vg/checkpoint/causal-motifs-sgdet/inference/VG_stanford_filtered_with_attribute_test/'
    # where to save the generated annotation
    #output_path = '/media/rafi/Samsung_T5/_DATASETS/vg/sg_of_causal_sgdet_ctx_only.json'
    output_path = detected_path + output_file_name

    # cap_graph = json.load(open('/data1/vg_capgraphs_anno.json'))
    # vg_data = h5py.File('/home/kaihua/projects/maskrcnn-benchmark/datasets/vg/VG-SGG-with-attri.h5', 'r')
    # vg_dict = json.load(open('/home/kaihua/projects/maskrcnn-benchmark/datasets/vg/VG-SGG-dicts-with-attri.json'))
    # vg_info = json.load(open('/home/kaihua/projects/maskrcnn-benchmark/datasets/vg/image_data.json'))

    cap_graph = json.load(open(cap_graph_file))
    vg_data = h5py.File(roidb_file, 'r')
    vg_dict = json.load(open(vg_dict_file))
    vg_info = json.load(open(image_file))

    # generate union predicate vocabulary
    sgg_rel_vocab = list(set(cap_graph['idx_to_meta_predicate'].values()))
    txt_rel_vocab = list(set(cap_graph['cap_predicate'].keys()))
    uni_rel_vocab = list(set(list(cap_graph['cap_predicate'].keys()) + list(cap_graph['idx_to_meta_predicate'].values())))

    sgg_rel2id = {key: i+1 for i, key in enumerate(sgg_rel_vocab)}
    txt_rel2id = {key: i+1 for i, key in enumerate(txt_rel_vocab)}
    uni_rel2id = {key: i+1 for i, key in enumerate(uni_rel_vocab)}

    sgg_id2rel = {val:key for key,val in sgg_rel2id.items()}
    txt_id2rel = {val:key for key,val in txt_rel2id.items()}
    uni_id2rel = {val:key for key,val in uni_rel2id.items()}

    # generate union object vocabulary
    sgg_obj_vocab = list(set(vg_dict['idx_to_label'].values()))
    txt_obj_vocab = list(set(cap_graph['cap_category'].keys()))
    uni_obj_vocab = list(set(sgg_obj_vocab + txt_obj_vocab))

    sgg_obj2id = {key: i+1 for i, key in enumerate(sgg_obj_vocab)}
    txt_obj2id = {key: i+1 for i, key in enumerate(txt_obj_vocab)}
    uni_obj2id = {key: i+1 for i, key in enumerate(uni_obj_vocab)}

    sgg_id2obj = {val:key for key,val in sgg_obj2id.items()}
    txt_id2obj = {val:key for key,val in txt_obj2id.items()}
    uni_id2obj = {val:key for key,val in uni_obj2id.items()}

    # generate gt scene graph
    def generate_gt_sg():
        valid = torch.LongTensor(cap_graph['vg_valids'])
        img_obj_start = torch.LongTensor(vg_data['img_to_first_box'][:])
        img_obj_end = torch.LongTensor(vg_data['img_to_last_box'][:])
        img_rel_start = torch.LongTensor(vg_data['img_to_first_rel'][:])
        img_rel_end = torch.LongTensor(vg_data['img_to_last_rel'][:])

        assert valid.shape[0] == img_obj_start.shape[0]
        assert valid.shape[0] == img_obj_end.shape[0]
        assert valid.shape[0] == img_rel_start.shape[0]
        assert valid.shape[0] == img_rel_end.shape[0]

        img_obj_labels = torch.LongTensor(vg_data['labels'][:]).view(-1)
        img_rel_pairs = torch.LongTensor(vg_data['relationships'][:])
        img_rel_labels = torch.LongTensor(vg_data['predicates'][:]).view(-1)

        img_to_sg = {}
        for i in range(valid.shape[0]):
            coco_id = cap_graph['vg_coco_ids'][i]
            if int(valid[i]) == 0:
                continue
            elif (int(img_obj_start[i]) < 0) or (int(img_rel_start[i]) < 0):
                continue
            else:
                gt_boxes = img_obj_labels[int(img_obj_start[i]) : int(img_obj_end[i]) + 1].tolist()
                gt_boxes = [vg_dict['idx_to_label'][str(i)] for i in gt_boxes]
                gt_pairs = img_rel_pairs[int(img_rel_start[i]) : int(img_rel_end[i]) + 1] - int(img_obj_start[i])
                gt_pairs = gt_pairs.tolist()
                gt_rels = img_rel_labels[int(img_rel_start[i]) : int(img_rel_end[i]) + 1].tolist()
                gt_rels = [cap_graph['idx_to_meta_predicate'][str(i)] for i in gt_rels]
                gt_triplet = [[i[0], i[1], j] for i, j in zip(gt_pairs, gt_rels)]
                img_to_sg[str(coco_id)] = [{'entities' : gt_boxes, 'relations' : gt_triplet}, ]
        return img_to_sg

    #gt_scene_graph = generate_gt_sg()

    # generate scene graph from test results
    def generate_detect_sg(det_result, det_info, valid_ids, img_coco_map, obj_thres = 0.1):
        num_img = len(det_info)
        groundtruths = det_result['groundtruths']
        predictions = det_result['predictions']
        assert len(groundtruths) == num_img
        assert len(predictions) == num_img

        output = {}
        for i in range(num_img):
            # load detect result
            image_id = det_info[i]['img_file'].split('/')[-1].split('.')[0]
            if int(image_id) not in valid_ids:
                continue
            all_obj_labels = predictions[i].get_field('pred_labels')
            all_obj_scores = predictions[i].get_field('pred_scores')
            all_rel_pairs = predictions[i].get_field('rel_pair_idxs')
            all_rel_prob = predictions[i].get_field('pred_rel_scores')
            all_rel_scores, all_rel_labels = all_rel_prob.max(-1)

            # filter objects and relationships
            all_obj_scores[all_obj_scores < obj_thres] = 0.0
            obj_mask = all_obj_scores >= obj_thres
            triplet_score = all_obj_scores[all_rel_pairs[:, 0]] * all_obj_scores[all_rel_pairs[:, 1]] * all_rel_scores
            rel_mask = ((all_rel_labels > 0) + (triplet_score > 0)) > 0

            # generate filterred result
            num_obj = obj_mask.shape[0]
            num_rel = rel_mask.shape[0]
            rel_matrix = torch.zeros((num_obj, num_obj))
            for k in range(num_rel):
                if rel_mask[k]:
                    rel_matrix[int(all_rel_pairs[k, 0]), int(all_rel_pairs[k, 1])] = all_rel_labels[k]
            rel_matrix = rel_matrix[obj_mask][:, obj_mask].long()
            filter_obj = all_obj_labels[obj_mask]
            filter_pair = torch.nonzero(rel_matrix > 0)
            filter_rel = rel_matrix[filter_pair[:, 0], filter_pair[:, 1]]

            # generate labels
            pred_objs = [vg_dict['idx_to_label'][str(i)] for i in filter_obj.tolist()]
            pred_rels = [[i[0], i[1], cap_graph['idx_to_meta_predicate'][str(j)]] for i, j in zip(filter_pair.tolist(), filter_rel.tolist())]

            coco_id = img_coco_map[int(image_id)]
            output[str(coco_id)] = [{'entities' : pred_objs, 'relations' : pred_rels}, ]
        return output


    def generate_txt_img_sg(img_sg, txt_sg):
        txt_img_sg = {}
        num_img = len(cap_graph['vg_valids'])
        for i in range(num_img):
            coco_id = str(cap_graph['vg_coco_ids'][i])
            if cap_graph['vg_valids'][i] and (coco_id in img_sg) and (coco_id in txt_sg):
                img = img_sg[coco_id]
                txt = txt_sg[coco_id]
                encode_img = {'entities':[], 'relations':[]}
                encode_txt = {'entities':[], 'relations':[]}
                for item in img:
                    entities = [sgg_obj2id[e] for e in item['entities']]
                    relations = [[entities[r[0]], entities[r[1]], sgg_rel2id[r[2]]] for r in item['relations']]
                    encode_img['entities'] = encode_img['entities'] + entities
                    encode_img['relations'] = encode_img['relations'] + relations
                for item in txt:
                    entities = [txt_obj2id[e] for e in item['entities']]
                    relations = [[entities[r[0]], entities[r[1]], txt_rel2id[r[2]]] for r in item['relations']]
                    encode_txt['entities'] = encode_txt['entities'] + entities
                    encode_txt['relations'] = encode_txt['relations'] + relations
                # ===================================================================================
                # ============================== Acknowledgement ====================================
                # ===================================================================================
                #     Since I lost part of the code when I merged several jupyter notes into this
                # preprocessing.py files, the "image_graph" and "text_graph" are missing in the
                # original preprocessing.py. Thanks to the Haeyong Kang from KAIST, he filled in
                # the missing part by the following code.
                # ===================================================================================

                # === for image_graph ============================================here
                entities = encode_img['entities']
                relations = encode_img['relations']
                if len(relations) == 0:
                    img_graph = np.zeros((len(entities), 1))
                else:
                    img_graph = np.zeros((len(entities), len(relations)))

                image_graph = []
                for i, es in enumerate(entities):
                    for j, rs in enumerate(relations):
                        if es in rs:
                            img_graph[i,j] = 1
                        else:
                            img_graph[i,j] = 0

                image_graph.append(img_graph.tolist())

                # === for text_graph =============================================here
                entities = encode_txt['entities']
                relations = encode_txt['relations']
                if len(relations) == 0:
                    txt_graph = np.zeros((len(entities), 1))
                else:
                    txt_graph = np.zeros((len(entities), len(relations)))

                text_graph = []
                for i, es in enumerate(entities):
                    for j, rs in enumerate(relations):
                        if es in rs:
                            txt_graph[i,j] = 1
                        else:
                            txt_graph[i,j] = 0

                text_graph.append(txt_graph.tolist())

                #txt_img_sg[coco_id] = {'img':encode_img, 'txt':encode_txt}
                txt_img_sg[coco_id] = {
                    'img':encode_img,
                    'image_graph':image_graph,
                    'txt':encode_txt,
                    'text_graph':text_graph}

        return txt_img_sg


    def img_coco_mapping():
        img_coco_map = {}
        for img_id, coco_id in zip(cap_graph['vg_image_ids'], cap_graph['vg_coco_ids']):
            img_coco_map[int(img_id)] = int(coco_id)
        return img_coco_map


    detected_result = torch.load(detected_path + 'eval_results.pytorch')
    detected_info = json.load(open(detected_path + 'visual_info.json'))

    img_coco = img_coco_mapping()

    valid_ids = []
    for img_id, val in zip(cap_graph['vg_image_ids'], cap_graph['vg_valids']):
        if val > 0:
            valid_ids.append(img_id)

    output = generate_detect_sg(detected_result, detected_info, valid_ids, img_coco, obj_thres = 0.1)

    # You can replace cap_graph['vg_coco_id_to_capgraphs'] by the graphs produced in
    # maskrcnn_benchmark/image_retrieval/sentence_to_graph_processing.py if you want.
    txt_img_sg = generate_txt_img_sg(output, cap_graph['vg_coco_id_to_capgraphs'])

    with open(output_path, 'w') as outfile:
        json.dump(txt_img_sg, outfile)
    print("Output file created", output_path)


# Execute this file with two paratmeters:
# --test-results-path is the path to the results file produced tools/relation_test_net.py (contains eval_results.pytorch and visual_info.json)
# --output-file-name is the name of the output file (will be created under the path given for --test-results-path )
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocessing of Scene Graphs for Image Retrieval")
    parser.add_argument(
        "--test_results_path",
        default=f"/home/users/alatif/data/ImageCorpora/vg/checkpoint/causal-motifs-sgdet/inference/VG_stanford_filtered_with_attribute/",
        help="path to config file",
    )

    parser.add_argument(
        "--output_file_name",
        default=f"sg_of_causal_sgdet_ctx_only.json",
        help="creates this file under the path specified with  --test-results-path",
    )

    args = parser.parse_args()


    preprocess_scene_graphs_output(args.test_results_path, args.output_file_name)
