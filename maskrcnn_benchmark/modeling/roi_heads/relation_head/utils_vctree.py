import array
import os
import zipfile
import itertools
import six
from six.moves.urllib.request import urlretrieve
from tqdm import tqdm
import sys
from maskrcnn_benchmark.modeling.utils import cat

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def generate_forest(pair_scores, proposals, mode):
    """
    generate a list of trees that covers all the objects in a batch
    proposal.bbox: [obj_num, (x1, y1, x2, y2)]
    pair_scores: [obj_num, obj_num]
    output: list of trees, each present a chunk of overlaping objects
    """
    output_forest = []  # the list of trees, each one is a chunk of overlapping objects

    for pair_score, proposal in zip(pair_scores, proposals):
        num_obj = pair_score.shape[0]
        obj_label = proposal.get_field("labels") if mode == "predcls" else proposal.get_field("predict_logits").max(-1)[1]

        assert pair_score.shape[0] == len(proposal)
        assert pair_score.shape[0] == pair_score.shape[1]
        node_scores = pair_score.mean(1).view(-1)
        root_idx = int(node_scores.max(-1)[1])

        root = ArbitraryTree(root_idx, float(node_scores[root_idx]), int(obj_label[root_idx]), proposal.bbox[root_idx], is_root=True)

        node_container = []
        remain_index = []
        # put all nodes into node container
        for idx in list(range(num_obj)):
            if idx == root_idx:
                continue
            new_node = ArbitraryTree(idx, float(node_scores[idx]), int(obj_label[idx]),  proposal.bbox[idx])
            node_container.append(new_node)
            remain_index.append(idx)
        
        # iteratively generate tree
        gen_tree(node_container, pair_score, node_scores, root, remain_index, mode)
        output_forest.append(root)

    return output_forest

def gen_tree(node_container, pair_score, node_scores, root, remain_index, mode):
    """
    Step 1: Devide all nodes into left child container and right child container
    Step 2: From left child container and right child container, select their respective sub roots

    pair_scores: [obj_num, obj_num]
    node_scores: [obj_num]
    """
    num_nodes = len(node_container)
    device = pair_score.device
    # Step 0
    if  num_nodes == 0:
        return
    # Step 1
    select_node = []
    select_index = []
    select_node.append(root)
    select_index.append(root.index)

    while len(node_container) > 0:
        wid = len(remain_index)
        select_indexs = torch.tensor(select_index, device=device, dtype=torch.int64)
        remain_indexs = torch.tensor(remain_index, device=device, dtype=torch.int64)
        select_score_map = pair_score[select_indexs][:, remain_indexs].view(-1)
        best_id = select_score_map.max(0)[1]

        depend_id = int(best_id) // wid
        insert_id = int(best_id) % wid
        best_depend_node = select_node[depend_id]
        best_insert_node = node_container[insert_id]
        best_depend_node.add_child(best_insert_node)

        select_node.append(best_insert_node)
        select_index.append(best_insert_node.index)
        node_container.remove(best_insert_node)
        remain_index.remove(best_insert_node.index)




def arbForest_to_biForest(forest):
    """
    forest: a set of arbitrary Tree
    output: a set of corresponding binary Tree
    """
    output = []
    for i in range(len(forest)):
        result_tree = arTree_to_biTree(forest[i])
        output.append(result_tree)
        
    return output


def arTree_to_biTree(arTree):
    root_node = arTree.generate_bi_tree()
    arNode_to_biNode(arTree, root_node)

    return root_node

def arNode_to_biNode(arNode, biNode):
    if arNode.get_child_num() >= 1:
        new_bi_node = arNode.children[0].generate_bi_tree()
        biNode.add_left_child(new_bi_node)
        arNode_to_biNode(arNode.children[0], biNode.left_child)

    if arNode.get_child_num() > 1:
        current_bi_node = biNode.left_child
        for i in range(arNode.get_child_num() - 1):
            new_bi_node = arNode.children[i+1].generate_bi_tree()
            current_bi_node.add_right_child(new_bi_node)
            current_bi_node = current_bi_node.right_child
            arNode_to_biNode(arNode.children[i+1], current_bi_node)

def find_best_node(node_container):
    max_node_score = -1 
    best_node = None
    for i in range(len(node_container)):
        if node_container[i].score > max_node_score:
            max_node_score = node_container[i].score
            best_node = node_container[i]
    return best_node





class BasicBiTree(object):
    def __init__(self, idx, is_root=False):
        self.index = int(idx)
        self.is_root = is_root
        self.left_child = None
        self.right_child = None
        self.parent = None
        self.num_child = 0

    def add_left_child(self, child):
        if self.left_child is not None:
            print('Left child already exist')
            return
        child.parent = self
        self.num_child += 1
        self.left_child = child
    
    def add_right_child(self, child):
        if self.right_child is not None:
            print('Right child already exist')
            return
        child.parent = self
        self.num_child += 1
        self.right_child = child

    def get_total_child(self):
        sum = 0
        sum += self.num_child
        if self.left_child is not None:
            sum += self.left_child.get_total_child()
        if self.right_child is not None:
            sum += self.right_child.get_total_child()
        return sum

    def depth(self):
        if hasattr(self, '_depth'):
            return self._depth
        if self.parent is None:
            count = 1
        else:
            count = self.parent.depth() + 1
        self._depth = count
        return self._depth

    def max_depth(self):
        if hasattr(self, '_max_depth'):
            return self._max_depth
        count = 0
        if self.left_child is not None:
            left_depth = self.left_child.max_depth()
            if left_depth > count:
                count = left_depth
        if self.right_child is not None:
            right_depth = self.right_child.max_depth()
            if right_depth > count:
                count = right_depth
        count += 1
        self._max_depth = count
        return self._max_depth

    # by index
    def is_descendant(self, idx):
        left_flag = False
        right_flag = False
        # node is left child
        if self.left_child is not None:
            if self.left_child.index is idx:
                return True
            else:
                left_flag = self.left_child.is_descendant(idx)
        # node is right child
        if self.right_child is not None:
            if self.right_child.index is idx:
                return True
            else:
                right_flag = self.right_child.is_descendant(idx)
        # node is descendant
        if left_flag or right_flag:
            return True
        else:
            return False
            
    # whether input node is under left sub tree
    def is_left_descendant(self, idx):
        if self.left_child is not None:
            if self.left_child.index is idx:
                return True
            else:
                return self.left_child.is_descendant(idx)
        else:
            return False
    
    # whether input node is under right sub tree
    def is_right_descendant(self, idx):
        if self.right_child is not None:
            if self.right_child.index is idx:
                return True
            else:
                return self.right_child.is_descendant(idx)
        else:
            return False

    
class ArbitraryTree(object):
    def __init__(self, idx, score, label=-1, box=None, is_root=False):
        self.index = int(idx)
        self.is_root = is_root
        self.score = float(score)
        self.children = []
        self.label = label
        self.embeded_label = None
        self.box = box.view(-1) if box is not None else None #[x1,y1,x2,y2]
        self.parent = None
        self.node_order = -1 # the n_th node added to the tree
    
    def generate_bi_tree(self):
        # generate a BiTree node, parent/child relationship are not inherited
        return BiTree(self.index, self.score, self.label, self.box, self.is_root)

    def add_child(self, child):
        child.parent = self
        self.children.append(child)

    def print(self):
        print('index: ', self.index)
        print('node_order: ', self.node_order)
        print('num of child: ', len(self.children))
        for node in self.children:
            node.print()

    def find_node_by_order(self, order, result_node):
        if self.node_order == order:
            result_node = self
        elif len(self.children) > 0:
            for i in range(len(self.children)):
                result_node = self.children[i].find_node_by_order(order, result_node)
        
        return result_node
    
    def find_node_by_index(self, index, result_node):
        if self.index == index:
            result_node = self
        elif len(self.children) > 0:
            for i in range(len(self.children)):
                result_node = self.children[i].find_node_by_index(index, result_node)
                
        return result_node

    def search_best_insert(self, score_map, best_score, insert_node, best_depend_node, best_insert_node, ignore_root = True):
        if self.is_root and ignore_root:
            pass
        elif float(score_map[self.index, insert_node.index]) > float(best_score):
            best_score = score_map[self.index, insert_node.index]
            best_depend_node = self
            best_insert_node = insert_node
        
        # iteratively search child
        for i in range(self.get_child_num()):
            best_score, best_depend_node, best_insert_node = \
                self.children[i].search_best_insert(score_map, best_score, insert_node, best_depend_node, best_insert_node)

        return best_score, best_depend_node, best_insert_node

    def get_child_num(self):
        return len(self.children)
    
    def get_total_child(self):
        sum = 0
        num_current_child = self.get_child_num()
        sum += num_current_child
        for i in range(num_current_child):
            sum += self.children[i].get_total_child()
        return sum

# only support binary tree
class BiTree(BasicBiTree):
    def __init__(self, idx, node_score, label, box, is_root=False):
        super(BiTree, self).__init__(idx, is_root)
        self.state_c = None
        self.state_h = None
        self.state_c_backward = None
        self.state_h_backward = None
        # used to select node
        self.node_score = float(node_score)
        self.label = label
        self.embeded_label = None
        self.box = box.view(-1) #[x1,y1,x2,y2]



def bbox_intersection(box_a, box_b):
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy + 1.0), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def bbox_overlap(box_a, box_b):
    inter = bbox_intersection(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0] + 1.0) *
              (box_a[:, 3] - box_a[:, 1] + 1.0)).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2] - box_b[:, 0] + 1.0) *
              (box_b[:, 3] - box_b[:, 1] + 1.0)).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / (union + 1e-9)  


def bbox_area(bbox):
    area = (bbox[:,2] - bbox[:,0]) * (bbox[:,3] - bbox[:,1])
    return area.view(-1, 1)


def get_overlap_info(proposals):
    IM_SCALE = 1024
    assert proposals[0].mode == 'xyxy'
    overlap_info = []
    for proposal in proposals:
        boxes = proposal.bbox
        intersection = bbox_intersection(boxes, boxes).float()    # num, num
        overlap = bbox_overlap(boxes, boxes).float()                  # num, num
        area = bbox_area(boxes).float()                           # num, 1

        info1 = (intersection > 0.0).float().sum(1).view(-1, 1)
        info2 = intersection.sum(1).view(-1, 1) / float(IM_SCALE * IM_SCALE)
        info3 = overlap.sum(1).view(-1, 1)
        info4 = info2 / (info1 + 1e-9)
        info5 = info3 / (info1 + 1e-9)
        info6 = area / float(IM_SCALE * IM_SCALE)

        info = torch.cat([info1, info2, info3, info4, info5, info6], dim=1)
        overlap_info.append(info)

    return torch.cat(overlap_info, dim=0)