# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from .utils_relation import obj_prediction_nms

class PostProcessor(nn.Module):
    """
    From a set of classification scores, box regression and proposals,
    computes the post-processed boxes, and applies NMS to obtain the
    final results
    """

    def __init__(
        self,
        attribute_on,
        use_gt_box=False,
        later_nms_pred_thres=0.3,
    ):
        """
        Arguments:

        """
        super(PostProcessor, self).__init__()
        self.attribute_on = attribute_on
        self.use_gt_box = use_gt_box
        self.later_nms_pred_thres = later_nms_pred_thres

    def forward(self, x, rel_pair_idxs, boxes):
        """
        Arguments:
            x (tuple[tensor, tensor]): x contains the relation logits
                and finetuned object logits from the relation model.
            rel_pair_idxs （list[tensor]): subject and object indice of each relation,
                the size of tensor is (num_rel, 2)
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for ech image

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra fields labels and scores
        """
        relation_logits, refine_logits = x
        
        if self.attribute_on:
            if isinstance(refine_logits[0], (list, tuple)):
                finetune_obj_logits, finetune_att_logits = refine_logits
            else:
                # just use attribute feature, do not actually predict attribute
                self.attribute_on = False
                finetune_obj_logits = refine_logits
        else:
            finetune_obj_logits = refine_logits

        results = []
        for i, (rel_logit, obj_logit, rel_pair_idx, box) in enumerate(zip(
            relation_logits, finetune_obj_logits, rel_pair_idxs, boxes
        )):
            if self.attribute_on:
                att_logit = finetune_att_logits[i]
                att_prob = torch.sigmoid(att_logit)
            obj_class_prob = F.softmax(obj_logit, -1)
            obj_class_prob[:, 0] = 0  # set background score to 0
            num_obj_bbox = obj_class_prob.shape[0]
            num_obj_class = obj_class_prob.shape[1]

            if self.use_gt_box:
                obj_scores, obj_pred = obj_class_prob[:, 1:].max(dim=1)
                obj_pred = obj_pred + 1
            else:
                # NOTE: by kaihua, apply late nms for object prediction
                obj_pred = obj_prediction_nms(box.get_field('boxes_per_cls'), obj_logit, self.later_nms_pred_thres)
                obj_score_ind = torch.arange(num_obj_bbox, device=obj_logit.device) * num_obj_class + obj_pred
                obj_scores = obj_class_prob.view(-1)[obj_score_ind]
            
            assert obj_scores.shape[0] == num_obj_bbox
            obj_class = obj_pred

            if self.use_gt_box:
                boxlist = box
            else:
                # mode==sgdet
                # apply regression based on finetuned object class
                device = obj_class.device
                batch_size = obj_class.shape[0]
                regressed_box_idxs = obj_class
                boxlist = BoxList(box.get_field('boxes_per_cls')[torch.arange(batch_size, device=device), regressed_box_idxs], box.size, 'xyxy')
            boxlist.add_field('pred_labels', obj_class) # (#obj, )
            boxlist.add_field('pred_scores', obj_scores) # (#obj, )

            if self.attribute_on:
                boxlist.add_field('pred_attributes', att_prob)
            
            # sorting triples according to score production
            obj_scores0 = obj_scores[rel_pair_idx[:, 0]]
            obj_scores1 = obj_scores[rel_pair_idx[:, 1]]
            rel_class_prob = F.softmax(rel_logit, -1)
            rel_scores, rel_class = rel_class_prob[:, 1:].max(dim=1)
            rel_class = rel_class + 1
            # TODO Kaihua: how about using weighted some here?  e.g. rel*1 + obj *0.8 + obj*0.8
            triple_scores = rel_scores * obj_scores0 * obj_scores1
            _, sorting_idx = torch.sort(triple_scores.view(-1), dim=0, descending=True)
            rel_pair_idx = rel_pair_idx[sorting_idx]
            rel_class_prob = rel_class_prob[sorting_idx]
            rel_labels = rel_class[sorting_idx]

            boxlist.add_field('rel_pair_idxs', rel_pair_idx) # (#rel, 2)
            boxlist.add_field('pred_rel_scores', rel_class_prob) # (#rel, #rel_class)
            boxlist.add_field('pred_rel_labels', rel_labels) # (#rel, )
            # should have fields : rel_pair_idxs, pred_rel_class_prob, pred_rel_labels, pred_labels, pred_scores
            # Note
            # TODO Kaihua: add a new type of element, which can have different length with boxlist (similar to field, except that once 
            # the boxlist has such an element, the slicing operation should be forbidden.)
            # it is not safe to add fields about relation into boxlist!
            results.append(boxlist)
        return results


def make_roi_relation_post_processor(cfg):
    attribute_on = cfg.MODEL.ATTRIBUTE_ON
    use_gt_box = cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX
    later_nms_pred_thres = cfg.TEST.RELATION.LATER_NMS_PREDICTION_THRES

    postprocessor = PostProcessor(
        attribute_on,
        use_gt_box,
        later_nms_pred_thres,
    )
    return postprocessor
