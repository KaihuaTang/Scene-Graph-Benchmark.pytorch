# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch.nn import functional as F

from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.utils import cat


class AttributeHeadLossComputation(object):
    """
    Computes the loss for attribute head
    """

    def __init__(
        self,
        loss_weight=0.1,
        num_attri_cat=201,
        max_num_attri=10,
        attribute_sampling=True,
        attribute_bgfg_ratio=5,
        use_binary_loss=True,
        pos_weight=1,
    ):
        self.loss_weight = loss_weight
        self.num_attri_cat = num_attri_cat
        self.max_num_attri = max_num_attri
        self.attribute_sampling = attribute_sampling
        self.attribute_bgfg_ratio = attribute_bgfg_ratio
        self.use_binary_loss = use_binary_loss
        self.pos_weight = pos_weight

    def __call__(self, proposals, attri_logits):
        """
        Calculcate attribute loss
        """
        attributes = cat([proposal.get_field("attributes") for proposal in proposals], dim=0)
        assert attributes.shape[0] == attri_logits.shape[0]

        # generate attribute targets
        attribute_targets, selected_idxs = self.generate_attributes_target(attributes)

        attri_logits = attri_logits[selected_idxs]
        attribute_targets = attribute_targets[selected_idxs]

        attribute_loss = self.attribute_loss(attri_logits, attribute_targets)

        return attribute_loss * self.loss_weight

    
    def generate_attributes_target(self, attributes):
        """
        from list of attribute indexs to [1,0,1,0,0,1] form
        """
        assert self.max_num_attri == attributes.shape[1]
        num_obj = attributes.shape[0]

        with_attri_idx = (attributes.sum(-1) > 0).long()
        without_attri_idx = 1 - with_attri_idx
        num_pos = int(with_attri_idx.sum())
        num_neg = int(without_attri_idx.sum())
        assert num_pos + num_neg == num_obj
        
        if self.attribute_sampling:
            num_neg = min(num_neg, num_pos * self.attribute_bgfg_ratio) if num_pos > 0 else 1

        attribute_targets = torch.zeros((num_obj, self.num_attri_cat), device=attributes.device).float()
        if not self.use_binary_loss:
            attribute_targets[without_attri_idx > 0, 0] = 1.0

        pos_idxs = torch.nonzero(with_attri_idx).squeeze(1)
        perm = torch.randperm(num_obj - num_pos, device=attributes.device)[:num_neg]
        neg_idxs = torch.nonzero(without_attri_idx).squeeze(1)[perm]
        selected_idxs = torch.cat((pos_idxs, neg_idxs), dim=0)
        assert selected_idxs.shape[0] == num_neg + num_pos

        for idx in torch.nonzero(with_attri_idx).squeeze(1).tolist():
            for k in range(self.max_num_attri):
                att_id = int(attributes[idx, k])
                if att_id == 0:
                    break
                else:
                    attribute_targets[idx, att_id] = 1

        return attribute_targets, selected_idxs

    def attribute_loss(self, logits, labels):
        if self.use_binary_loss:
            all_loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=torch.FloatTensor([self.pos_weight] * self.num_attri_cat).cuda())
            return all_loss 
        else:
            # soft cross entropy
            # cross entropy attribute deteriorate the box head, even with 0.1 weight (although buttom-up top-down use cross entropy attribute)
            all_loss = -F.softmax(logits, dim=-1).log()
            all_loss = (all_loss * labels).sum(-1) / labels.sum(-1)
            return all_loss.mean()


def make_roi_attribute_loss_evaluator(cfg):
    loss_evaluator = AttributeHeadLossComputation(
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.ATTRIBUTE_LOSS_WEIGHT,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.MAX_ATTRIBUTES,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.ATTRIBUTE_BGFG_SAMPLE,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.ATTRIBUTE_BGFG_RATIO,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.USE_BINARY_LOSS,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.POS_WEIGHT,
    )

    return loss_evaluator
