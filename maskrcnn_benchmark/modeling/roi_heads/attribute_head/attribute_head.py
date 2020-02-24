# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

from .roi_attribute_feature_extractors import make_roi_attribute_feature_extractor
from .roi_attribute_predictors import make_roi_attribute_predictor
from .loss import make_roi_attribute_loss_evaluator

def add_attribute_logits(proposals, attri_logits):
    slice_idxs = [0]
    for i in range(len(proposals)):
        slice_idxs.append(len(proposals[i])+slice_idxs[-1])
        proposals[i].add_field("attribute_logits", attri_logits[slice_idxs[i]:slice_idxs[i+1]])
    return proposals

class ROIAttributeHead(torch.nn.Module):
    """
    Generic ATTRIBUTE Head class.
    """

    def __init__(self, cfg, in_channels):
        super(ROIAttributeHead, self).__init__()
        self.cfg = cfg.clone()
        self.feature_extractor = make_roi_attribute_feature_extractor(cfg, in_channels, half_out=self.cfg.MODEL.ATTRIBUTE_ON)
        self.predictor = make_roi_attribute_predictor(cfg, self.feature_extractor.out_channels)
        self.loss_evaluator = make_roi_attribute_loss_evaluator(cfg)

    def forward(self, features, proposals, targets=None):
        """
        features:  extracted from box_head
        """
        # Attribute head is fixed when we train the relation head
        if self.cfg.MODEL.RELATION_ON:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX and self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                # mode==predcls
                # no need to predict attribute, get grond truth
                x = self.feature_extractor(features, proposals)
                return x, proposals, {}
            # mode==sgcls  or sgdet
            else:
                x = self.feature_extractor(features, proposals)
                attri_logits = self.predictor(x)
                assert sum([len(p) for p in proposals]) == attri_logits.shape[0]
                proposals = add_attribute_logits(proposals, attri_logits)
                return x, proposals, {}
            
        # Train/Test the attribute head
        x = self.feature_extractor(features, proposals)
        attri_logits = self.predictor(x)
        assert sum([len(p) for p in proposals]) == attri_logits.shape[0]
        proposals = add_attribute_logits(proposals, attri_logits)
        
        if not self.training:
            return x, proposals, {}

        # proposals need to contain the attributes fields
        loss_attribute = self.loss_evaluator(proposals, attri_logits)
        return x, proposals, dict(loss_attribute=loss_attribute)

def build_roi_attribute_head(cfg, in_channels):
    """
    Constructs a new attribute head.
    By default, uses ROIAttributeHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIAttributeHead(cfg, in_channels)
