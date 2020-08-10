# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

from .roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_box_predictors import make_roi_box_predictor
from .inference import make_roi_box_post_processor
from .loss import make_roi_box_loss_evaluator
from .sampling import make_roi_box_samp_processor

def add_predict_logits(proposals, class_logits):
    slice_idxs = [0]
    for i in range(len(proposals)):
        slice_idxs.append(len(proposals[i])+slice_idxs[-1])
        proposals[i].add_field("predict_logits", class_logits[slice_idxs[i]:slice_idxs[i+1]])
    return proposals

class ROIBoxHead(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg, in_channels):
        super(ROIBoxHead, self).__init__()
        self.cfg = cfg.clone()
        self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels, half_out=self.cfg.MODEL.ATTRIBUTE_ON)
        self.predictor = make_roi_box_predictor(
            cfg, self.feature_extractor.out_channels)
        self.post_processor = make_roi_box_post_processor(cfg)
        self.loss_evaluator = make_roi_box_loss_evaluator(cfg)
        self.samp_processor = make_roi_box_samp_processor(cfg)

    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """
        ###################################################################
        # box head specifically for relation prediction model
        ###################################################################
        if self.cfg.MODEL.RELATION_ON:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
                # use ground truth box as proposals
                proposals = [target.copy_with_fields(["labels", "attributes"]) for target in targets]
                x = self.feature_extractor(features, proposals)
                if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                    # mode==predcls
                    # return gt proposals and no loss even during training
                    return x, proposals, {}
                else:
                    # mode==sgcls
                    # add field:class_logits into gt proposals, note field:labels is still gt
                    class_logits, _ = self.predictor(x)
                    proposals = add_predict_logits(proposals, class_logits)
                    return x, proposals, {}
            else:
                # mode==sgdet
                if self.training or not self.cfg.TEST.CUSTUM_EVAL:
                    proposals = self.samp_processor.assign_label_to_proposals(proposals, targets)
                x = self.feature_extractor(features, proposals)
                class_logits, box_regression = self.predictor(x)
                proposals = add_predict_logits(proposals, class_logits)
                # post process:
                # filter proposals using nms, keep original bbox, add a field 'boxes_per_cls' of size (#nms, #cls, 4)
                x, result = self.post_processor((x, class_logits, box_regression), proposals, relation_mode=True)
                # note x is not matched with processed_proposals, so sharing x is not permitted
                return x, result, {}

        #####################################################################
        # Original box head (relation_on = False)
        #####################################################################
        if self.training:
            # Faster R-CNN subsamples during training the proposals with a fixed
            # positive / negative ratio
            with torch.no_grad():
                proposals = self.samp_processor.subsample(proposals, targets)

        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        x = self.feature_extractor(features, proposals)
        # final classifier that converts the features into predictions
        class_logits, box_regression = self.predictor(x)
        
        if not self.training:
            x, result = self.post_processor((x, class_logits, box_regression), proposals)

            # if we want to save the proposals, we need sort them by confidence first.
            if self.cfg.TEST.SAVE_PROPOSALS:
                _, sort_ind = result.get_field("pred_scores").view(-1).sort(dim=0, descending=True)
                x = x[sort_ind]
                result = result[sort_ind]
                result.add_field("features", x.cpu().numpy())

            return x, result, {}

        loss_classifier, loss_box_reg = self.loss_evaluator([class_logits], [box_regression], proposals)

        return x, proposals, dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg)

def build_roi_box_head(cfg, in_channels):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIBoxHead(cfg, in_channels)
