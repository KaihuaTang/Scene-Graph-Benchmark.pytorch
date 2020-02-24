# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch.nn import functional as F

from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.balanced_positive_negative_sampler import (
    BalancedPositiveNegativeSampler
)
from maskrcnn_benchmark.modeling.utils import cat


class FastRCNNLossComputation(object):
    """
    Computes the loss for Faster R-CNN.
    Also supports FPN
    """

    def __init__(self, cls_agnostic_bbox_reg=False):
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg

    def assign_label_to_proposals(self, proposals, targets):
        for img_idx, (target, proposal) in enumerate(zip(targets, proposals)):
            match_quality_matrix = boxlist_iou(target, proposal)
            matched_idxs = self.proposal_matcher(match_quality_matrix)
            # Fast RCNN only need "labels" field for selecting the targets
            target = target.copy_with_fields(["labels", "attributes"])
            matched_targets = target[matched_idxs.clamp(min=0)]
            
            labels_per_image = matched_targets.get_field("labels").to(dtype=torch.int64)
            attris_per_image = matched_targets.get_field("attributes").to(dtype=torch.int64)

            labels_per_image[matched_idxs < 0] = 0
            attris_per_image[matched_idxs < 0, :] = 0
            proposals[img_idx].add_field("labels", labels_per_image)
            proposals[img_idx].add_field("attributes", attris_per_image)
        return proposals


    def __call__(self, class_logits, box_regression, proposals):
        """
        Computes the loss for Faster R-CNN.
        This requires that the subsample method has been called beforehand.

        Arguments:
            class_logits (list[Tensor])
            box_regression (list[Tensor])
            proposals (list[BoxList])

        Returns:
            classification_loss (Tensor)
            box_loss (Tensor)
        """

        class_logits = cat(class_logits, dim=0)
        box_regression = cat(box_regression, dim=0)
        device = class_logits.device

        labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        regression_targets = cat([proposal.get_field("regression_targets") for proposal in proposals], dim=0)

        classification_loss = F.cross_entropy(class_logits, labels.long())

        # get indices that correspond to the regression targets for
        # the corresponding ground truth labels, to be used with
        # advanced indexing
        sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
        labels_pos = labels[sampled_pos_inds_subset]
        if self.cls_agnostic_bbox_reg:
            map_inds = torch.tensor([4, 5, 6, 7], device=device)
        else:
            map_inds = 4 * labels_pos[:, None] + torch.tensor([0, 1, 2, 3], device=device)

        box_loss = smooth_l1_loss(
            box_regression[sampled_pos_inds_subset[:, None], map_inds],
            regression_targets[sampled_pos_inds_subset],
            size_average=False,
            beta=1,
        )
        box_loss = box_loss / labels.numel()

        return classification_loss, box_loss


def make_roi_box_loss_evaluator(cfg):
    cls_agnostic_bbox_reg = cfg.MODEL.CLS_AGNOSTIC_BBOX_REG

    loss_evaluator = FastRCNNLossComputation(cls_agnostic_bbox_reg)

    return loss_evaluator
