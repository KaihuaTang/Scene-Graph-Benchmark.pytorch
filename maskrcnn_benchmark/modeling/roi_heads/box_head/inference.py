# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.modeling.box_coder import BoxCoder


class PostProcessor(nn.Module):
    """
    From a set of classification scores, box regression and proposals,
    computes the post-processed boxes, and applies NMS to obtain the
    final results
    """

    def __init__(
        self,
        score_thresh=0.05,
        nms=0.5,
        post_nms_per_cls_topn=300,
        nms_filter_duplicates=True,
        detections_per_img=100,
        box_coder=None,
        cls_agnostic_bbox_reg=False,
        bbox_aug_enabled=False,
        save_proposals=False,
        custum_eval=False
    ):
        """
        Arguments:
            score_thresh (float)
            nms (float)
            detections_per_img (int)
            box_coder (BoxCoder)
        """
        super(PostProcessor, self).__init__()
        self.score_thresh = score_thresh
        self.nms = nms
        self.post_nms_per_cls_topn = post_nms_per_cls_topn
        self.nms_filter_duplicates = nms_filter_duplicates
        self.detections_per_img = detections_per_img
        if box_coder is None:
            box_coder = BoxCoder(weights=(10., 10., 5., 5.))
        self.box_coder = box_coder
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg
        self.bbox_aug_enabled = bbox_aug_enabled
        self.save_proposals = save_proposals
        self.custum_eval = custum_eval

    def forward(self, x, boxes, relation_mode=False):
        """
        Arguments:
            x (tuple[tensor, tensor]): x contains the class logits
                and the box_regression from the model.
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for ech image

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra fields labels and scores
        """
        features, class_logits, box_regression = x
        class_prob = F.softmax(class_logits, -1)

        # TODO think about a representation of batch of boxes
        image_shapes = [box.size for box in boxes]
        boxes_per_image = [len(box) for box in boxes]
        concat_boxes = torch.cat([a.bbox for a in boxes], dim=0)

        if self.cls_agnostic_bbox_reg:
            box_regression = box_regression[:, -4:]
        # add rpn regression offset to the original proposals
        proposals = self.box_coder.decode(
            box_regression.view(sum(boxes_per_image), -1), concat_boxes
        ) # tensor of size (num_box, 4*num_cls)
        if self.cls_agnostic_bbox_reg:
            proposals = proposals.repeat(1, class_prob.shape[1])

        num_classes = class_prob.shape[1]

        features = features.split(boxes_per_image, dim=0)
        proposals = proposals.split(boxes_per_image, dim=0)
        class_prob = class_prob.split(boxes_per_image, dim=0)

        results = []
        nms_features = []
        for i, (prob, boxes_per_img, image_shape) in enumerate(zip(
            class_prob, proposals, image_shapes
        )):
            boxlist = self.prepare_boxlist(boxes_per_img, prob, image_shape)
            boxlist = boxlist.clip_to_image(remove_empty=False)
            assert self.bbox_aug_enabled == False
            if not self.bbox_aug_enabled:  # If bbox aug is enabled, we will do it later
                boxlist, orig_inds, boxes_per_cls = self.filter_results(boxlist, num_classes)
            # add 
            boxlist = self.add_important_fields(i, boxes, orig_inds, boxlist, boxes_per_cls, relation_mode)
            
            results.append(boxlist)
            nms_features.append(features[i][orig_inds])
        
        nms_features = torch.cat(nms_features, dim=0)
        return nms_features, results

    def add_important_fields(self, i, boxes, orig_inds, boxlist, boxes_per_cls, relation_mode=False):
        if relation_mode:
            if not self.custum_eval:
                gt_labels = boxes[i].get_field('labels')[orig_inds]
                gt_attributes = boxes[i].get_field('attributes')[orig_inds]
        
                boxlist.add_field('labels', gt_labels)
                boxlist.add_field('attributes', gt_attributes)

            predict_logits = boxes[i].get_field('predict_logits')[orig_inds]
            boxlist.add_field('boxes_per_cls', boxes_per_cls)
            boxlist.add_field('predict_logits', predict_logits)

        return boxlist

    # discarded by kaihua
    def jiaxin_undo_regression(self, i, boxes, orig_inds, boxlist, boxes_per_img):
        # by Jiaxin
        selected_boxes = boxes[i][orig_inds]
        # replace bbox after regression with original bbox before regression
        boxlist.bbox = selected_boxes.bbox
        # add maintain fields
        for field_name in boxes[i].extra_fields.keys():
            if field_name not in boxes[i].triplet_extra_fields:
                boxlist.add_field(field_name, selected_boxes.get_field(field_name))
        # replace background bbox after regression with bbox before regression
        boxes_per_cls = torch.cat((
            boxlist.bbox, boxes_per_img[orig_inds][:,4:]), dim=1).view(len(boxlist), num_classes, 4) # tensor of size (#nms, #cls, 4) mode=xyxy
        boxlist.add_field('boxes_per_cls', boxes_per_cls) # will be used in motif predictor
        return boxlist

    def prepare_boxlist(self, boxes, scores, image_shape):
        """
        Returns BoxList from `boxes` and adds probability scores information
        as an extra field
        `boxes` has shape (#detections, 4 * #classes), where each row represents
        a list of predicted bounding boxes for each of the object classes in the
        dataset (including the background class). The detections in each row
        originate from the same object proposal.
        `scores` has shape (#detection, #classes), where each row represents a list
        of object detection confidence scores for each of the object classes in the
        dataset (including the background class). `scores[i, j]`` corresponds to the
        box at `boxes[i, j * 4:(j + 1) * 4]`.
        """
        boxes = boxes.reshape(-1, 4)
        scores = scores.reshape(-1)
        boxlist = BoxList(boxes, image_shape, mode="xyxy")
        boxlist.add_field("pred_scores", scores)
        return boxlist

    def filter_results(self, boxlist, num_classes):
        """Returns bounding-box detection results by thresholding on scores and
        applying non-maximum suppression (NMS).
        """
        # unwrap the boxlist to avoid additional overhead.
        # if we had multi-class NMS, we could perform this directly on the boxlist
        boxes = boxlist.bbox.reshape(-1, num_classes * 4)
        boxes_per_cls = boxlist.bbox.reshape(-1, num_classes, 4)
        scores = boxlist.get_field("pred_scores").reshape(-1, num_classes)

        device = scores.device
        result = []
        orig_inds = []
        # Apply threshold on detection probabilities and apply NMS
        # Skip j = 0, because it's the background class
        inds_all = scores > self.score_thresh
        for j in range(1, num_classes):
            inds = inds_all[:, j].nonzero().squeeze(1)
            scores_j = scores[inds, j]
            boxes_j = boxes[inds, j * 4 : (j + 1) * 4]
            boxlist_for_class = BoxList(boxes_j, boxlist.size, mode="xyxy")
            boxlist_for_class.add_field("pred_scores", scores_j)
            boxlist_for_class, keep = boxlist_nms(
                boxlist_for_class, self.nms, max_proposals=self.post_nms_per_cls_topn, score_field='pred_scores'
            )
            inds = inds[keep]
            num_labels = len(boxlist_for_class)
            boxlist_for_class.add_field(
                "pred_labels", torch.full((num_labels,), j, dtype=torch.int64, device=device)
            )
            result.append(boxlist_for_class)
            orig_inds.append(inds)

        #NOTE: kaihua, according to Neural-MOTIFS (and my experiments, we need remove duplicate bbox)
        if self.nms_filter_duplicates or self.save_proposals:
            assert len(orig_inds) == (num_classes - 1)
            # set all bg to zero
            inds_all[:, 0] = 0 
            for j in range(1, num_classes):
                inds_all[:, j] = 0
                orig_idx = orig_inds[j-1]
                inds_all[orig_idx, j] = 1
            dist_scores = scores * inds_all.float()
            scores_pre, labels_pre = dist_scores.max(1)
            final_inds = scores_pre.nonzero()
            assert final_inds.dim() != 0
            final_inds = final_inds.squeeze(1)

            scores_pre = scores_pre[final_inds]
            labels_pre = labels_pre[final_inds]

            result = BoxList(boxes_per_cls[final_inds, labels_pre], boxlist.size, mode="xyxy")
            result.add_field("pred_scores", scores_pre)
            result.add_field("pred_labels", labels_pre)
            orig_inds = final_inds
        else:
            result = cat_boxlist(result)
            orig_inds = torch.cat(orig_inds, dim=0)
        
        number_of_detections = len(result)
        # Limit to max_per_image detections **over all classes**
        if number_of_detections > self.detections_per_img > 0:
            cls_scores = result.get_field("pred_scores")
            image_thresh, _ = torch.kthvalue(
                cls_scores.cpu(), number_of_detections - self.detections_per_img + 1
            )
            keep = cls_scores >= image_thresh.item()
            keep = torch.nonzero(keep).squeeze(1)
            result = result[keep]
            orig_inds = orig_inds[keep]
        return result, orig_inds, boxes_per_cls[orig_inds]


def make_roi_box_post_processor(cfg):
    use_fpn = cfg.MODEL.ROI_HEADS.USE_FPN

    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = BoxCoder(weights=bbox_reg_weights)

    score_thresh = cfg.MODEL.ROI_HEADS.SCORE_THRESH
    nms_thresh = cfg.MODEL.ROI_HEADS.NMS
    detections_per_img = cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMG
    cls_agnostic_bbox_reg = cfg.MODEL.CLS_AGNOSTIC_BBOX_REG
    bbox_aug_enabled = cfg.TEST.BBOX_AUG.ENABLED
    post_nms_per_cls_topn = cfg.MODEL.ROI_HEADS.POST_NMS_PER_CLS_TOPN
    nms_filter_duplicates = cfg.MODEL.ROI_HEADS.NMS_FILTER_DUPLICATES
    save_proposals = cfg.TEST.SAVE_PROPOSALS
    custum_eval = cfg.TEST.CUSTUM_EVAL

    postprocessor = PostProcessor(
        score_thresh,
        nms_thresh,
        post_nms_per_cls_topn,
        nms_filter_duplicates,
        detections_per_img,
        box_coder,
        cls_agnostic_bbox_reg,
        bbox_aug_enabled,
        save_proposals,
        custum_eval
    )
    return postprocessor
