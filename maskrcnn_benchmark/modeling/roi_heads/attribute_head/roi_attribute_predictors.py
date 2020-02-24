# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from maskrcnn_benchmark.modeling import registry
from torch import nn


@registry.ROI_ATTRIBUTE_PREDICTOR.register("FastRCNNPredictor")
class FastRCNNPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(FastRCNNPredictor, self).__init__()
        assert in_channels is not None
        num_inputs = in_channels

        num_attributes = cfg.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.att_score = nn.Linear(num_inputs, num_attributes)

        nn.init.normal_(self.att_score.weight, mean=0, std=0.01)
        nn.init.constant_(self.att_score.bias, 0)

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        att_logit = self.att_score(x)

        return att_logit


@registry.ROI_ATTRIBUTE_PREDICTOR.register("FPNPredictor")
class FPNPredictor(nn.Module):
    def __init__(self, cfg, in_channels):
        super(FPNPredictor, self).__init__()
        num_attributes = cfg.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        representation_size = in_channels

        self.att_score = nn.Linear(representation_size, num_attributes)

        nn.init.normal_(self.att_score.weight, std=0.01)
        nn.init.constant_(self.att_score.bias, 0)

    def forward(self, x):
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
            x = x.view(x.size(0), -1)

        att_logit = self.att_score(x)

        return att_logit


def make_roi_attribute_predictor(cfg, in_channels):
    func = registry.ROI_ATTRIBUTE_PREDICTOR[cfg.MODEL.ROI_ATTRIBUTE_HEAD.PREDICTOR]
    return func(cfg, in_channels)
