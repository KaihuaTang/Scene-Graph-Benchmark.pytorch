# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# from ._utils import _C
from maskrcnn_benchmark import _C

try:
    from torchvision.ops import nms
except:
    nms = _C.nms


# nms.__doc__ = """
# This function performs Non-maximum suppresion"""
