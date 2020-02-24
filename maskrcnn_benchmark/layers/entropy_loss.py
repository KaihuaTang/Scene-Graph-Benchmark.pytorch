# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch


def entropy_loss(input, e=1e-9, reduction='sum'):
    assert len(input.shape) == 2
    loss = - (input * (input + e).log())

    if reduction == 'sum':
        loss = loss.sum(-1)
    elif reduction == 'mean':
        loss = loss.mean(-1)

    return loss.mean()