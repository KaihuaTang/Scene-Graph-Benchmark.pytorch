# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch


def kl_div_loss(input, target, e=1e-9, reduction='sum'):
    assert len(input.shape) == 2
    assert len(target.shape) == 2

    log_target = (target + e).log()
    log_input =  (input + e).log()

    loss = target.detach() * (log_target.detach() - log_input)

    if reduction == 'sum':
        loss = loss.sum(-1)
    elif reduction == 'mean':
        loss = loss.mean(-1)

    return loss.mean()