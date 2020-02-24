# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from .lr_scheduler import WarmupMultiStepLR, WarmupReduceLROnPlateau


def make_optimizer(cfg, model, logger, slow_heads=None, slow_ratio=5.0, rl_factor=1.0):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        if slow_heads is not None:
            for item in slow_heads:
                if item in key:
                    logger.info("SLOW HEADS: {} is slow down by ratio of {}.".format(key, str(slow_ratio)))
                    lr = lr / slow_ratio
                    break
        params += [{"params": [value], "lr": lr * rl_factor, "weight_decay": weight_decay}]

    optimizer = torch.optim.SGD(params, lr=cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM)
    return optimizer


def make_lr_scheduler(cfg, optimizer, logger=None):
    if cfg.SOLVER.SCHEDULE.TYPE == "WarmupMultiStepLR":
        return WarmupMultiStepLR(
            optimizer,
            cfg.SOLVER.STEPS,
            cfg.SOLVER.GAMMA,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
        )
    
    elif cfg.SOLVER.SCHEDULE.TYPE == "WarmupReduceLROnPlateau":
        return WarmupReduceLROnPlateau(
            optimizer,
            cfg.SOLVER.SCHEDULE.FACTOR,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
            patience=cfg.SOLVER.SCHEDULE.PATIENCE,
            threshold=cfg.SOLVER.SCHEDULE.THRESHOLD,
            cooldown=cfg.SOLVER.SCHEDULE.COOLDOWN,
            logger=logger,
        )
    
    else:
        raise ValueError("Invalid Schedule Type")
