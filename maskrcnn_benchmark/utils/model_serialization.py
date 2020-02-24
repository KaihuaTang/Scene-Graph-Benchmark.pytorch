# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from collections import OrderedDict
import logging

import torch

from maskrcnn_benchmark.utils.imports import import_file


def align_and_update_state_dicts(model_state_dict, loaded_state_dict, load_mapping):
    """
    Strategy: suppose that the models that we will create will have prefixes appended
    to each of its keys, for example due to an extra level of nesting that the original
    pre-trained weights from ImageNet won't contain. For example, model.state_dict()
    might return backbone[0].body.res2.conv1.weight, while the pre-trained model contains
    res2.conv1.weight. We thus want to match both parameters together.
    For that, we look for each model weight, look among all loaded keys if there is one
    that is a suffix of the current weight name, and use it if that's the case.
    If multiple matches exist, take the one with longest size
    of the corresponding name. For example, for the same model as before, the pretrained
    weight file can contain both res2.conv1.weight, as well as conv1.weight. In this case,
    we want to match backbone[0].body.conv1.weight to conv1.weight, and
    backbone[0].body.res2.conv1.weight to res2.conv1.weight.
    """
    logger = logging.getLogger(__name__)
    current_keys = sorted(list(model_state_dict.keys()))
    loaded_keys = sorted(list(loaded_state_dict.keys()))
    # get a matrix of string matches, where each (i, j) entry correspond to the size of the
    # loaded_key string, if it matches
    # NOTE: Kaihua Tang, since some modules of current model will be initialized from assigned layer of 
    # loaded model, we use load_mapping to do such operation
    mapped_current_keys = current_keys.copy()
    for i, key in enumerate(mapped_current_keys):
        for source_key, target_key in load_mapping.items():
            if source_key in key:
                mapped_current_keys[i] = key.replace(source_key, target_key)
                logger.info("MAPPING {} in current model to {} in loaded model.".format(key, mapped_current_keys[i]))

    match_matrix = [
        len(j) if i.endswith(j) else 0 for i in mapped_current_keys for j in loaded_keys
    ]
    match_matrix = torch.as_tensor(match_matrix).view(
        len(current_keys), len(loaded_keys)
    )
    max_match_size, idxs = match_matrix.max(1)
    # remove indices that correspond to no-match
    idxs[max_match_size == 0] = -1

    # used for logging
    max_size = max([len(key) for key in current_keys]) if current_keys else 1
    max_size_loaded = max([len(key) for key in loaded_keys]) if loaded_keys else 1
    log_str_template = "REMATCHING! {: <{}} loaded from {: <{}} of shape {}"
    for idx_new, idx_old in enumerate(idxs.tolist()):
        if idx_old == -1:
            key = current_keys[idx_new]
            logger.info("NO-MATCHING of current module: {} of shape {}".format(key, 
                                    tuple(model_state_dict[key].shape)))
            continue
        key = current_keys[idx_new]
        key_old = loaded_keys[idx_old]
        model_state_dict[key] = loaded_state_dict[key_old]
        # add a control gate for this logger (it's too large)
        if ((not key.startswith('module.'))  and key != key_old) or (key.startswith('module.') and key[7:] != key_old):
            logger.info(
                log_str_template.format(
                    key,
                    max_size,
                    key_old,
                    max_size_loaded,
                    tuple(loaded_state_dict[key_old].shape),
                )
            )


def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict


def load_state_dict(model, loaded_state_dict, load_mapping):
    model_state_dict = model.state_dict()
    # if the state_dict comes from a model that was wrapped in a
    # DataParallel or DistributedDataParallel during serialization,
    # remove the "module" prefix before performing the matching
    loaded_state_dict = strip_prefix_if_present(loaded_state_dict, prefix="module.")
    align_and_update_state_dicts(model_state_dict, loaded_state_dict, load_mapping)

    # use strict loading
    model.load_state_dict(model_state_dict)
