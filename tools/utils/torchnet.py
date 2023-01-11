import logging
from functools import lru_cache

import torch
import torch.nn as nn

from .device import get_default_device


@lru_cache()
def _load_model_weight_ckpt(weights):
    return torch.load(weights, map_location=get_default_device())


# model tools
def load_weights(model, weights='', gpus=1, strict=True, resume=False, just_weight=False):
    """
    load model from weights, remove "module" if weights is dataparallel
    :param model:
    :param weights:
    :param gpus:
    :return:
    """
    model_weights = _load_model_weight_ckpt(weights)
    if not just_weight:
        model_weights = model_weights['optim'] if resume else model_weights['state_dict']

    try:
        model.load_state_dict(model_weights, strict=strict)
    except:
        model.load_state_dict(_weights_rm_module(model_weights), strict=strict)
    logging.info(f'Loading {weights} success!')

    if gpus > 1:
        logging.info(f'Loading from DataParallel module, due to gpus={gpus!r} ...')
        model = nn.DataParallel(model, device_ids=[i for i in range(gpus)])
    return model


def _weights_rm_module(weights):
    w = {}
    for key, value in weights.items():
        segments = key.split('.')
        assert segments[0] == 'module'

        # remove `module.` prefix,
        # `module.HG_5.skip_conv0.res.0.weight` --> `HG_5.skip_conv0.res.0.weight`
        key = '.'.join(segments[1:])
        w[key] = value

    return w
