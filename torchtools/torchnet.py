import sys
from collections import OrderedDict

import torch
import torch.nn as nn

from .device import get_default_device


# model tools
def load_weights(model, weights='', gpus=1, strict=True, resume=False, just_weight=False):
    """
    load model from weights, remove "module" if weights is dataparallel
    :param model:
    :param weights:
    :param gpus:
    :return:
    """
    model_weights = torch.load(weights, map_location=get_default_device())
    if not just_weight:
        model_weights = model_weights['optim'] if resume else model_weights['state_dict']

    try:
        model.load_state_dict(model_weights, strict=strict)
    except:
        print('Loading from DataParallel module......')
        model = _rm_module(model, model_weights)
    print('Loading %s success.....' % weights)

    if gpus > 1:
        model = nn.DataParallel(model, device_ids=[i for i in range(gpus)])
    sys.stdout.flush()
    return model


def _rm_module(model, weights):
    new_state_dict = OrderedDict()
    for k, v in weights.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    return model
