import torch


def get_default_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
