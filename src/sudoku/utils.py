import numpy as np
from datetime import datetime
import torch.cuda as cutorch
import random
import torch

# set random seed to 0
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.set_default_tensor_type('torch.DoubleTensor')

_print = print


def flatten(lst):
    return [item for sublist in lst for item in sublist]


def get_combinations(a, b):
    return np.array(np.meshgrid(a, b)).T.reshape(-1, 2)


def one_hot_encode(x):
    encoding = torch.zeros(x.shape + (torch.max(x)+1, ))
    if x.is_cuda:
        encoding = encoding.cuda(x.get_device())
    dim = len(x.shape)
    x = x.view(x.shape + (1, ))
    return encoding.scatter_(dim, x, 1)


def puzzle_as_dist(x):
    x = one_hot_encode(x)

    # to account for arbitrary dimensionality
    dims = len(x.shape)
    order = [dims - 1] + list(range(dims - 1))
    x = x.permute(order)[1:]
    order = list(range(1, dims)) + [0]
    x = x.permute(order)

    x = x.numpy()
    p = np.ones(x.shape[-1]) / x.shape[-1]
    x = np.apply_along_axis(lambda a: a if a.any() else p, len(x.shape) - 1, x)
    return torch.tensor(x)

