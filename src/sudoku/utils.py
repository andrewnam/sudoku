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


def datetime_to_str(datetime):
    return datetime.strftime("%Y-%m-%d_%H:%M:%S")

def print(s):
    _print("({}) {}".format(datetime_to_str(datetime.now()), s))

def count_model_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_tensor_memory_size(tensor):
    return tensor.element_size() * tensor.nelement() // (2**20)

def get_gpu_memory(device):
    return cutorch.memory_allocated(device) // (2**20)
