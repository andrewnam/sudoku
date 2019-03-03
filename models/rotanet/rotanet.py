import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import pickle
from sudoku.dataset import Datasets

# import sudoku.grid_string as grid_string

project_path = '/Users/andrew/Desktop/sudoku'
os.chdir(project_path)

data_path = 'models/rotanet/data/datasets.pkl'
dataset = Datasets.load(data_path)
