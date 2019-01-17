SUDOKU_PATH = '/home/ajhnam/sudoku'

import os
import sys
sys.path.append(SUDOKU_PATH + '/src/sudoku')
sys.path.append(SUDOKU_PATH + '/src/misc')
sys.path.append(SUDOKU_PATH + '/src/models')

from dataset import Datasets
from rrn_utils import train_rrn

train_size_per_num_hints = 1200 # * 3 = 3600
valid_size_per_num_hints = 112 # * 3 = 336

hyperparameters = {
    'device': 7,
    'dim_x': 2,
    'dim_y': 2,
    'num_iters': 32,
    'train_size': train_size_per_num_hints*3,
    'valid_size': valid_size_per_num_hints*3,
    'batch_size': 400,
    'epochs': 600,
    'save_epochs': 25,
    'embed_size': 6,
    'hidden_layer_size': 32,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4
}

dataset = Datasets.load('./data/train_datasets.pkl')
ext_valid_datasets = Datasets.load('./data/ext_valid_datasets.pkl')

split_inputs, split_outputs = dataset.split_data([train_size_per_num_hints,
                                                  train_size_per_num_hints+valid_size_per_num_hints])
extrapolation_inputs, extrapolation_outputs = ext_valid_datasets.split_data([valid_size_per_num_hints])

train_rrn(hyperparameters,
          train_inputs=split_inputs[0],
          train_outputs=split_outputs[0],
          other_inputs={"interpolation": split_inputs[1], "extrapolation": extrapolation_inputs[0]},
          other_outputs={"interpolation": split_outputs[1], "extrapolation": extrapolation_outputs[0]})

