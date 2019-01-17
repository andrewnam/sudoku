"""
Standard RRN on 1200 train size and 60 valid size
Embedding dimensions = 4
"""

SUDOKU_PATH = '/home/ajhnam/sudoku'

import sys
sys.path.append(SUDOKU_PATH + '/src/sudoku')
sys.path.append(SUDOKU_PATH + '/src/misc')
sys.path.append(SUDOKU_PATH + '/src/models')

from dataset import Datasets
import rrn_utils


train_size_per_num_hints = 100 # * 12 = 1200
valid_size_per_num_hints = 5 # * 12 = 60

hyperparameters = {
    'device': 8,
    'dim_x': 2, 
    'dim_y': 2,
    'num_iters': 32,
    'train_size': train_size_per_num_hints*12,
    'valid_size': valid_size_per_num_hints*12,
    'batch_size': 500,
    'epochs': 600,
    'valid_epochs': 25,
    'save_epochs': 25,
    'embed_size': 4,
    'hidden_layer_size': 32,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4
}

dataset = Datasets.load('../../4x4_all_reimbed/data/datasets.pkl')


split_inputs, split_outputs = dataset.split_data([train_size_per_num_hints,
                                                  train_size_per_num_hints+valid_size_per_num_hints])


rrn_utils.train_rrn(hyperparameters,
              train_inputs = split_inputs[0],
              train_outputs = split_outputs[0],
              other_inputs = {'validation': split_inputs[1]},
              other_outputs = {'validation': split_outputs[1]})
