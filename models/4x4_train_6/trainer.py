SUDOKU_PATH = '/home/ajhnam/sudoku'

import os
import sys
sys.path.append(SUDOKU_PATH + '/src/sudoku')
sys.path.append(SUDOKU_PATH + '/src/misc')
sys.path.append(SUDOKU_PATH + '/src/models')

from dataset import Dataset
from rrn_utils import train_rrn

hyperparameters = {
    'device': 2,
    'dim_x': 2,
    'dim_y': 2,
    'num_iters': 32,
    'train_size': 3000,
    'valid_size': 500,
    'batch_size': 500,
    'epochs': 600,
    'save_epochs': 25,
    'embed_size': 6,
    'hidden_layer_size': 32,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4
}

dataset = Dataset.load('./data/datasets.pkl')

split_inputs, split_outputs = dataset.split_data([hyperparameters['train_size'],
                                                  hyperparameters['train_size']+hyperparameters['valid_size']])

data = {
    'train_inputs': split_inputs[0],
    'train_outputs': split_outputs[0],
    'valid_inputs': split_inputs[1],
    'valid_outputs': split_outputs[1]
}

train_rrn(hyperparameters, data)

