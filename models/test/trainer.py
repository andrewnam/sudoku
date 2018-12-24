SUDOKU_PATH = '/home/ajhnam/sudoku'

import sys
sys.path.append(SUDOKU_PATH + '/src/sudoku')
sys.path.append(SUDOKU_PATH + '/src/misc')
sys.path.append(SUDOKU_PATH + '/src/models')

from dataset import Datasets

from rrn_utils import train_rrn


train_size_per_num_hints = 20 # * 12 = 240
valid_size_per_num_hints = 5 # * 12 = 60

hyperparameters = {
    'device': 1,
    'dim_x': 2,
    'dim_y': 2,
    'num_iters': 32,
    'train_size': train_size_per_num_hints*12,
    'valid_size': valid_size_per_num_hints*12,
    'batch_size': 120,
    'epochs': 7,
    'save_epochs': 2,
    'embed_size': 6,
    'hidden_layer_size': 32,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4
}

dataset = Datasets.load('./data/datasets.pkl')

split_inputs, split_outputs = dataset.split_data([train_size_per_num_hints,
                                                  train_size_per_num_hints+valid_size_per_num_hints])

data = {
    'train_inputs': split_inputs[0],
    'train_outputs': split_outputs[0],
    'valid_inputs': split_inputs[1],
    'valid_outputs': split_outputs[1]
}

train_rrn(hyperparameters, data)

