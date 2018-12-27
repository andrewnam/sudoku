SUDOKU_PATH = '/home/ajhnam/sudoku'

import sys
sys.path.append(SUDOKU_PATH + '/src/sudoku')
sys.path.append(SUDOKU_PATH + '/src/misc')
sys.path.append(SUDOKU_PATH + '/src/models')

from dataset import Datasets

import rrn_utils
from utils import print

# devices = [0,1,2,3,4,5,6,7,8,9]
train_size_per_num_hints = 250 # * 12 = 3000
valid_size_per_num_hints = 50 # * 12 = 600

dataset = Datasets.load('./data/datasets.pkl')

split_inputs, split_outputs = dataset.split_data([train_size_per_num_hints,
                                                  train_size_per_num_hints+valid_size_per_num_hints])


for i in range(10):
    devices = list(range(i+1))
    print("Running on devices {}".format(devices))
    hyperparameters = {
        'devices': devices,
        'dim_x': 2,
        'dim_y': 2,
        'num_iters': 32,
        'train_size': train_size_per_num_hints*12,
        'valid_size': valid_size_per_num_hints*12,
        'batch_size': 500*len(devices),
        'epochs': 4,
        'valid_epochs': 25,
        'save_epochs': 25,
        'embed_size': 6,
        'hidden_layer_size': 32,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4
    }

    rrn_utils.train_rrn(hyperparameters,
                  train_inputs = split_inputs[0],
                  train_outputs = split_outputs[0],
                  other_inputs = { 'validation': split_inputs[1] },
                  other_outputs = { 'validation': split_outputs[1] })

