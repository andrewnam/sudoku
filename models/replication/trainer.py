SUDOKU_PATH = '/home/ajhnam/sudoku'

import sys
sys.path.append(SUDOKU_PATH + '/src/sudoku')
sys.path.append(SUDOKU_PATH + '/src/misc')
sys.path.append(SUDOKU_PATH + '/src/models')

from dataset import Dataset
from rrn_utils import train_rrn

train_set = Dataset.load('./data/train_set.pkl')
valid_set = Dataset.load('./data/valid_set.pkl')

for i in range(10):
    devices = list(range(i + 1))
    print("Running on devices {}".format(devices))
    hyperparameters = {
        'devices': devices,
        'dim_x': 3,
        'dim_y': 3,
        'num_iters': 32,
        'batch_size': 32*len(devices),
        'epochs': 3,
        'valid_epochs': 25,
        'save_epochs': 25,
        'embed_size': 16,
        'hidden_layer_size': 96,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4
    }

    train_rrn(hyperparameters,
              train_inputs=train_set.keys()[:2000],
              train_outputs=train_set.values()[:2000],
              other_inputs={"validation": valid_set.keys()},
              other_outputs={"validation": valid_set.values()})

