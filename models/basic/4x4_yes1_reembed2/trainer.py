SUDOKU_PATH = '/home/ajhnam/sudoku'

import sys
sys.path.append(SUDOKU_PATH + '/src/sudoku')
sys.path.append(SUDOKU_PATH + '/src/misc')
sys.path.append(SUDOKU_PATH + '/src/models')

from dataset import Dataset

import rrn_utils

train_set = Dataset.load('../4x4_yes1/data/without_one.pkl')
test_set = Dataset.load('../4x4_yes1/data/with_one.pkl')


hyperparameters = {
    'device': 9,
    'dim_x': 2,
    'dim_y': 2,
    'num_iters': 32,
    'train_size': 1200,
    'valid_size': 240,
    'batch_size': 400,
    'epochs': 600,
    'valid_epochs': 25,
    'save_epochs': 25,
    'embed_size': 6,
    'hidden_layer_size': 32,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4
}

rrn_utils.train_reembedrrn(hyperparameters,
              train_inputs = train_set.get_input_data(0, 1200),
              train_outputs = train_set.get_output_data(0, 1200),
              other_inputs = { 'validation': train_set.get_input_data(1200, 1440),
                               'test': test_set.get_input_data(0, 240)},
              other_outputs = { 'validation': train_set.get_output_data(1200, 1440),
                               'test': test_set.get_output_data(0, 240)})
