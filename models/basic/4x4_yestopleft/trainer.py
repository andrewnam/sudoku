SUDOKU_PATH = '/home/ajhnam/sudoku'

import sys
sys.path.append(SUDOKU_PATH + '/src/sudoku')
sys.path.append(SUDOKU_PATH + '/src/misc')
sys.path.append(SUDOKU_PATH + '/src/models')

from dataset import Dataset

import rrn_utils

train_set = Dataset.load('./data/without_top_left.pkl')
test_set = Dataset.load('./data/with_top_left.pkl')


hyperparameters = {
    'device': 7,
    'dim_x': 2,
    'dim_y': 2,
    'num_iters': 32,
    'train_size': 3000,
    'valid_size': 500,
    'batch_size': 500,
    'epochs': 600,
    'valid_epochs': 25,
    'save_epochs': 25,
    'embed_size': 6,
    'hidden_layer_size': 32,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4
}

rrn_utils.train_rrn(hyperparameters,
              train_inputs = train_set.get_input_data(0, 3000),
              train_outputs = train_set.get_output_data(0, 3000),
              other_inputs = { 'validation': train_set.get_input_data(3000, 3500),
                               'test': test_set.get_input_data(0, 500)},
              other_outputs = { 'validation': train_set.get_output_data(3000, 3500),
                               'test': test_set.get_output_data(0, 500)})
