SUDOKU_PATH = '/home/ajhnam/sudoku'

import sys
sys.path.append(SUDOKU_PATH + '/src/sudoku')
sys.path.append(SUDOKU_PATH + '/src/models')

from dataset import Dataset
from grid_string import read_solutions_file

from script_utils import train_rrn


hyperparameters = {
    'model_name': 'no_corner_hint',
    'description': "Training the model on dataset with no hint at (0,0)."
                   "Want to test if it can perform well on data with hint at (0, 0).",
    'device': 5,
    'dim_x': 2,
    'dim_y': 2,
    'num_hints': 6,
    'num_iters': 32,
    'train_size': 1200,
    'valid_size': 240,
    'test_size': 240,
    'batch_size': 400,
    'epochs': 600,
    'save_epochs': 25,
    'embed_size': 6,
    'hidden_layer_size': 32,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4
}

dataset = Dataset.load(SUDOKU_PATH + '/data/4x4_without_hint_6.pkl')
with_one = read_solutions_file(SUDOKU_PATH + '/data/4x4_with_hint_6.sol')
test_inputs = list(with_one.keys())

data = {
    'train_inputs': dataset.get_input_data(0),
    'train_outputs': dataset.get_output_data(0),
    'valid_inputs': dataset.get_input_data(1),
    'valid_outputs': dataset.get_output_data(1),
    'test_inputs': test_inputs,
    'test_outputs': [with_one[k] for k in test_inputs]
}


train_rrn(hyperparameters, data)

