SUDOKU_PATH = '/home/ajhnam/sudoku'

import sys
sys.path.append(SUDOKU_PATH + '/src/sudoku')
sys.path.append(SUDOKU_PATH + '/src/models')

from dataset import Dataset
from grid_string import read_solutions_file

from script_utils import train_rrn


hyperparameters = {
    'model_name': 'yes_sol_one',
    'description': "Training the model on dataset with 1's at (0,0) in any of its solutions."
                   "Want to test if it can perform well on data with no 1's at (0, 0) in solutions.",
    'device': 8,
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

dataset = Dataset.load(SUDOKU_PATH + '/data/4x4_with_sol_one_6.pkl')
testset = read_solutions_file(SUDOKU_PATH + '/data/4x4_without_sol_one_6.sol')
test_inputs = list(testset.keys())

data = {
    'train_inputs': dataset.get_input_data(0),
    'train_outputs': dataset.get_output_data(0),
    'valid_inputs': dataset.get_input_data(1),
    'valid_outputs': dataset.get_output_data(1),
    'test_inputs': test_inputs,
    'test_outputs': [testset[k] for k in test_inputs]
}


train_rrn(hyperparameters, data)

