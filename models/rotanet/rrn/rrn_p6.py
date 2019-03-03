import os
from sudoku.dataset import Dataset
from models import RRN
from sudoku2 import Solver

project_path = '/Users/andrew/Desktop/sudoku'
os.chdir(project_path)

puzzles6 = Dataset.load('models/rotanet/data/puzzles6.dst')

train_n = 1000
test_n = 1000
train_inputs = puzzles6.get_input_data(train_n)
train_outputs = puzzles6.get_output_data(train_n)
validation_inputs = puzzles6.get_input_data(train_n, train_n + test_n)
validation_outputs = puzzles6.get_output_data(train_n, train_n + test_n)

model = RRN(dim_x=2, dim_y=2, mlp_layers=1, embed_size=6, hidden_layer_size=32)
solver = Solver(model, 'models/rotanet/rrn')
solver.train_inputs = train_inputs
solver.train_outputs = train_outputs
solver.other_inputs['validation'] = validation_inputs
solver.other_outputs['validation'] = validation_outputs
solver.train()
