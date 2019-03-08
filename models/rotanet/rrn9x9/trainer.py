import os
from sudoku.dataset import Dataset
from models import RRN
from sudoku2 import Solver

puzzles6 = Dataset.load('../data/puzzles3x3_test.dst')

train_n = 1000
test_n = 1000
train_inputs = puzzles6.get_input_data(train_n)
train_outputs = puzzles6.get_output_data(train_n)
validation_inputs = puzzles6.get_input_data(train_n, train_n + test_n)
validation_outputs = puzzles6.get_output_data(train_n, train_n + test_n)

device = 4
model = RRN(dim_x=3, dim_y=3, mlp_layers=1, embed_size=6, hidden_layer_size=48).cuda(device)
solver = Solver(model, '.')
solver.train_inputs = train_inputs
solver.train_outputs = train_outputs
solver.other_inputs['validation'] = validation_inputs
solver.other_outputs['validation'] = validation_outputs
solver.device = device
solver.train() 
