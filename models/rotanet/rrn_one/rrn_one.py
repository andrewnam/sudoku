import os
from sudoku.dataset import Datasets
from models import RRN
from sudoku2 import Solver

puzzles = Datasets.load('../data/puzzles6_one_count.dsts')
# 0: 1440, 1: 9264, 2: 8946, 3: 1536, 4: 6

puzzles[0]

train_n = 1000
valid_n = 1000

device = 1
model = RRN(dim_x=2, dim_y=2, mlp_layers=1, embed_size=6, hidden_layer_size=32).cuda(device)
solver = Solver(model, '.')
solver.train_inputs = puzzles[0].get_input_data(train_n)
solver.train_outputs = puzzles[0].get_output_data(train_n)
solver.other_inputs['zero_1'] = puzzles[0].get_input_data(train_n, train_n + valid_n)
solver.other_outputs['zero_1'] = puzzles[0].get_output_data(train_n, train_n + valid_n)
solver.other_inputs['one_1'] = puzzles[1].get_input_data(valid_n)
solver.other_outputs['one_1'] = puzzles[1].get_output_data(valid_n)
solver.other_inputs['two_1'] = puzzles[2].get_input_data(valid_n)
solver.other_outputs['two_1'] = puzzles[2].get_output_data(valid_n)
solver.other_inputs['three_1'] = puzzles[3].get_input_data(valid_n)
solver.other_outputs['three_1'] = puzzles[3].get_output_data(valid_n)

solver.device = device
solver.train()
