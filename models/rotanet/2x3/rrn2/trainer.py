from sudoku.dataset import Datasets
from models import RRN, Solver

puzzles = Datasets.load('../2x3_16.dsts')

seeds = list(puzzles.keys())
train_seeds = seeds[:90]
validation_seeds = seeds[90:]

n_per_puzzle = 10
train_inputs = []
train_outputs = []
validation_inputs = []
validation_outputs = []

for seed in train_seeds:
    dst = puzzles[seed]
    train_inputs += dst.get_input_data(n_per_puzzle)
    train_outputs += dst.get_output_data(n_per_puzzle)
for seed in validation_seeds:
    dst = puzzles[seed]
    validation_inputs += dst.get_input_data(n_per_puzzle)
    validation_outputs += dst.get_output_data(n_per_puzzle)

# device = None
device = 1
# model = RRN(dim_x=2, dim_y=3, mlp_layers=1, embed_size=6, hidden_layer_size=32)
model = RRN(dim_x=2, dim_y=3, mlp_layers=2, embed_size=10, hidden_layer_size=48).cuda(device)
solver = Solver(model, '.')
solver.train_inputs = train_inputs
solver.train_outputs = train_outputs
solver.other_inputs['validation'] = validation_inputs
solver.other_outputs['validation'] = validation_outputs
solver.batch_size = 400
solver.epochs = 200
solver.device = device
solver.train()
