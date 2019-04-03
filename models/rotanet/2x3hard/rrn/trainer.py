from sudoku2.dataset import Datasets
from models import RRN, Solver

datasets = Datasets.load('../2x3_hard.dsts')

train_inputs = datasets['train'].keys()
train_outputs = datasets['train'].values()
validation_inputs = datasets['valid'].keys()
validation_outputs = datasets['valid'].values()

device = 0
model = RRN(dim_x=2, dim_y=3, mlp_layers=1, embed_size=6, hidden_layer_size=32).cuda(device)
solver = Solver(model, '.')
solver.train_inputs = train_inputs
solver.train_outputs = train_outputs
solver.other_inputs['validation'] = validation_inputs
solver.other_outputs['validation'] = validation_outputs
solver.batch_size = 800
solver.epochs = 800
solver.device = device
solver.train()
