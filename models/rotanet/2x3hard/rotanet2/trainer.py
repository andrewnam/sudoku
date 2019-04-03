from sudoku2.dataset import Datasets
from models import Rotanet, Solver

datasets = Datasets.load('../2x3_hard.dsts')

train_inputs = datasets['train'].keys()
train_outputs = datasets['train'].values()
validation_inputs = datasets['valid'].keys()
validation_outputs = datasets['valid'].values()

device = 3
model = Rotanet(dim_x=2, dim_y=3, mlp_layers=2, embed_size=10, hidden_layer_size=48).cuda(device)
solver = Solver(model, '.')
solver.train_inputs = train_inputs
solver.train_outputs = train_outputs
solver.other_inputs['validation'] = validation_inputs
solver.other_outputs['validation'] = validation_outputs
solver.batch_size = 100
solver.epochs = 400
solver.device = device
solver.train()
