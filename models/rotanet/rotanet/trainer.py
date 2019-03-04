from sudoku.dataset import Datasets
from models import RelationalLayer, determine_edges, MLP
from sudoku2 import Solver
from sudoku2.solver import encode_grid_string
from sudoku2 import GridString
import numpy as np
import torch
import torch.nn as nn
import utils

puzzles = Datasets.load('../data/puzzles6_one_count.dsts')
inputs = puzzles[0].get_input_data(2)
outputs = puzzles[0].get_output_data(2)
inputs = torch.stack([encode_grid_string(GridString.load(s)) for s in inputs]).type(torch.long)
outputs = torch.stack([encode_grid_string(GridString.load(s)) for s in outputs]).type(torch.long) - 1
inputs.shape

dim_x, dim_y = 2, 2
max_digit = dim_x*dim_y
mask = torch.zeros(inputs.size() + (max_digit, 3))
mask.shape
mask[:,:,:,0] = (inputs == 0).reshape(inputs.size() + (1, )).expand(inputs.size() + (4,))
for i in range(max_digit):
    mask[:,:,i,1] = (inputs == i+1)
    mask[:,:,i,2] = (inputs != i+1) & (inputs != 0)

inputs[0]

embed_size = 6
hidden_layer_size = 32
mlp_layers = 1

encode_layer = MLP([3, embed_size, hidden_layer_size])
X = encode_layer(mask)
H = torch.tensor(X)
rel_layer_cell = RelationalLayer([hidden_layer_size]*(mlp_layers+1), utils.fully_connected_graph(4), 2)
msg_cell = rel_layer_cell(H)
rel_layer_digit = RelationalLayer([hidden_layer_size]*(mlp_layers+1), determine_edges(dim_x, dim_y), 1)
msg_digit = rel_layer_digit(H)


class RelationalLayer(nn.Module):

    def __init__(self, layer_sizes, edges, dim):
        """
        A module for relational networks.
        Assumes that the input is of shape (..., A, ..., B) where
            A = number of relational objects in a single input item
            B = vector of length layer_sizes[0]
        :param layer_sizes: Similar to MLP
        :param edges: a 2-d list of shape (a, n) where
            a = number of relational objects in a single input item
            n = number of neighbors that the object relates with.

            the i_th entry is a list of other cells' indices that
            object i shares a house with
        :param dim: the dimension that the relationship edges are based on,
            i.e. dimension of A
        """

        super(RelationalLayer, self).__init__()
        self.layer_sizes = layer_sizes[:]
        self.edges = edges
        self.dim = dim

        self.layer_sizes[0] *= 2
        self.mlp = MLP(self.layer_sizes)

    def forward(self, x):
        permutation = list(range(len(x.shape)))
        permutation[self.dim] = 0
        permutation[0] = self.dim
        x = x.permute(permutation)

        M = torch.empty(x.shape[:-1] + (self.layer_sizes[-1],))
        if x.is_cuda:
            M = M.cuda(x.get_device())

        for i in range(len(self.edges)):
            msgs = [self.mlp(torch.cat([x[i, ..., :], x[other, ..., :]], dim=-1)) for other in self.edges[i]]
            msgs = torch.stack(msgs, dim=0)
            M[i, ..., :] = torch.sum(msgs, dim=0)

        return M.permute(permutation)

H.shape
H[..., 3, :].shape
len(H.shape)-2

v = np.vectorize(lambda k: np.concatenate([np.arange(k), np.arange(k+1, 10)], axis=0))
v(np.arange(10))
np.array([np.concatenate([np.arange(k), np.arange(k+1, 10)], axis=0) for k in range(10)])

k = 4
list(range(k)) + list(range(k+1, 10))

class Rotanet(nn.Module):
    def __init__(self, dim_x, dim_y, mlp_layers=3, embed_size=16, hidden_layer_size=96):
        super(Rotanet, self).__init__()
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.mlp_layers = mlp_layers
        self.max_digit = dim_x * dim_y
        self.embed_size = embed_size
        self.hidden_layer_size = hidden_layer_size

        self.edges = determine_edges(dim_x, dim_y)

        self.embed_layer = nn.Embedding(3, self.embed_size)
        self.input_mlp = MLP([self.embed_size] + [self.hidden_layer_size]*self.mlp_layers)

        self.rel_layer = RelationalLayer([self.hidden_layer_size]*(self.mlp_layers+1),
                                         determine_edges(dim_x, dim_y))
        self.g_mlp = MLP([2 * self.hidden_layer_size] + [self.hidden_layer_size]*self.mlp_layers)
        self.g_lstm = nn.LSTM(self.hidden_layer_size, self.hidden_layer_size)
        self.r = MLP([self.hidden_layer_size]*self.mlp_layers + [self.max_digit])

    def forward(self, grids, iters):
        device = grids.get_device() if grids.is_cuda else None

        batch_size = len(grids)
        num_nodes = self.max_digit ** 2
        lstm_layer_shape = (1, batch_size * num_nodes, self.hidden_layer_size)

        embeddings = self.embed_layer(grids)
        X = self.input_mlp(embeddings)
        H = X.clone().detach()

        if device is not None:
            H = H.cuda(device)

        g_lstm_h = H.view(lstm_layer_shape)
        g_lstm_c = torch.randn(lstm_layer_shape)
        outputs = torch.empty(iters, batch_size, num_nodes, self.max_digit)

        if device is not None:
            g_lstm_c = g_lstm_c.cuda(device)
            outputs = outputs.cuda(device)

        self.g_lstm.flatten_parameters()
        for i in range(iters):
            M = self.rel_layer(H)
            input_to_g_lstm = self.g_mlp(torch.cat([X, M], dim=2)).view(lstm_layer_shape)
            _, (g_lstm_h, g_lstm_c) = self.g_lstm(input_to_g_lstm, (g_lstm_h, g_lstm_c))
            H = g_lstm_h.view(X.shape)
            output = self.r(H)
            outputs[i] = output
        return outputs

# class Rotanet(nn.Module):
#     def __init__(self, dim_x, dim_y, mlp_layers=3, embed_size=16, hidden_layer_size=96):
#         super(Rotanet, self).__init__()
#         self.dim_x = dim_x
#         self.dim_y = dim_y
#         self.mlp_layers = mlp_layers
#         self.max_digit = dim_x * dim_y
#         self.embed_size = embed_size
#         self.hidden_layer_size = hidden_layer_size
#
#         self.edges = determine_edges(dim_x, dim_y)
#
#         self.embed_layer = nn.Embedding(3, self.embed_size)
#         self.input_mlp = MLP([self.embed_size] + [self.hidden_layer_size]*self.mlp_layers)
#
#         self.rel_layer = RelationalLayer([self.hidden_layer_size]*(self.mlp_layers+1),
#                                          self.edges)
#         self.g_mlp = MLP([2 * self.hidden_layer_size] + [self.hidden_layer_size]*self.mlp_layers)
#         self.g_lstm = nn.LSTM(self.hidden_layer_size, self.hidden_layer_size)
#         self.r = MLP([self.hidden_layer_size]*self.mlp_layers + [self.max_digit])
#
#     def forward(self, grids, iters):
#         device = grids.get_device() if grids.is_cuda else None
#
#         batch_size = len(grids)
#         num_nodes = self.max_digit ** 2
#         lstm_layer_shape = (1, batch_size * num_nodes, self.hidden_layer_size)
#
#         embeddings = self.embed_layer(grids)
#         X = self.input_mlp(embeddings)
#         H = X.clone().detach()
#
#         if device is not None:
#             H = H.cuda(device)
#
#         g_lstm_h = H.view(lstm_layer_shape)
#         g_lstm_c = torch.randn(lstm_layer_shape)
#         outputs = torch.empty(iters, batch_size, num_nodes, self.max_digit)
#
#         if device is not None:
#             g_lstm_c = g_lstm_c.cuda(device)
#             outputs = outputs.cuda(device)
#
#         self.g_lstm.flatten_parameters()
#         for i in range(iters):
#             M = self.rel_layer(H)
#             input_to_g_lstm = self.g_mlp(torch.cat([X, M], dim=2)).view(lstm_layer_shape)
#             _, (g_lstm_h, g_lstm_c) = self.g_lstm(input_to_g_lstm, (g_lstm_h, g_lstm_c))
#             H = g_lstm_h.view(X.shape)
#             output = self.r(H)
#             outputs[i] = output
#         return outputs


puzzles = Datasets.load('../data/puzzles6_one_count.dsts')
# 0: 1440, 1: 9264, 2: 8946, 3: 1536, 4: 6

train_n = 1000
valid_n = 1000

device = 1
model = Rotanet(dim_x=2, dim_y=2, mlp_layers=1, embed_size=6, hidden_layer_size=32).cuda(device)
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
