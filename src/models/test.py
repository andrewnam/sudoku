SUDOKU_PATH = '/home/ajhnam/sudoku'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda as cutorch
import torch.optim as optim
import numpy as np
import itertools
import random
from tqdm import tqdm

import sys
sys.path.append(SUDOKU_PATH + '/src/sudoku')

from grid_string import GridString
from dataset import Dataset

# set random seed to 0
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.set_default_tensor_type('torch.DoubleTensor')


def count_model_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_tensor_memory_size(tensor):
    return tensor.element_size() * tensor.nelement() // (2 ** 20)


def get_gpu_memory(device):
    return cutorch.memory_allocated(device) // (2 ** 20)


def determine_edges(dim_x, dim_y):
    """
    Returns a 2-d array of (max_digit**2, n) where the i_th entry is a list of
        other cells' indices that cell i shares a house with
    """
    max_digit = dim_x * dim_y
    edges = []
    for row in range(max_digit):
        row_edges = []
        for col in range(max_digit):
            # row & column
            col_edges = {(row, i) for i in range(max_digit)}
            col_edges |= {(i, col) for i in range(max_digit)}

            # box
            x_min = (row // dim_x) * dim_x
            y_min = (col // dim_y) * dim_y
            col_edges |= set(itertools.product(range(x_min, x_min + dim_x), range(y_min, y_min + dim_y)))

            # removing self
            col_edges -= {(row, col)}
            col_edges = [row * max_digit + col for row, col in col_edges]
            row_edges.append(sorted(col_edges))
        edges.append(row_edges)
    edges = torch.tensor(edges)
    shape = edges.shape
    return edges.reshape(max_digit ** 2, shape[2])


def encode_input(grid_string: GridString):
    return torch.tensor(list(grid_string.traverse_grid()))


def encode_output(grid_string: GridString):
    return torch.tensor(list(grid_string.traverse_grid())) - 1

num_hints = 12

dataset = Dataset.load(SUDOKU_PATH + '/data/4x4_{}.txt'.format(num_hints))

max_digit = 4
num_cells = max_digit**2
cell_vec_dim = max_digit + 1

train_inputs = dataset.get_input_data(0)
train_outputs = dataset.get_output_data(0)
valid_inputs = dataset.get_input_data(1)
valid_outputs = dataset.get_output_data(1)
test_inputs = dataset.get_input_data(2)
test_outputs = dataset.get_output_data(2)

train_x = torch.cat([encode_input(p) for p in train_inputs]).reshape(len(train_inputs), num_cells)
train_y = torch.cat([encode_output(p) for p in train_outputs]).reshape(len(train_outputs), num_cells)
valid_x = torch.cat([encode_input(p) for p in valid_inputs]).reshape(len(valid_inputs), num_cells)
valid_y = torch.cat([encode_output(p) for p in valid_outputs]).reshape(len(valid_outputs), num_cells)
test_x = torch.cat([encode_input(p) for p in test_inputs]).reshape(len(test_inputs), num_cells)
test_y = torch.cat([encode_output(p) for p in test_outputs]).reshape(len(test_outputs), num_cells)


class MLP(nn.Module):
    def __init__(self, layer_sizes):
        super(MLP, self).__init__()
        self.layer_sizes = layer_sizes

        self.layers = nn.ModuleList()
        self.nonlinear = nn.ReLU()

        prev_layer_size = self.layer_sizes[0]
        for size in self.layer_sizes[1:]:
            self.layers.append(nn.Linear(prev_layer_size, size))
            prev_layer_size = size

    def forward(self, X):
        vector = X
        for layer in self.layers[:-1]:
            vector = self.nonlinear(layer(vector))
        return self.layers[-1](vector)


class RRN(nn.Module):
    def __init__(self, dim_x, dim_y, embed_size=16, hidden_layer_size=96):
        super(RRN, self).__init__()
        self.max_digit = dim_x * dim_y
        self.embed_size = embed_size
        self.hidden_layer_size = hidden_layer_size

        self.edges = determine_edges(dim_x, dim_y)

        self.embed_layer = nn.Embedding(self.max_digit + 1, self.embed_size)
        self.input_mlp = MLP([self.embed_size,
                              self.hidden_layer_size,
                              self.hidden_layer_size,
                              self.hidden_layer_size])

        self.f = MLP([2 * self.hidden_layer_size,
                      self.hidden_layer_size,
                      self.hidden_layer_size,
                      self.hidden_layer_size])
        self.g_mlp = MLP([2 * self.hidden_layer_size,
                          self.hidden_layer_size,
                          self.hidden_layer_size,
                          self.hidden_layer_size])
        self.g_lstm = nn.LSTM(self.hidden_layer_size, self.hidden_layer_size)
        self.r = MLP([self.hidden_layer_size,
                      self.hidden_layer_size,
                      self.hidden_layer_size,
                      self.max_digit])

    def compute_messages(self, H):
        messages = torch.zeros(H.shape)
        batch_size = H.shape[0]
        num_nodes = H.shape[1]
        for puzzle_index in range(batch_size):  # for puzzle in batch
            messages[puzzle_index] = torch.tensor(
                [torch.sum(H[puzzle_index][self.edges[n]]) for n in range(num_nodes)]).cuda(H.get_device())
        return messages

    def forward(self, grids, iters):
        device = grids.get_device()

        batch_size = len(grids)
        num_nodes = self.max_digit ** 2

        embeddings = self.embed_layer(grids)
        X = self.input_mlp(embeddings)
        H = X.clone().detach().cuda(device)
        g_lstm_h = H.reshape(1, batch_size * num_nodes, self.hidden_layer_size)
        g_lstm_c = torch.randn(1, batch_size * num_nodes, self.hidden_layer_size).cuda(device)

        outputs = torch.empty(iters, batch_size, num_nodes, self.max_digit).cuda(device)
        for i in range(iters):
            M = torch.empty(batch_size, self.max_digit ** 2, self.hidden_layer_size).cuda(device)
            for node in range(num_nodes):
                msgs = torch.stack(
                    [self.f(torch.cat([H[:, node, :], H[:, other, :]], dim=1)) for other in self.edges[node]], dim=1)
                M[:, node, :] = torch.sum(msgs, dim=1)
            input_to_g_lstm = self.g_mlp(torch.cat([X, M], dim=2)).view(1, batch_size * num_nodes,
                                                                        self.hidden_layer_size)
            _, (g_lstm_h, g_lstm_c) = self.g_lstm(input_to_g_lstm, (g_lstm_h, g_lstm_c))
            H = g_lstm_h.view(X.shape)
            output = self.r(H)
            outputs[i] = output
        return outputs

train_x = train_x[:1]
train_y = train_y[:1]

device = 0
num_iters = 32
epochs = 5
batch_size = 1024
model = RRN( dim_x=2, dim_y=2, embed_size=6, hidden_layer_size=32).cuda(device)
optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)

def closure():
    optimizer.zero_grad()
    total_loss = 0
    for i in tqdm(range(0, len(train_x), batch_size), leave=False):
        x_batch = train_x[i:i+batch_size].cuda(device)
        y_batch = train_y[i:i+batch_size].cuda(device)
        predictions = [p.permute(0,2,1) for p in model(x_batch, num_iters)]
        loss = sum([F.cross_entropy(p, y_batch) for p in predictions])
        loss.backward()
        total_loss += loss
    return total_loss

for i in tqdm(range(epochs)):
    loss = optimizer.step(closure)
    train_loss = round(float(loss), 3)
    print("Iter {} | TLoss {}".format(i, train_loss))