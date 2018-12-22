import torch
import torch.nn as nn
import numpy as np
import itertools
import random

from mlp import MLP

# set random seed to 0
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.set_default_tensor_type('torch.DoubleTensor')
device = 0


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
    return edges.view(max_digit ** 2, shape[2])


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
        lstm_layer_shape = (1, batch_size * num_nodes, self.hidden_layer_size)

        embeddings = self.embed_layer(grids)
        X = self.input_mlp(embeddings)
        H = X.clone().detach().cuda(device)
        g_lstm_h = H.view(lstm_layer_shape)
        g_lstm_c = torch.randn(lstm_layer_shape).cuda(device)

        outputs = torch.empty(iters, batch_size, num_nodes, self.max_digit).cuda(device)
        for i in range(iters):
            M = torch.empty(batch_size, self.max_digit ** 2, self.hidden_layer_size).cuda(device)
            for node in range(num_nodes):
                msgs = torch.stack(
                    [self.f(torch.cat([H[:, node, :], H[:, other, :]], dim=1)) for other in self.edges[node]], dim=1)
                M[:, node, :] = torch.sum(msgs, dim=1)
            input_to_g_lstm = self.g_mlp(torch.cat([X, M], dim=2)).view(lstm_layer_shape)
            _, (g_lstm_h, g_lstm_c) = self.g_lstm(input_to_g_lstm, (g_lstm_h, g_lstm_c))
            H = g_lstm_h.view(X.shape)
            output = self.r(H)
            outputs[i] = output
        return outputs