import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sudoku_model_utils
import random

from mlp import MLP
from sudoku_model_utils import collect_batches

# set random seed to 0
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.set_default_tensor_type('torch.DoubleTensor')
device = 0

class EmbedRRN(nn.Module):
    def __init__(self, dim_x, dim_y, embed_size=16, hidden_layer_size=96):
        super(EmbedRRN, self).__init__()
        self.max_digit = dim_x * dim_y
        self.embed_size = embed_size
        self.hidden_layer_size = hidden_layer_size

        self.edges = sudoku_model_utils.determine_edges(dim_x, dim_y)

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
                      self.embed_size])

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

        self.g_lstm.flatten_parameters()

        outputs = torch.empty(iters, batch_size, num_nodes, self.embed_size).cuda(device)
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


class RRN(nn.Module):
    def __init__(self, dim_x, dim_y, embed_size=16, hidden_layer_size=96):
        super(RRN, self).__init__()
        self.max_digit = dim_x * dim_y
        self.embed_size = embed_size
        self.hidden_layer_size = hidden_layer_size

        self.edges = sudoku_model_utils.determine_edges(dim_x, dim_y)

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

        self.g_lstm.flatten_parameters()

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

class ReEmbedRRN(nn.Module):
    def __init__(self, dim_x, dim_y, embed_size=16, hidden_layer_size=96):
        super(ReEmbedRRN, self).__init__()
        self.max_digit = dim_x * dim_y
        self.embed_size = embed_size
        self.hidden_layer_size = hidden_layer_size

        self.edges = sudoku_model_utils.determine_edges(dim_x, dim_y)

        self.embed_layer = nn.Linear(self.max_digit + 1, self.embed_size)
        self.decoder = nn.Linear(self.embed_size, self.max_digit + 1)

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
                      self.max_digit + 1])

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

        self.g_lstm.flatten_parameters()

        decode_output = self.decoder(embeddings)
        pre_outputs = torch.empty(iters, batch_size, num_nodes, self.max_digit+1).cuda(device);
        outputs = torch.empty(iters, batch_size, num_nodes, self.max_digit+1).cuda(device)
        for i in range(iters):
            M = torch.empty(batch_size, self.max_digit ** 2, self.hidden_layer_size).cuda(device)
            for node in range(num_nodes):
                msgs = torch.stack(
                    [self.f(torch.cat([H[:, node, :], H[:, other, :]], dim=1)) for other in self.edges[node]], dim=1)
                M[:, node, :] = torch.sum(msgs, dim=1)
            input_to_g_lstm = self.g_mlp(torch.cat([X, M], dim=2)).view(lstm_layer_shape)
            _, (g_lstm_h, g_lstm_c) = self.g_lstm(input_to_g_lstm, (g_lstm_h, g_lstm_c))
            H = g_lstm_h.view(X.shape)
            pre_output = self.r(H)
            pre_outputs[i] = pre_output
            output = self.decoder(self.embed_layer(F.softmax(pre_output, dim=2)))
            outputs[i] = output
        return decode_output, pre_outputs, outputs