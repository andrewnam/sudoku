import torch
import torch.nn as nn
from mlp import MLP

import sudoku_model_utils


class RelationalLayer(nn.Module):

    def __init__(self, layer_sizes, edges):
        """
        A module for relational networks.
        Assumes that the input is of shape (N, a, b) where
            N = batch size
            a = number of relational objects in a single input item
            b = representation layer size for each relational object
        :param layer_sizes: Similar to MLP
        :param edges: a 2-d list of shape (a, n) where
            a = number of relational objects in a single input item
            n = number of neighbors that the object relates with.

            the i_th entry is a list of other cells' indices that
            object i shares a house with
        """

        super(RelationalLayer, self).__init__()
        self.layer_sizes = layer_sizes[:]
        self.edges = edges

        self.layer_sizes[0] *= 2
        self.mlp = MLP(self.layer_sizes)

    def forward(self, x):
        device = x.get_device()

        M = torch.empty(x.shape[0], x.shape[1], self.layer_sizes[-1]).cuda(device)
        for i in range(len(self.edges)):
            msgs = [self.mlp(torch.cat([x[:, i, :], x[:, other, :]], dim=1)) for other in self.edges[i]]
            msgs = torch.stack(msgs, dim=1)
            M[:, i, :] = torch.sum(msgs, dim=1)

        return M


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

        self.rel_layer = RelationalLayer([self.hidden_layer_size,
                                          self.hidden_layer_size,
                                          self.hidden_layer_size,
                                          self.hidden_layer_size],
                                         self.edges)
        self.g_mlp = MLP([2 * self.hidden_layer_size,
                          self.hidden_layer_size,
                          self.hidden_layer_size,
                          self.hidden_layer_size])
        self.g_lstm = nn.LSTM(self.hidden_layer_size, self.hidden_layer_size)
        self.r = MLP([self.hidden_layer_size,
                      self.hidden_layer_size,
                      self.hidden_layer_size,
                      self.max_digit])

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
            M = self.rel_layer(H)
            input_to_g_lstm = self.g_mlp(torch.cat([X, M], dim=2)).view(lstm_layer_shape)
            _, (g_lstm_h, g_lstm_c) = self.g_lstm(input_to_g_lstm, (g_lstm_h, g_lstm_c))
            H = g_lstm_h.view(X.shape)
            output = self.r(H)
            outputs[i] = output
        return outputs