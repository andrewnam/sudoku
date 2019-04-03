import torch
import torch.nn as nn
from .mlp import MLP
from .relational_network import RelationalLayer, determine_edges
import utils

class Rotanet(nn.Module):
    mask_dim = 3

    def __init__(self, dim_x, dim_y, mlp_layers=3, embed_size=16, hidden_layer_size=96):
        super(Rotanet, self).__init__()
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.max_digit = self.dim_x * self.dim_y
        self.mlp_layers = mlp_layers
        self.embed_size = embed_size
        self.hidden_layer_size = hidden_layer_size

        relnet_sizes = [self.hidden_layer_size]*(self.mlp_layers+1)

        self.encode_layer = MLP([Rotanet.mask_dim, self.embed_size, self.hidden_layer_size])
        self.cell_relnet = RelationalLayer(relnet_sizes, utils.fully_connected_graph(self.max_digit), 2)
        self.digit_relnet = RelationalLayer(relnet_sizes, determine_edges(dim_x, dim_y), 1)
        self.g_mlp = MLP([3 * self.hidden_layer_size] + [self.hidden_layer_size]*self.mlp_layers)
        self.g_lstm = nn.LSTM(self.hidden_layer_size, self.hidden_layer_size)
        self.r = MLP([self.hidden_layer_size]*self.mlp_layers + [1])

    def forward(self, inputs, iters):
        batch_size = len(inputs)
        num_nodes = self.max_digit ** 2
        lstm_layer_shape = (1, batch_size * num_nodes * self.max_digit, self.hidden_layer_size)
        device = inputs.device.index

        mask = torch.zeros(inputs.size() + (self.max_digit, Rotanet.mask_dim), device=device)
        mask[:,:,:,0] = (inputs == 0).reshape(inputs.size() + (1, )).expand(inputs.size() + (self.max_digit,))
        for i in range(self.max_digit):
            mask[:,:,i,1] = (inputs == i+1)
            mask[:,:,i,2] = (inputs != i+1) & (inputs != 0)

        X = self.encode_layer(mask)
        H = torch.tensor(X, device=device)
        g_lstm_h = H.view(lstm_layer_shape)
        g_lstm_c = torch.randn(lstm_layer_shape, device=device)
        outputs = torch.empty(iters, batch_size, num_nodes, self.max_digit, device=device)

        self.g_lstm.flatten_parameters()
        for i in range(iters):
            msg_cell = self.cell_relnet(H)
            msg_digit = self.digit_relnet(H)

            input_to_g_lstm = self.g_mlp(torch.cat([X, msg_cell, msg_digit], dim=3)).view(lstm_layer_shape)
            _, (g_lstm_h, g_lstm_c) = self.g_lstm(input_to_g_lstm, (g_lstm_h, g_lstm_c))
            H = g_lstm_h.view(X.shape)
            output = self.r(H).squeeze()
            outputs[i] = output
        return outputs
