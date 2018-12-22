import torch.nn as nn

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