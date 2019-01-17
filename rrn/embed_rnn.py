SUDOKU_PATH = '/home/ajhnam/sudoku'

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import os
import sys
sys.path.append(SUDOKU_PATH + '/src/sudoku')
sys.path.append(SUDOKU_PATH + '/src/models')
sys.path.append(SUDOKU_PATH + '/src/misc')

import utils
from dataset import Datasets
from rrn import RRN, collect_batches, EmbedRRN
from mlp import MLP
import rrn_utils

np.random.seed(0)
torch.manual_seed(0)
torch.set_default_tensor_type('torch.DoubleTensor')

train_size_per_num_hints = 250 # * 12 = 3000
valid_size_per_num_hints = 50 # * 12 = 600

dataset = Datasets.load('../models/4x4_all/data/datasets.pkl')
split_inputs, split_outputs = dataset.split_data([train_size_per_num_hints,
                                                  train_size_per_num_hints+valid_size_per_num_hints])

hyperparameters = {
    'device': 6,
    'dim_x': 2,
    'dim_y': 2,
    'num_iters': 32,
    'train_size': train_size_per_num_hints*12,
    'valid_size': valid_size_per_num_hints*12,
    'batch_size': 125,
    'epochs': 4,
    'valid_epochs': 25,
    'save_epochs': 25,
    'embed_size': 6,
    'hidden_layer_size': 32,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4
}

train_inputs = split_inputs[0]
train_outputs = split_outputs[0]
other_inputs = { 'validation': split_inputs[1] }
other_outputs = { 'validation': split_outputs[1] }

dim_x = hyperparameters['dim_x']
dim_y = hyperparameters['dim_y']
num_iters = hyperparameters['num_iters']
batch_size = hyperparameters['batch_size']
epochs = hyperparameters['epochs']
valid_epochs = hyperparameters['valid_epochs']
save_epochs = hyperparameters['save_epochs']
embed_size = hyperparameters['embed_size']
hidden_layer_size = hyperparameters['hidden_layer_size']
learning_rate = hyperparameters['learning_rate']
weight_decay = hyperparameters['weight_decay']
device = hyperparameters['device']

train_x = torch.stack([rrn_utils.encode_input(p) for p in train_inputs]).cuda(device)
train_y = torch.stack([rrn_utils.encode_output(p) for p in train_outputs]).cuda(device)

other_x = {}
other_y = {}
for k in other_inputs:
    other_x[k] = torch.stack([rrn_utils.encode_input(p) for p in other_inputs[k]]).cuda(device)
    other_y[k] = torch.stack([rrn_utils.encode_output(p) for p in other_outputs[k]]).cuda(device)

# model = EmbedRRN(dim_x=dim_x, dim_y=dim_y, embed_size=embed_size, hidden_layer_size=hidden_layer_size).cuda(device)
model = RRN(dim_x=dim_x, dim_y=dim_y, embed_size=embed_size, hidden_layer_size=hidden_layer_size).cuda(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


# ones = torch.ones(10, 16).cuda(device)

def closure():
    optimizer.zero_grad()
    total_loss = 0
    epoch_accuracies = []
    shuffle_indices = np.arange(len(train_x))
    np.random.shuffle(shuffle_indices)

    for i in tqdm(range(0, len(train_x), batch_size), leave=False):
        x_batch = train_x[shuffle_indices[i:i + batch_size]]
        y_batch = train_y[shuffle_indices[i:i + batch_size]]
        #         target_embed = model.embed_layer(train_y)
        predictions = model(train_x, num_iters)

    # loss = sum([F.cosine_embedding_loss(predictions[i].permute(0,2,1),
    #                         target_embed.permute(0,2,1),
    #                         ones) for i in range(len(predictions))])
    #         loss, accuracies = get_performance(model=model,
    #                                            x=x_batch,
    #                                            y=y_batch,
    #                                            no_grad=False,
    #                                            num_iters=num_iters)
    #         loss.backward()
    #         total_loss += loss
    #     train_losses.append(float(total_loss))
    #     epoch_accuracies.append(accuracies)
    #     train_accuracies.append(np.concatenate(epoch_accuracies))
    return 0  # total_loss

train_loss = optimizer.step(closure)
print(train_loss)