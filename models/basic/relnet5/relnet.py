"""
Relnet2: New Feed forward relnet
Relnet3: Relnet 2 didn't train. Appending original puzzle into embedding.
Relnet4: Make the network smaller. Embed size = 3.
Relnet5: Reduce num_iters to 12.
"""

SUDOKU_PATH = '/home/ajhnam/sudoku'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
from datetime import datetime
from tqdm import tqdm
import pickle
import os
import sys
sys.path.append(SUDOKU_PATH + '/src/sudoku')
sys.path.append(SUDOKU_PATH + '/src/misc')
sys.path.append(SUDOKU_PATH + '/src/models')

import sudoku_model_utils
from dataset import Datasets
from mlp import MLP
from relational_network import RelationalLayer
import rrn_utils
import andrew_utils as utils


class RelNet(nn.Module):

    def __init__(self, dim_x, dim_y, embed_size, hidden_layer_size):
        super(RelNet, self).__init__()
        self.max_digit = dim_x*dim_y
        self.embed_size = embed_size
        self.hidden_layer_size = hidden_layer_size

        self.edges = sudoku_model_utils.determine_edges(dim_x, dim_y)

        self.embed_layer = nn.Embedding(self.max_digit+1, self.embed_size)
        self.embed_mlp = MLP([self.embed_size + self.max_digit,
                              self.hidden_layer_size,
                              self.hidden_layer_size,
                              self.hidden_layer_size])
        self.rel_layer = RelationalLayer([self.hidden_layer_size,
                                          self.hidden_layer_size,
                                          self.hidden_layer_size,
                                          self.hidden_layer_size],
                                         self.edges)
        self.decode_mlp = MLP([self.hidden_layer_size,
                      self.hidden_layer_size,
                      self.hidden_layer_size,
                      self.max_digit])

    def forward(self, x_grid_form, x_prob_form, iters):
        device = x_grid_form.get_device()

        batch_size = len(x_grid_form)
        num_nodes = self.max_digit ** 2
        outputs = torch.empty(iters, batch_size, num_nodes, self.max_digit).cuda(device)

        embedding = self.embed_layer(x_grid_form)
        x = x_prob_form

        for i in range(iters):
            x = torch.cat([embedding, x], dim=len(x.shape)-1)
            x = self.embed_mlp(x)
            x = self.rel_layer(x)
            x = self.decode_mlp(x)

            outputs[i] = x
            x = F.softmax(x, dim=2)

        return outputs


def get_performance(model, x_grid, x_prob, y, num_iters=32, no_grad=True):
    """
    Predicts x using model across num_iters
    Returns loss, accuracies
    loss (torch loss): sum loss over all num_iters
    accuracies (numpy array of shape ( len(x), num_iters ) ): number of correct cells for
        each input at each timestep

    """
    def run():
        predictions = model(x_grid, x_prob, num_iters)
        loss = sum([F.cross_entropy(p.permute(0, 2, 1), y) for p in predictions])
        accuracies = torch.sum(torch.argmax(predictions, dim=3) == y, dim=2).permute(1, 0)
        return loss, accuracies.cpu().data.numpy()

    if no_grad:
        with torch.no_grad():
            return run()
    else:
        return run()

data_path = '../4x4_all_reimbed/data/datasets.pkl'
train_size_per_num_hints = 100 # * 12 = 1200
valid_size_per_num_hints = 5 # * 12 = 60
hp = {
    'device': 6,
    'dim_x': 2,
    'dim_y': 2,
    'num_iters': 12,
    'train_size_per_num_hints': train_size_per_num_hints,
    'valid_size_per_num_hints': valid_size_per_num_hints,
    'train_size': train_size_per_num_hints*12,
    'valid_size': valid_size_per_num_hints*12,
    'batch_size': 500,
    'epochs': 600,
    'valid_epochs': 25,
    'save_epochs': 25,
    'embed_size': 3,
    'hidden_layer_size': 32,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'data_path': os.path.abspath(data_path)
}

dataset = Datasets.load(data_path)
split_inputs, split_outputs = dataset.split_data([train_size_per_num_hints,
                                                  train_size_per_num_hints+valid_size_per_num_hints])

train_inputs = split_inputs[0]
train_outputs = split_outputs[0]
other_inputs = {'validation': split_inputs[1]}
other_outputs = {'validation': split_outputs[1]}


model = RelNet(dim_x=hp['dim_x'],
               dim_y=hp['dim_y'],
               embed_size=hp['embed_size'],
               hidden_layer_size=hp['hidden_layer_size']).cuda(hp['device'])


optimizer = optim.Adam(model.parameters(),
                       lr=hp['learning_rate'],
                       weight_decay=hp['weight_decay'])

train_x_grid = torch.stack([rrn_utils.encode_input(p) for p in train_inputs])
train_x_prob = utils.puzzle_as_dist(train_x_grid).cuda(hp['device'])
train_x_grid = train_x_grid.cuda(hp['device'])
train_y = torch.stack([rrn_utils.encode_output(p) for p in train_outputs]).cuda(hp['device'])

other_x_grid = {}
other_x_prob = {}
other_y = {}
for k in other_inputs:
    other_x_grid[k] = torch.stack([rrn_utils.encode_input(p) for p in other_inputs[k]])
    other_x_prob[k] = utils.puzzle_as_dist(other_x_grid[k]).cuda(hp['device'])
    other_x_grid[k] = other_x_grid[k].cuda(hp['device'])
    other_y[k] = torch.stack([rrn_utils.encode_output(p) for p in other_outputs[k]]).cuda(hp['device'])

train_losses = []  # (epoch)
train_accuracies = []  # (epoch, grid, timestep)
other_losses = {name: [] for name in other_x_grid} # (epoch)
other_accuracies = {name: [] for name in other_x_grid} # (epoch, grid, timestep)
times = []

if not os.path.exists('./checkpoints'):
    os.makedirs('./checkpoints')
if not os.path.exists('./logs'):
    os.makedirs('./logs')

def closure():
    optimizer.zero_grad()
    total_loss = 0
    epoch_accuracies = []
    shuffle_indices = np.arange(len(train_x_grid))
    np.random.shuffle(shuffle_indices)
    for i in tqdm(range(0, len(train_x_grid), hp['batch_size']), leave=False):
        x_grid_batch = train_x_grid[shuffle_indices[i:i + hp['batch_size']]]
        x_prob_batch = train_x_prob[shuffle_indices[i:i + hp['batch_size']]]
        y_batch = train_y[shuffle_indices[i:i + hp['batch_size']]]
        loss, accuracies = get_performance(model=model,
                                           x_grid=x_grid_batch,
                                           x_prob=x_prob_batch,
                                           y=y_batch,
                                           no_grad=False,
                                           num_iters=hp['num_iters'])
        loss.backward()
        total_loss += loss

    train_losses.append(float(total_loss))
    epoch_accuracies.append(accuracies)
    train_accuracies.append(np.concatenate(epoch_accuracies))
    return total_loss

for i in tqdm(range(hp['epochs'])):
    start_time_str = utils.now()
    start_time = time.time()

    train_loss = optimizer.step(closure)

    run_validate = i == 0 or (i + 1) % hp['valid_epochs'] == 0
    if run_validate:
        for name in other_x_grid:
            loss, accuracy = get_performance(
                                model=model,
                                x_grid=other_x_grid[name],
                                x_prob=other_x_prob[name],
                                y=other_y[name],
                                num_iters=hp['num_iters'],
                                no_grad=True)
            other_losses[name].append(float(loss))
            other_accuracies[name].append(accuracy)

    if (i + 1) % hp['save_epochs'] == 0:
        model_filename = "./checkpoints/epoch_{}.mdl".format(i + 1)
        train_data_filename = "./logs/training.pkl"
        print("Saving model to {}".format(model_filename))
        torch.save(model.state_dict(), model_filename)
        with open(train_data_filename, 'wb') as f:
            pickle.dump({'hyperparameters': hp,
                         'train_losses': train_losses,
                         'train_accuracies': train_accuracies,
                         'other_losses': other_losses,
                         'other_accuracies': other_accuracies,
                         'times': times}, f)

    end_time_str = utils.now()
    end_time = time.time()
    runtime = end_time - start_time
    times.append({
        'start_time': start_time_str,
        'end_time': end_time_str,
        'runtime': runtime
    })
    print("duration: {}s\t iter: {}\t| loss: {}\t| accuracy: {}".format(
        round(runtime, 1),
        i,
        round(float(train_loss), 3),
        round(np.mean(train_accuracies[-1][:, -1]), 3)))
    if run_validate:
        for name in sorted(other_x_grid):
            print("data: {}\t| loss: {}\t| accuracy: {}".format(
                name,
                round(other_losses[name][-1], 3),
                round(np.mean(other_accuracies[name][-1][:, -1]), 3)))

model_filename = "./model.mdl"
print("Saving model to {}".format(model_filename))
torch.save(model.state_dict(), model_filename)
