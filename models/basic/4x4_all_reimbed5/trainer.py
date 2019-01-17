"""
Reimbed 4: Standard RRN on 1200 train size and 60 valid size
Reimbed 5: The embedding layer is substituted with an actual linear layer.
    This requires that the input is a one-hot.
"""

SUDOKU_PATH = '/home/ajhnam/sudoku'

import torch
import torch.nn as nn
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

from dataset import Datasets
import rrn
from mlp import MLP
import rrn_utils
import utils

class NewRRN(nn.Module):
    def __init__(self, dim_x, dim_y, embed_size=16, hidden_layer_size=96):
        super(NewRRN, self).__init__()
        self.max_digit = dim_x * dim_y
        self.embed_size = embed_size
        self.hidden_layer_size = hidden_layer_size

        self.edges = rrn.determine_edges(dim_x, dim_y)

        self.embed_layer = nn.Linear(self.max_digit + 1, self.embed_size)
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


def train_rrn(hyperparameters: dict,
              train_inputs: list,
              train_outputs: list,
              other_inputs: dict=None,
              other_outputs: dict=None):
    """
    :param hyperparameters: Check below for what fields must exist in hyperparameters
    :param train_inputs: list of GridStrings
    :param train_outputs: list of GridStrings, corresponding in index to train_inputs
    :param other_inputs: dictionary of GridStrings where the key is name of the dataset
    :param other_outputs: dictionary of GridStrings where the key is name of the dataset,
        corresponding in index to inputs of same name
    :return:
    """

    if other_inputs is None:
        other_inputs = {}
    if other_outputs is None:
        other_outputs = {}
    assert set(other_inputs.keys()) == set(other_outputs.keys())

    if not os.path.exists('./checkpoints'):
        os.makedirs('./checkpoints')
    if not os.path.exists('./logs'):
        os.makedirs('./logs')


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
    parallel = False

    if 'devices' in hyperparameters:
        if len(hyperparameters['devices']) > 1:
            devices = hyperparameters['devices']
            parallel = True
        device = hyperparameters['devices'][0]
    else:
        device = hyperparameters['device']


    train_x = utils.one_hot_encode(torch.stack([rrn_utils.encode_input(p) for p in train_inputs]).cuda(device))
    train_y = torch.stack([rrn_utils.encode_output(p) for p in train_outputs]).cuda(device)

    other_x = {}
    other_y = {}
    for k in other_inputs:
        other_x[k] = utils.one_hot_encode(torch.stack([rrn_utils.encode_input(p) for p in other_inputs[k]]).cuda(device))
        other_y[k] = torch.stack([rrn_utils.encode_output(p) for p in other_outputs[k]]).cuda(device)

    model = NewRRN(dim_x=dim_x, dim_y=dim_y, embed_size=embed_size, hidden_layer_size=hidden_layer_size).cuda(device)
    if parallel:
        model = nn.DataParallel(model, device_ids=devices)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_losses = []  # (epoch, )
    train_accuracies = []  # (epoch, grid, timestep)
    other_losses = {name: [] for name in other_x} # (epoch, )
    other_accuracies = {name: [] for name in other_x} # (epoch, grid, timestep)
    times = []

    def closure():
        optimizer.zero_grad()
        total_loss = 0
        epoch_accuracies = []
        shuffle_indices = np.arange(len(train_x))
        np.random.shuffle(shuffle_indices)
        for i in tqdm(range(0, len(train_x), batch_size), leave=False):
            x_batch = train_x[shuffle_indices[i:i + batch_size]]
            y_batch = train_y[shuffle_indices[i:i + batch_size]]
            loss, accuracies = rrn_utils.get_performance(model=model,
                                               x=x_batch,
                                               y=y_batch,
                                               no_grad=False,
                                               num_iters=num_iters)
            loss.backward()
            total_loss += loss

        train_losses.append(float(total_loss))
        epoch_accuracies.append(accuracies)
        train_accuracies.append(np.concatenate(epoch_accuracies))
        return total_loss

    for i in tqdm(range(epochs)):
        start_time_str = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
        start_time = time.time()

        train_loss = optimizer.step(closure)

        run_validate = i == 0 or (i + 1) % valid_epochs == 0
        if run_validate:
            for name in other_x:
                loss, accuracy = rrn_utils.get_performance(
                                    model=model,
                                    x=other_x[name],
                                    y=other_y[name],
                                    num_iters=num_iters,
                                    no_grad=True)
                other_losses[name].append(float(loss))
                other_accuracies[name].append(accuracy)

        if (i + 1) % save_epochs == 0:
            model_filename = "./checkpoints/epoch_{}.mdl".format(i + 1)
            train_data_filename = "./logs/training.pkl"
            print("Saving model to {}".format(model_filename))
            torch.save(model.state_dict(), model_filename)
            with open(train_data_filename, 'wb') as f:
                pickle.dump({'hyperparameters': hyperparameters,
                             'train_losses': train_losses,
                             'train_accuracies': train_accuracies,
                             'other_losses': other_losses,
                             'other_accuracies': other_accuracies,
                             'times': times}, f)

        end_time_str = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
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
            for name in sorted(other_x):
                print("data: {}\t| loss: {}\t| accuracy: {}".format(
                    name,
                    round(other_losses[name][-1], 3),
                    round(np.mean(other_accuracies[name][-1][:, -1]), 3)))

    model_filename = "./model.mdl"
    print("Saving model to {}".format(model_filename))
    torch.save(model.state_dict(), model_filename)
    return model


train_size_per_num_hints = 100 # * 12 = 1200
valid_size_per_num_hints = 5 # * 12 = 60

hyperparameters = {
    'device': 5,
    'dim_x': 2,
    'dim_y': 2,
    'num_iters': 32,
    'train_size': train_size_per_num_hints*12,
    'valid_size': valid_size_per_num_hints*12,
    'batch_size': 500,
    'epochs': 600,
    'valid_epochs': 25,
    'save_epochs': 25,
    'embed_size': 6,
    'hidden_layer_size': 32,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4
}

dataset = Datasets.load('../4x4_all_reimbed/data/datasets.pkl')


split_inputs, split_outputs = dataset.split_data([train_size_per_num_hints,
                                                  train_size_per_num_hints+valid_size_per_num_hints])


train_rrn(hyperparameters,
          train_inputs = split_inputs[0],
          train_outputs = split_outputs[0],
          other_inputs = {'validation': split_inputs[1]},
          other_outputs = {'validation': split_outputs[1]})
