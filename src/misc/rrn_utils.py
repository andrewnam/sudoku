SUDOKU_PATH = '/home/ajhnam/sudoku'

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from tqdm import tqdm
from datetime import datetime
import time
import os
import pickle

import sys
sys.path.append(SUDOKU_PATH + '/src/sudoku')
sys.path.append(SUDOKU_PATH + '/src/models')

from grid_string import GridString
from rrn import RRN, collect_batches, ReEmbedRRN
from utils import print
import utils

# set random seed to 0
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.set_default_tensor_type('torch.DoubleTensor')

def get_puzzles_by_hints(shuffled_puzzles_filename=None, num_hints=None):
    if shuffled_puzzles_filename is None:
        shuffled_puzzles_filename = SUDOKU_PATH + '/data/shuffled_puzzles.txt'

    with open(shuffled_puzzles_filename) as f:
        lines = f.read().splitlines()
    all_puzzles = {}
    for line in lines:
        puzzle, solution = line.split(',')
        all_puzzles[GridString(puzzle)] = GridString(solution)

    puzzles_by_hints = {i: {} for i in range(4, 17)}
    num_cells = 16
    for p in all_puzzles:
        hints = num_cells - p.grid.count('.')
        puzzles_by_hints[hints][p] = all_puzzles[p]

    if num_hints:
        return puzzles_by_hints[num_hints]
    return puzzles_by_hints


def encode_input(grid_string: GridString):
    return torch.tensor(list(grid_string.traverse_grid()))


def encode_output(grid_string: GridString):
    return torch.tensor(list(grid_string.traverse_grid())) - 1


def get_performance(model, x, y, num_iters=32, no_grad=True ):
    """
    Predicts x using model across num_iters
    Returns loss, accuracies
    loss (torch loss): sum loss over all num_iters
    accuracies (numpy array of shape ( len(x), num_iters ) ): number of correct cells for
        each input at each timestep

    """
    def run():
        predictions = collect_batches(model(x, num_iters), num_iters)
        loss = sum([F.cross_entropy(p.permute(0, 2, 1), y) for p in predictions])
        accuracies = torch.sum(torch.argmax(predictions, dim=3) == y, dim=2).permute(1, 0)
        return loss, accuracies.cpu().data.numpy()

    if no_grad:
        with torch.no_grad():
            return run()
    else:
        return run()


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


    train_x = torch.stack([encode_input(p) for p in train_inputs]).cuda(device)
    train_y = torch.stack([encode_output(p) for p in train_outputs]).cuda(device)

    other_x = {}
    other_y = {}
    for k in other_inputs:
        other_x[k] = torch.stack([encode_input(p) for p in other_inputs[k]]).cuda(device)
        other_y[k] = torch.stack([encode_output(p) for p in other_outputs[k]]).cuda(device)

    model = RRN(dim_x=dim_x, dim_y=dim_y, embed_size=embed_size, hidden_layer_size=hidden_layer_size).cuda(device)
    if parallel:
        model = nn.DataParallel(model, device_ids=devices)
    # else:
    #     model = model.cuda(device)

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
            loss, accuracies = get_performance(model=model,
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
                loss, accuracy = get_performance(
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


def get_reembed_performance(model, x, x_oh, y, num_iters=32, no_grad=True):
    """
    Predicts x using model across num_iters
    Returns loss, accuracies
    loss (torch loss): sum loss over all num_iters
    accuracies (numpy array of shape ( len(x), num_iters ) ): number of correct cells for
        each input at each timestep

    """
    def run():
        decode_predictions, pre_predictions, sudoku_predictions = model(x_oh, num_iters)
        pre_predictions = collect_batches(pre_predictions, num_iters)
        sudoku_predictions = collect_batches(sudoku_predictions, num_iters)
        decode_loss = F.cross_entropy(decode_predictions.permute(0, 2, 1), x)
        pre_loss = sum([F.cross_entropy(p.permute(0, 2, 1), y) for p in pre_predictions])
        sudoku_loss = sum([F.cross_entropy(p.permute(0, 2, 1), y) for p in sudoku_predictions])
        pre_accuracies = torch.sum(torch.argmax(pre_predictions, dim=3) == y, dim=2).permute(1, 0)
        accuracies = torch.sum(torch.argmax(sudoku_predictions, dim=3) == y, dim=2).permute(1, 0)
        return decode_loss, pre_loss, sudoku_loss, pre_accuracies.cpu().data.numpy(), accuracies.cpu().data.numpy()

    if no_grad:
        with torch.no_grad():
            return run()
    else:
        return run()

def train_reembedrrn(hyperparameters: dict,
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

    with torch.no_grad():
        train_x = torch.stack([encode_input(p) for p in train_inputs]).cuda(device)
        train_y = torch.stack([encode_output(p) for p in train_outputs]).cuda(device) + 1
        train_x_oh = utils.one_hot_encode(train_x)

        other_x = {}
        other_x_oh = {}
        other_y = {}
        for k in other_inputs:
            other_x[k] = torch.stack([encode_input(p) for p in other_inputs[k]]).cuda(device)
            other_y[k] = torch.stack([encode_output(p) for p in other_outputs[k]]).cuda(device)
            other_x_oh[k] = utils.one_hot_encode(other_x[k])

    model = ReEmbedRRN(dim_x=dim_x, dim_y=dim_y, embed_size=embed_size, hidden_layer_size=hidden_layer_size).cuda(device)
    if parallel:
        model = nn.DataParallel(model, device_ids=devices)
    # else:
    #     model = model.cuda(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_decode_losses = []
    train_pre_losses = []
    train_sudoku_losses = []# (epoch, )
    train_pre_accuracies = []
    train_accuracies = []  # (epoch, grid, timestep)
    other_decode_losses = {name: [] for name in other_x} # (epoch, )
    other_pre_losses = {name: [] for name in other_x}  # (epoch, )
    other_sudoku_losses = {name: [] for name in other_x}  # (epoch, )
    other_accuracies = {name: [] for name in other_x} # (epoch, grid, timestep)
    other_pre_accuracies = {name: [] for name in other_x}  # (epoch, grid, timestep)
    times = []

    def closure():
        optimizer.zero_grad()
        total_loss = 0
        total_decode_loss = 0
        total_pre_loss = 0
        total_sudoku_loss = 0
        epoch_accuracies = []
        epoch_pre_accuracies = []
        shuffle_indices = np.arange(len(train_x))
        np.random.shuffle(shuffle_indices)
        for i in tqdm(range(0, len(train_x), batch_size), leave=False):
            x_batch = train_x[shuffle_indices[i:i + batch_size]]
            y_batch = train_y[shuffle_indices[i:i + batch_size]]
            x_batch_oh = train_x_oh[shuffle_indices[i:i + batch_size]]
            decode_loss, pre_loss, sudoku_loss, pre_accuracies, accuracies = get_reembed_performance(model=model,
                                               x=x_batch,
                                               x_oh=x_batch_oh,
                                               y=y_batch,
                                               no_grad=False,
                                               num_iters=num_iters)
            loss = decode_loss + sudoku_loss + pre_loss
            # loss = sudoku_loss
            loss.backward()

            total_decode_loss += decode_loss
            total_sudoku_loss += sudoku_loss
            total_pre_loss += pre_loss
            total_loss += loss

        train_decode_losses.append(float(total_decode_loss))
        train_sudoku_losses.append(float(total_sudoku_loss))
        train_pre_losses.append(float(total_pre_loss))
        epoch_accuracies.append(accuracies)
        epoch_pre_accuracies.append(pre_accuracies)
        train_accuracies.append(np.concatenate(epoch_accuracies))
        train_pre_accuracies.append(np.concatenate(epoch_pre_accuracies))
        return total_loss

    for i in tqdm(range(epochs)):
        start_time_str = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
        start_time = time.time()

        train_loss = optimizer.step(closure)

        run_validate = i == 0 or (i + 1) % valid_epochs == 0
        if run_validate:
            for name in other_x:
                decode_loss, pre_loss, sudoku_loss, pre_accuracy, accuracy = get_reembed_performance(
                                                            model=model,
                                                            x=other_x[name],
                                                            x_oh=other_x_oh[name],
                                                            y=other_y[name],
                                                            num_iters=num_iters,
                                                            no_grad=True)
                other_decode_losses[name].append(float(decode_loss))
                other_pre_losses[name].append(float(pre_loss))
                other_sudoku_losses[name].append(float(sudoku_loss))
                other_accuracies[name].append(accuracy)
                other_pre_accuracies[name].append(pre_accuracy)
        if (i + 1) % save_epochs == 0:
            model_filename = "./checkpoints/epoch_{}.mdl".format(i + 1)
            train_data_filename = "./logs/training.pkl"
            print("Saving model to {}".format(model_filename))
            torch.save(model.state_dict(), model_filename)
            with open(train_data_filename, 'wb') as f:
                pickle.dump({'hyperparameters': hyperparameters,
                             'train_decode_losses': train_decode_losses,
                             'train_sudoku_losses': train_sudoku_losses,
                             'train_pre_losses': train_pre_losses,
                             'train_accuracies': train_accuracies,
                             'train_pre_accuracies': train_pre_accuracies,
                             'other_decode_losses': other_decode_losses,
                             'other_sudoku_losses': other_sudoku_losses,
                             'other_pre_losses': other_pre_losses,
                             'other_accuracies': other_accuracies,
                             'other_pre_accuracies': other_pre_accuracies,
                             'times': times}, f)

        end_time_str = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
        end_time = time.time()
        runtime = end_time - start_time
        times.append({
            'start_time': start_time_str,
            'end_time': end_time_str,
            'runtime': runtime
        })
        print("duration: {}s\t iter: {}\t| d_loss: {}\t| p_loss: {}\t s_loss: {}\t| p_accuracy: {}\t accuracy: {}".format(
            round(runtime, 1),
            i,
            round(train_decode_losses[-1], 3),
            round(train_pre_losses[-1], 3),
            round(train_sudoku_losses[-1], 3),
            round(np.mean(train_pre_accuracies[-1][:, -1]), 3),
            round(np.mean(train_accuracies[-1][:, -1]), 3)))
        if run_validate:
            for name in sorted(other_x):
                print("data: {}\t| d_loss: {}\t| p_loss: {}\t s_loss: {}\t| p_accuracy: {}\t| accuracy: {}".format(
                    name,
                    round(other_decode_losses[name][-1], 3),
                    round(other_pre_losses[name][-1], 3),
                    round(other_sudoku_losses[name][-1], 3),
                    round(np.mean(other_pre_accuracies[name][-1][:, -1]), 3),
                    round(np.mean(other_accuracies[name][-1][:, -1]), 3)))

    model_filename = "./model.mdl"
    print("Saving model to {}".format(model_filename))
    torch.save(model.state_dict(), model_filename)
    return model