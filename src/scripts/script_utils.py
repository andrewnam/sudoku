SUDOKU_PATH = '/home/ajhnam/sudoku'

import torch
import torch.nn.functional as F

import torch.optim as optim
import numpy as np
import random
from tqdm import tqdm
from datetime import datetime
import time
import pickle

import sys
sys.path.append(SUDOKU_PATH + '/src/sudoku')
sys.path.append(SUDOKU_PATH + '/src/models')

from grid_string import GridString
from rrn import RRN
from andrew_utils import print

# set random seed to 0
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.set_default_tensor_type('torch.DoubleTensor')


def encode_input(grid_string: GridString):
    return torch.tensor(list(grid_string.traverse_grid()))


def encode_output(grid_string: GridString):
    return torch.tensor(list(grid_string.traverse_grid())) - 1


def get_performance(model, x, y, num_iters=32):
    """
    Predicts x using model across num_iters
    Returns loss, accuracies
    loss (torch loss): sum loss over all num_iters
    accuracies (numpy array of shape ( len(x), num_iters ) ): number of correct cells for
        each input at each timestep

    """
    predictions = model(x, num_iters)
    loss = sum([F.cross_entropy(p.permute(0, 2, 1), y) for p in predictions])
    accuracies = torch.sum(torch.argmax(predictions, dim=3) == y, dim=2).permute(1, 0)
    return loss, accuracies.cpu().data.numpy()

def train_rrn(hyperparameters: dict, data: dict):
    model_name = hyperparameters['model_name']
    device = hyperparameters['device']
    dim_x = hyperparameters['dim_x']
    dim_y = hyperparameters['dim_y']
    num_iters = hyperparameters['num_iters']
    train_size = hyperparameters['train_size']
    valid_size = hyperparameters['valid_size']
    test_size = hyperparameters['test_size']
    batch_size = hyperparameters['batch_size']
    epochs = hyperparameters['epochs']
    save_epochs = hyperparameters['save_epochs']
    embed_size = hyperparameters['embed_size']
    hidden_layer_size = hyperparameters['hidden_layer_size']
    learning_rate = hyperparameters['learning_rate']
    weight_decay = hyperparameters['weight_decay']

    train_inputs = data['train_inputs']
    train_outputs = data['train_outputs']
    valid_inputs = data['valid_inputs']
    valid_outputs = data['valid_outputs']
    test_inputs = data['test_inputs']
    test_outputs = data['test_outputs']

    all_train_x = torch.stack([encode_input(p) for p in train_inputs])
    all_train_y = torch.stack([encode_output(p) for p in train_outputs])
    all_valid_x = torch.stack([encode_input(p) for p in valid_inputs])
    all_valid_y = torch.stack([encode_output(p) for p in valid_outputs])
    all_test_x = torch.stack([encode_input(p) for p in test_inputs])
    all_test_y = torch.stack([encode_output(p) for p in test_outputs])

    model = RRN(dim_x=dim_x, dim_y=dim_y, embed_size=embed_size, hidden_layer_size=hidden_layer_size).cuda(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_losses = []  # epoch x batch
    train_accuracies = []  # epoch x batch x grid x timestep
    valid_losses = []  # epoch x batch
    valid_accuracies = []  # epoch x batch x grid x timestep
    times = []

    train_x = all_train_x[:train_size].cuda(device)
    train_y = all_train_y[:train_size].cuda(device)
    valid_x = all_valid_x[:valid_size].cuda(device)
    valid_y = all_valid_y[:valid_size].cuda(device)
    test_x = all_test_x[:test_size].cuda(device)
    test_y = all_test_y[:test_size].cuda(device)

    def closure():
        optimizer.zero_grad()
        total_loss = 0
        shuffle_indices = np.arange(len(train_x))
        np.random.shuffle(shuffle_indices)
        for i in tqdm(range(0, len(train_x), batch_size), leave=False):
            x_batch = train_x[shuffle_indices[i:i + batch_size]]
            y_batch = train_y[shuffle_indices[i:i + batch_size]]
            loss, accuracies = get_performance(model, x_batch, y_batch, num_iters)
            loss.backward()
            total_loss += loss

            train_losses[-1].append(float(loss))
            train_accuracies[-1].append(accuracies)
        return total_loss

    for i in tqdm(range(epochs)):
        start_time_str = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
        start_time = time.time()

        train_losses.append([])
        train_accuracies.append([])

        train_loss = optimizer.step(closure)
        train_accuracies[-1] = np.array(train_accuracies[-1])
        valid_loss, valid_accuracy = get_performance(model, valid_x, valid_y, num_iters)
        valid_losses.append(float(valid_loss))
        valid_accuracies.append(valid_accuracy)

        train_accuracies[-1] = np.array(train_accuracies[-1])

        train_loss = round(float(train_loss), 3)
        train_accuracy = round(np.mean(train_accuracies[-1][:, :, -1]), 3)
        valid_loss = round(valid_losses[-1], 3)
        valid_accuracy = round(np.mean(valid_accuracies[-1][:, -1]), 3)

        end_time_str = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
        end_time = time.time()
        runtime = end_time - start_time
        times.append({
            'start_time': start_time_str,
            'end_time': end_time_str,
            'runtime': runtime
        })
        print("({}s): Iter {}\t| TrLoss {}\t| VLoss {}\t| TrAcc {}\t| VAcc {}".format(
            round(runtime, 1),
            i, train_loss, valid_loss, train_accuracy, valid_accuracy))

        if (i + 1) % save_epochs == 0:
            model_filename = SUDOKU_PATH + "/models/{}_{}.mdl".format(model_name, i + 1)
            train_data_filename = SUDOKU_PATH + "/pickles/{}.pkl".format(model_name)
            print("Saving model to {}".format(model_filename))
            torch.save(model.state_dict(), model_filename)
            with open(train_data_filename, 'wb') as f:
                pickle.dump({'hyperparameters': hyperparameters,
                             'train_losses': train_losses,
                             'train_accuracies': train_accuracies,
                             'valid_losses': valid_losses,
                             'valid_accuracies': valid_accuracies,
                             'times': times}, f)
            test_loss, test_accuracy = get_performance(model, test_x, test_y, num_iters)
            test_loss = round(float(test_loss), 3)
            test_accuracy = round(np.mean(test_accuracy[:, -1]), 3)
            print("TeLoss {}\t| TeAcc {}".format(test_loss, test_accuracy))

    return model