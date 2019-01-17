"""
Reimbed 4: Standard RRN on 1200 train size and 60 valid size
new_imbed: Replace embedding layer with embedding layer from Reimbed4
Run on yes_1 datasets
new_embed2: Make last layer of last MLP the decoding layer for the embedding layer
    trained on the solutions of the 1200 training data over 1000 epochs
    (literally takes 1 second to train).
new_embed3: Use embedding from small_embed/emb3
"""

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
sys.path.append(SUDOKU_PATH + '/src/misc')
sys.path.append(SUDOKU_PATH + '/src/models')

from dataset import Dataset
import rrn_utils
import rrn
from rrn import RRN
from mlp import MLP
from dataset import Datasets


class DigitEncoder(nn.Module):
    def __init__(self, orig_embed_layer):
        super(DigitEncoder, self).__init__()
        self.num_embeddings = orig_embed_layer.num_embeddings
        self.embedding_dim = orig_embed_layer.embedding_dim

        self.encoder = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.encoder.load_state_dict(orig_embed_layer.state_dict())
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.decoder = nn.Linear(self.embedding_dim, self.num_embeddings-1)

    def forward(self, x):
        return self.decoder(self.encoder(x))

def create_encoder():
    all_train_dir = '../small_embed/emb3/'
    dataset = Datasets.load('../4x4_all_reimbed/data/datasets.pkl')

    model_filename = all_train_dir + "model.mdl"
    train_log_filename = all_train_dir + "logs/training.pkl"

    with open(train_log_filename, 'rb') as f:
        train_log = pickle.load(f)
    hp = train_log['hyperparameters']

    model = rrn.RRN(dim_x=hp['dim_x'],
                    dim_y=hp['dim_y'],
                    embed_size=hp['embed_size'],
                    hidden_layer_size=hp['hidden_layer_size'])
    model.load_state_dict(torch.load(model_filename), strict=False)
    model.eval()

    for k, v in model.named_modules():
        if k == 'embed_layer':
            orig_embed_layer = v

    device = 4
    split_inputs, split_outputs = dataset.split_data([100])
    train_inputs = split_outputs[0]
    train_outputs = split_outputs[0]
    train_x = torch.stack([rrn_utils.encode_input(p) for p in train_inputs]).cuda(device)
    train_y = torch.stack([rrn_utils.encode_output(p) for p in train_outputs]).cuda(device)

    digitEncoder = DigitEncoder(orig_embed_layer).cuda(device)
    optimizer = optim.Adam(digitEncoder.parameters())

    def closure():
        optimizer.zero_grad()
        predictions = digitEncoder(train_x)
        loss = F.cross_entropy(predictions.permute(0, 2, 1), train_y)
        loss.backward()

        return loss

    for i in range(1000):
        optimizer.step(closure)

    encoder = nn.Embedding(digitEncoder.num_embeddings, digitEncoder.embedding_dim)
    encoder.load_state_dict(digitEncoder.encoder.state_dict())
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    decoder = nn.Linear(digitEncoder.embedding_dim, digitEncoder.num_embeddings-1)
    decoder.load_state_dict(digitEncoder.decoder.state_dict())
    decoder.eval()
    for p in decoder.parameters():
        p.requires_grad = False

    return encoder, decoder


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


    train_x = torch.stack([rrn_utils.encode_input(p) for p in train_inputs]).cuda(device)
    train_y = torch.stack([rrn_utils.encode_output(p) for p in train_outputs]).cuda(device)

    other_x = {}
    other_y = {}
    for k in other_inputs:
        other_x[k] = torch.stack([rrn_utils.encode_input(p) for p in other_inputs[k]]).cuda(device)
        other_y[k] = torch.stack([rrn_utils.encode_output(p) for p in other_outputs[k]]).cuda(device)

    model = RRN(dim_x=dim_x, dim_y=dim_y, embed_size=embed_size, hidden_layer_size=hidden_layer_size)

    encoder, decoder = create_encoder()
    model.embed_layer = encoder
    max_digit = dim_x*dim_y
    model.r = MLP([hidden_layer_size,
                      hidden_layer_size,
                      embed_size,
                      max_digit])
    model.r.layers[-1] = decoder

    model = model.cuda(device)
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

train_set = Dataset.load('../4x4_yes1/data/without_one.pkl')
test_set = Dataset.load('../4x4_yes1/data/with_one.pkl')


hyperparameters = {
    'device': 7,
    'dim_x': 2,
    'dim_y': 2,
    'num_iters': 32,
    'train_size': 1200,
    'valid_size': 240,
    'batch_size': 400,
    'epochs': 600,
    'valid_epochs': 25,
    'save_epochs': 25,
    'embed_size': 3,
    'hidden_layer_size': 32,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4
}

train_rrn(hyperparameters,
              train_inputs = train_set.get_input_data(0, 1200),
              train_outputs = train_set.get_output_data(0, 1200),
              other_inputs = { 'validation': train_set.get_input_data(1200, 1440),
                               'test': test_set.get_input_data(0, 240)},
              other_outputs = { 'validation': train_set.get_output_data(1200, 1440),
                               'test': test_set.get_output_data(0, 240)})
