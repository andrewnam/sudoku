# from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from datetime import datetime
from sudoku2.grid import GridString
from utils import Dict
import utils
import time
import logging

utils.setup_logging()
logger = logging.getLogger(__name__)

def one_hot_encode(x):
    encoding = torch.zeros(x.shape + (torch.max(x)+1, ))
    if x.is_cuda:
        encoding = encoding.cuda(x.get_device())
    dim = len(x.shape)
    x = x.view(x.shape + (1, ))
    return encoding.scatter_(dim, x, 1).type(torch.long)

def collect_batches(outputs, iters):
    """
    Collects parallel-gpu batches from running forward on a DataParallel RRN model.
    If x and y are both of shape (n, c) where n = number of puzzles and c = number of cells,
     a forward pass with iters iterations will return a tensor of shape (iters, n, c, max_digit).
    If using DataParallel with d devices, the shape will instead be (iters*d, n/d, c, max_digit).
    This function takes the latter and returns the former and is safe to use even when d is 1.
    Therefore, it is advised to always call collect_batches(model(x, iters), iters).
    This cannot be put into the forward function since it needs to wrap around the DataParallel abstraction.

    :param outputs: Tensor of shape (iters*d, n/d, c, max_digit)
    :param iters: Number of iters when running forward
    :return: Tensor of shape (iters, n, c, max_digit)
    """
    return torch.cat([outputs[i:i + iters] for i in range(0, len(outputs), iters)], dim=1)

def encode_grid_string(grid_string: GridString):
    return torch.tensor(list(grid_string.traverse_grid()))


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


def log_performance(name, iter, loss, accuracy):
    logger.info(f"{name} iter: {iter}\t| loss: {round(float(loss), 3)}\t| accuracy: {round(accuracy, 3)}")


class Solver:

    def __init__(self, model, filepath):
        self.model = model
        self.filepath = filepath

        self.dim_x = model.dim_x
        self.dim_y = model.dim_y
        self.max_digit = self.model.max_digit

        # training parameters
        self.weight_decay = 1e-4
        self.learning_rate = 1e-3
        self.num_iters = 16
        self.batch_size = 128
        self.epochs = 200
        self.save_epochs = 25
        self.device = None

        # datasets
        self.train_inputs = []
        self.train_outputs = []
        self.other_inputs = {}
        self.other_outputs = {}

        # train data
        self.train_losses = []  # (epoch, )
        self.train_accuracies = []  # (epoch, grid, timestep)
        self.other_losses = Dict(list) # (epoch, )
        self.other_accuracies = Dict(list) # (epoch, grid, timestep)
        self.times = []
        logger.info(f"Constructed solver for filepath: {self.filepath}")


    def init_paths(self):
        if not os.path.exists(self.filepath + '/checkpoints'):
            os.makedirs(self.filepath + '/checkpoints')
        if not os.path.exists(self.filepath + '/logs'):
            os.makedirs(self.filepath + '/logs')

    def check_data(self):
        assert len(self.train_inputs) == len(self.train_outputs)
        assert set(self.other_inputs.keys()) == set(self.other_outputs.keys())
        for k in self.other_inputs:
            assert len(self.other_inputs[k]) == len(self.other_outputs[k])

    def prepare_data(self, input, output):
        input = torch.stack([encode_grid_string(GridString.load(s)) for s in input]).type(torch.long)
        output = torch.stack([encode_grid_string(GridString.load(s)) for s in output]).type(torch.long) - 1
        if self.device is not None:
            input = input.cuda(self.device)
            output = output.cuda(self.device)
        return input, output

    def make_closure(self, optimizer, inputs, outputs):
        def closure():
            optimizer.zero_grad()
            total_loss = 0
            epoch_accuracies = []
            shuffle_indices = np.arange(len(inputs))
            np.random.shuffle(shuffle_indices)
            for i in range(0, len(inputs), self.batch_size):
            # for i in tqdm(range(0, len(inputs), self.batch_size), leave=False):
                x_batch = inputs[shuffle_indices[i:i + self.batch_size]]
                y_batch = outputs[shuffle_indices[i:i + self.batch_size]]
                loss, accuracies = get_performance(model=self.model,
                                                             x=x_batch,
                                                             y=y_batch,
                                                             no_grad=False,
                                                             num_iters=self.num_iters)
                loss.backward()
                total_loss += loss

            self.train_losses.append(float(total_loss))
            epoch_accuracies.append(accuracies)
            self.train_accuracies.append(np.concatenate(epoch_accuracies))
            return total_loss
        return closure

    def train(self):
        self.check_data()
        self.init_paths()
        train_inputs, train_outputs = self.prepare_data(self.train_inputs, self.train_outputs)
        other_inputs, other_outputs = {}, {}
        for k in self.other_inputs:
            i, o = self.prepare_data(self.other_inputs[k], self.other_outputs[k])
            other_inputs[k] = i
            other_outputs[k] = o

        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        closure = self.make_closure(optimizer, train_inputs, train_outputs)

        # for i in tqdm(range(self.epochs)):
        for i in range(self.epochs):
            start_time_str = utils.now()
            start_time = time.time()

            train_loss = optimizer.step(closure)

            run_validate = i == 0 or (i + 1) % self.save_epochs == 0
            if run_validate:
                for name in other_inputs:
                    loss, accuracy = get_performance(
                        model=self.model,
                        x=other_inputs[name],
                        y=other_outputs[name],
                        num_iters=self.num_iters,
                        no_grad=True)
                    self.other_losses[name].append(float(loss))
                    self.other_accuracies[name].append(accuracy)

            if (i + 1) % self.save_epochs == 0:
                filename = f"{self.filepath}/checkpoints/epoch_{i+1}.json"
                utils.save(filename, self)

            end_time_str = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
            end_time = time.time()
            runtime = end_time - start_time
            self.times.append({
                'start_time': start_time_str,
                'end_time': end_time_str,
                'runtime': runtime
            })

            log_performance('train', i, train_loss, np.mean(self.train_accuracies[-1][:, -1]))
            if run_validate:
                for name in sorted(self.other_inputs):
                    log_performance(name, i, self.other_losses[name][-1], np.mean(self.other_accuracies[name][-1][:, -1]))

        filename = f"{self.filepath}/train_log.json"
        utils.save(filename, self)
