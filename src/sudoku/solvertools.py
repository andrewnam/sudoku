import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import itertools
import random
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm, tqdm_notebook

from board import Board
from solutions import Solutions
import utils

np.random.seed(0)
torch.manual_seed(0)
torch.set_default_tensor_type('torch.DoubleTensor')


def vector_encode(board_string: str):
    """
    :param board_string: output of board.stringify()
    :return: numpy array length max_dim^3 where
        if the cell entry exists, the cell is a one-hot
        else, the cell is a uniform distribution of 1/max_dim
    """

    dim_x, dim_y, board = board_string.split('.')
    max_digit = int(dim_x) * int(dim_y)
    vector = np.zeros((max_digit*len(board),), dtype=np.float64)
    for i in range(len(board)):
        if board[i] != '0':
            vector[i*max_digit + int(board[i]) - 1] = 1
        else:
            vector[i*max_digit:(i+1)*max_digit] = 1/max_digit
    return vector


def get_board_entries(board_string: str):
    """
    Takes a board.stringify() output and returns numpy array of its cell values
    :param board_string: string
    :return: (max_dim, ) numpy array
    """
    return np.array(list(board_string[4:]), dtype=np.int)


def generate_derivatives(puzzles: list, solutions: Solutions, num_derivatives: int=0):
    """
    :param puzzles: list of seed Boards
    :param solutions: Solutions containing the seed_puzzles
    :param num_derivatives: max number of derivatives to generate per seed. Default: 0 -> all
    :return:
        derivative_puzzles: dict - seed board string -> list of derivative board strings
        derivative_puzzle_solutions: dict - derivative board string -> derivative board solution string
    """
    max_digit = puzzles[0].board.shape[0]
    all_digits_string = ''.join([str(i) for i in range(1, max_digit+1)])
    permutations = [''.join(lst) for lst in itertools.permutations(all_digits_string, max_digit)]
    random.shuffle(permutations)
    if num_derivatives > 0:
        permutations = permutations[:num_derivatives]
    derivative_puzzles = {}
    derivative_puzzle_solutions = {}
    for puzzle in puzzles:
        puzzle_string = puzzle.stringify()
        derivatives = Board.shuffle_numbers(puzzle_string, permutations)
        derivative_solutions = Board.shuffle_numbers(solutions[puzzle].stringify(), permutations)
        derivative_puzzles[puzzle_string] = derivatives
        for i in range(len(permutations)):
            derivative_puzzle_solutions[derivatives[i]] = derivative_solutions[i]
    return derivative_puzzles, derivative_puzzle_solutions


def split_data(data: iter, boundaries: iter):
    """
    Shuffles and splits data according to the sorted boundaries set
    :param data: iterable
    :param boundaries: iterable of boundaries in (0, 1)
    :return:
    """
    assert boundaries[0] > 0 and boundaries[-1] < 1 and boundaries == sorted(boundaries)
    assert len(data) > len(boundaries)

    data = list(data)
    np.random.shuffle(data)

    split = []
    last_boundary = 0
    for boundary in boundaries + [1]:
        next_boundary = int(len(data) * boundary)
        split.append(data[last_boundary:next_boundary])
        last_boundary = next_boundary
    return split


def generate_dataset(seed_puzzles: list, solutions: Solutions, boundaries: iter, num_derivatives: int=0):
    """

    :param seed_puzzles: list of seed Boards
    :param solutions: Solutions containing the seed_puzzles
    :param boundaries: iterable of boundaries in (0, 1)
    :param num_derivatives: max number of derivatives to generate per seed. Default: 0 -> all
    :return:
    """
    derivative_puzzles, derivative_puzzle_solutions = generate_derivatives(seed_puzzles, solutions, num_derivatives)

    train_seeds, valid_seeds, test_seeds = split_data([p.stringify() for p in seed_puzzles], boundaries)
    train_seeds_derivs = utils.flatten([derivative_puzzles[puzzle] for puzzle in train_seeds])
    train_puzzles, valid_deriv_puzzles, test_deriv_puzzles = split_data(train_seeds_derivs, boundaries)

    train = {puzzle: derivative_puzzle_solutions[puzzle] for puzzle in train_puzzles}
    valid_deriv = {puzzle: derivative_puzzle_solutions[puzzle] for puzzle in valid_deriv_puzzles}
    test_deriv = {puzzle: derivative_puzzle_solutions[puzzle] for puzzle in test_deriv_puzzles}

    valid_nonderiv_puzzles = utils.flatten([derivative_puzzles[puzzle] for puzzle in valid_seeds])
    valid_nonderiv = {puzzle: derivative_puzzle_solutions[puzzle] for puzzle in valid_nonderiv_puzzles}
    test_nonderiv_puzzles = utils.flatten([derivative_puzzles[puzzle] for puzzle in test_seeds])
    test_nonderiv = {puzzle: derivative_puzzle_solutions[puzzle] for puzzle in test_nonderiv_puzzles}

    # sanity check
    assert not train.keys() & valid_deriv.keys()
    assert not train.keys() & test_deriv.keys()
    assert not train.keys() & valid_nonderiv.keys()
    assert not train.keys() & test_nonderiv.keys()
    assert not valid_deriv.keys() & test_deriv.keys()
    assert not valid_deriv.keys() & valid_nonderiv.keys()
    assert not valid_deriv.keys() & test_nonderiv.keys()
    assert not test_deriv.keys() & valid_nonderiv.keys()
    assert not test_deriv.keys() & test_nonderiv.keys()
    assert not valid_nonderiv.keys() & test_nonderiv.keys()

    return train, valid_deriv, test_deriv, valid_nonderiv, test_nonderiv


def generate_XY(puzzles: dict):
    """
    :param puzzles: puzzle board string -> solution to puzzle board string
    :return:
        X: tensor of (n, max_dim^3), one for each possible candidate to each cell
        Y: tensor of (n, max_dim^2), one for each correct cell value, 0-indexed
    """
    keys = sorted(puzzles)
    X = torch.tensor([vector_encode(p) for p in keys])
    Y = torch.tensor([get_board_entries(puzzles[p]) for p in keys], dtype=torch.int64) - 1
    return X, Y


def flatten_prediction(pred_distributions):
    a0, a1, a2 = pred_distributions.shape
    return pred_distributions.reshape(a0, a1*a2)


def get_predicted_boards(pred_distributions):
    return np.argmax(pred_distributions.detach().numpy(), axis=2)


def count_correct_cells(pred_distributions, solutions):
    return torch.sum(get_predicted_boards(pred_distributions) == solutions, dim=1)


def get_board_accuracies(pred_distributions, solutions):
    return torch.tensor(count_correct_cells(pred_distributions, solutions), dtype=torch.float32) / solutions.shape[1]


def get_model_accuracy(pred_distributions, solutions):
    return np.average(get_board_accuracies(pred_distributions, solutions) == 1)


def get_performance(model, X, Y):
    """
    :param model: callable function on X: (n, max_dim^3) -> (n, max_dim^2, max_dim)
    :param X: tensor: (n, max_dim^3)
    :param Y: tensor: (n, max_dim^2)
    :return: model loss, model accuracy
    """
    prediction = model(X)
    total_cells = Y.shape[0]*Y.shape[1]
    flattened_prediction = prediction.reshape(total_cells, int(np.sqrt(Y.shape[1])))
    flattened_Y = Y.reshape(total_cells)
    loss = nn.functional.nll_loss(flattened_prediction, flattened_Y)
    return loss, get_model_accuracy(prediction, Y)


def plot_metrics(data: dict, title: str):
    """
    Plots the data points
    :param data: name of data (str) -> list of data points
        Every list must contain same length
    :return:
    """
    for label, points in data.items():
        plt.plot(range(len(points)), points, label=label)
    plt.title(title)
    plt.legend(loc='upper right')
    plt.figure(figsize=(6, 10))
    plt.show()