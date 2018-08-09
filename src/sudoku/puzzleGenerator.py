from board import Board
from solutions import Solutions

import numpy as np
import random
random.seed(0)
np.random.seed(0)


def create_board(x_dim, y_dim):
    board = Board(x_dim, y_dim)
    for y in range(x_dim*y_dim):
        board.write(0, y, y+1)
    return write(board)


def write(board):
    """
    Randomly fills the board choosing coordinates with highest restrictions. Returns resulting filled board if one
    is found. Returns None if the board cannot be filled.
    :param board:
    :return:
    """
    possibilities = board.get_cell_possibilities_count()
    minPossibilities = [coord for coord in possibilities if possibilities[coord] == min(possibilities.values())]
    x, y = random.choice(minPossibilities)
    for digit in board.get_possible_digits(x, y):
        next_board = board.copy()
        next_board.write(x, y, digit)
        if next_board.all_filled():
            return next_board
        if next_board.is_solvable():
            next_board = write(next_board)
            if next_board:
                return next_board
    return None


def remove_cell(board: Board, solutions: dict, cells_to_remove: int=0):
    """
    Given a board with a unique solution and a dictionary of board-hash -> solution-board-hash, remove a random cell
     from the board and test if the resulting board also has a unique solution (every legal move also results boards
     that have unique solutions). Also adds the new board to unique_solutions
    If no board with a unique solution can be created, returns None
    :param board:
    :param unique_solutions:
    :param cells_to_remove:
    :return:
    """
    assert board in solutions
    assert cells_to_remove >= 0
    new_puzzles = []
    xs, ys = np.nonzero(board.board)
    indices = np.arange(len(xs))
    np.random.shuffle(indices)
    for i in indices:
        x, y = xs[i], ys[i]
        new_board = board.remove(x, y)
        if new_board in solutions: # if this puzzle already exists, skip
            pass
        if len(find_all_solutions(new_board, solutions)) == 1:
            new_puzzles.append(new_board)
            if cells_to_remove and len(new_puzzles) >= cells_to_remove:
                return new_puzzles
    return new_puzzles


def find_all_solutions(board: Board, solutions: dict) -> set:
    """
    Returns a set of Boards that contain all possible solutions to the input board
    :param board:
    :param solutions: dictionary of board K -> board V where V is the solution to K. V may be None
    :return: set of Boards
    """
    puzzles = {board: solutions[board] for board in solutions if solutions[board]}
    # if puzzles:
    #     print(len(set(list(puzzles.values()))))
    if board in solutions:
        return {solutions[board]}

    found_solutions = set()
    possibilities = board.get_cell_possibilities_count()
    min_possibilities = [coord for coord in possibilities if possibilities[coord] == min(possibilities.values())]
    x, y = random.choice(min_possibilities)
    for digit in board.get_possible_digits(x, y):
        next_board = board.copy()
        next_board.write(x, y, digit)
        if next_board in solutions:
            found_solutions.add(solutions[next_board])
        elif next_board.all_filled():
            solutions[next_board] = next_board
            found_solutions.add(solutions[next_board])
        if next_board.is_solvable():
            found_solutions |= find_all_solutions(next_board, solutions)
    if len(found_solutions) == 1:
        solutions[board] = list(found_solutions)[0]
    else:
        solutions[board] = None
    return found_solutions


if __name__ == '__main__':
    # solutions = Solutions()
    # board = Board(2, 2)
    # for i in range(4):
    #     board.write(0, i, i+1)
    #
    # find_all_solutions(board, solutions)
    # puzzles = {board: solutions[board] for board in solutions if solutions[board]}
    # print(len(puzzles))
    # print(len(set(list(puzzles.values()))))
    # print(np.min([board.count_filled_cells() for board in puzzles]))
    # solutions.save('solutions.txt')

    solutions = Solutions().load('solutions3.txt')
    puzzles = {board: solutions[board] for board in solutions if solutions[board]}

    new_puzzles = []
    for puzzle in list(puzzles.keys()):
        new_puzzles += remove_cell(puzzle, solutions, 4)
        solutions.save('solutions4.txt')
        print(len(new_puzzles))
    # new_puzzles = [item for sublist in new_puzzles for item in sublist]
    # print(np.min([board.count_filled_cells() for board in new_puzzles]))
