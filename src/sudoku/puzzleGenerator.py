from board import Board
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

"""
Puzzle generator:
Start with a completed board.
Given a valid board (solvable to a unique solution), remove a random cell.
    Attempt to solve the board by trying every remaining legal digits in that cell. If any lead to a solution, backtrack
    but also save the new resultant puzzle. This way, starting with one board can result in many similar games and
    puzzles.
"""
def remove_cell(board: Board, solutions: dict, num_new_puzzles: int):
    """
    Given a board with a unique solution and a dictionary of board-hash -> solution-board-hash, remove a random cell
     from the board and test if the resulting board also has a unique solution (every legal move also results boards
     that have unique solutions). Also adds the new board to unique_solutions
    If no board with a unique solution can be created, returns None
    :param board:
    :param unique_solutions:
    :return:
    """
    assert board in solutions
    new_puzzles_generated = 0
    xs, ys = np.nonzero(board.board)
    indices = np.arange(len(xs))
    np.random.shuffle(indices)
    for i in indices:
        x, y = xs[i], ys[i]
        new_board = board.remove(x, y)
    return None

def find_all_solutions(board: Board, solutions: dict) -> set:
    puzzles = {board: solutions[board] for board in solutions if solutions[board]}
    if puzzles:
        print(len(set(list(puzzles.values()))))
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
    solutions = {}
    # board = Board(2, 2)
    # board.write(0, 0, 1)
    # board.write(0, 1, 2)
    # board.write(0, 2, 3)

    board = Board(3, 3)
    for i in range(9):
        board.write(0, i, i+1)
    # board.write(0, 1, 2)
    # board.write(0, 2, 3)

    find_all_solutions(board, solutions)
    puzzles = {board: solutions[board] for board in solutions if solutions[board]}
    print(len(puzzles))
    print(len(set(list(puzzles.values()))))
    print(np.min([board.count_filled_cells() for board in puzzles]))