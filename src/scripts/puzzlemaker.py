import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import argparse
from datetime import datetime

from sudoku import Board
from sudoku import Solutions
import andrew_utils as utils


def generate_new_puzzle(board: Board, solutions: Solutions):
    new_puzzles = [board]
    # new_puzzles = remove_cell(board, solutions, cells_to_remove=1)
    # new_puzzles = {p for p in new_puzzles if solutions[p]}
    while new_puzzles:
        new_board = random.sample(new_puzzles, 1)[0]
        new_puzzles = remove_cell(new_board, solutions, cells_to_remove=1)
        new_puzzles = {p for p in new_puzzles if solutions[p]}
        for p in new_puzzles:
            print(p.stringify())

    # currently does nothing to check if the final resulting puzzle already exists
    # This is a hard problem since 'parent' boards should be allowed to already exist in solutions.
    return new_board if new_board != board else None


def remove_cell(board: Board, solutions: Solutions, cells_to_remove: int=0) -> list:
    """
    Given a board with a unique solution and a dictionary of board-hash -> solution-board-hash, remove a random cell
     from the board and test if the resulting board also has a unique solution (every legal move also results boards
     that have unique solutions). Also adds the new board to unique_solutions
    If no board with a unique solution can be created, returns None
    :param board:
    :param solutions:
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
        # remove is a relatively expensive step (Still O(1)). Can short circuited for checking if the board
        # already exists in solutions, but b/c of the way hashing works, not quite so simple to
        # implement. Might be worth doing in the future.
        if (new_board in solutions and solutions[new_board]) or len(solutions.find_all_solutions(new_board)) == 1:
            new_puzzles.append(new_board)
            if cells_to_remove and len(new_puzzles) >= cells_to_remove:
                return new_puzzles
    return new_puzzles if new_puzzles else None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--solution_file", help="Path to the solutions save file. If none provided, automatically"
                                                    + "outputs to ../../data/solutions_{timestamp}.txt.")
    parser.add_argument("-x", "--dim_x", help="Include if solution_file was not given. Used to create new boards.", type=int)
    parser.add_argument("-y", "--dim_y", help="Include if solution_file was not given. Used to create new boards.", type=int)
    parser.add_argument("-p", "--puzzles", help="The number of new minimum hint puzzles to create.", type=int)
    parser.add_argument("-v", "--verbose", help="increase output verbosity",
                        action="store_true")
    args = parser.parse_args()

    assert args.solution_file or (args.dim_x and args.dim_y)

    solution_file = args.solution_file
    verbose = args.verbose

    if solution_file:
        solutions = Solutions(solution_file)
    else:
        solution_file = '../../data/solutions_{0}.txt'.format(utils.datetime_to_str(datetime.utcnow()))
        solutions = Solutions(solution_file)
        board = Board(args.dim_x, args.dim_y)
        for i in range(args.dim_x * args.dim_y):
            board.write(0, i, i + 1)
        print("Finding solutions")
        solutions.find_all_solutions(board)

    print("Beginning sequence")
    for i in tqdm(range(args.puzzles)):
        sol = solutions.get_min_puzzle_seed_solution()
        generate_new_puzzle(sol, solutions)
    solutions.save()