from sudoku2 import Grid
import numpy as np
import random


def count_candidates(grid: Grid):
    """
    :param grid:
    :return: dict (x, y) -> number of candidates at (x, y)
    """

    possibilities = np.sum(grid.pencil_marks, axis=2)
    xs, ys = np.nonzero(possibilities)
    return {(x, y): possibilities[x][y] for x, y in zip(xs, ys)}

def solve(grid: Grid):
    if grid.num_unfilled_cells == 0:
        return grid

    num_candidates = count_candidates(grid)
    min_num_candidates = min(num_candidates.values())
    candidate_cells = [coord for coord in num_candidates if num_candidates[coord] == min_num_candidates]
    random.shuffle(candidate_cells)
    for x, y in candidate_cells:
        for digit in grid.get_candidates(x, y):
            next_grid = grid.copy()
            next_grid.write(x, y, digit)
            if next_grid.complete:
                return next_grid
            if next_grid.is_solvable:
                solved = solve(next_grid)
                if solved:
                    return solved
