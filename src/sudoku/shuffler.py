"""
Shuffler is a factory class for ShuffledGrid, making it easy to create many transformations of a single
seed grid across all permutations.
"""


import itertools
from shuffled_grid import ShuffledGrid
from grid_string import GridString
from grid import Grid


class Shuffler:
    def __init__(self, grid: Grid):
        self.grid = grid

    def permute_labels(self):
        max_digit = self.grid.max_digit
        perm_tuples = itertools.permutations(range(1, max_digit + 1), max_digit)
        permutations = [{i + 1: tupl[i] for i in range(len(tupl))} for tupl in perm_tuples]
        shuffled_grids = [ShuffledGrid(self.grid_string, perm) for perm in permutations]
        return shuffled_grids

class Shuffler2:
    def __init__(self, grid_string: GridString):
        self.grid_string = grid_string

    def permute_labels(self):
        max_digit = self.grid_string.max_digit
        perm_tuples = itertools.permutations(range(1, max_digit + 1), max_digit)
        permutations = [{i + 1: tupl[i] for i in range(len(tupl))} for tupl in perm_tuples]
        shuffled_grids = [ShuffledGrid(self.grid_string, perm) for perm in permutations]
        return shuffled_grids
