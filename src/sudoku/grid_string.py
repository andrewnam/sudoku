"""
GridString is a utility interface that holds a grid string rather than a numpy array or Board
It is designed to make lightweight operations such as extracting dimensionality easier
without rewriting the string operations each time

Note that a valid GridString is of the format r.c.xxxxxxxx where
    r = number of rows in a box
    c = number of columns in a box
    and there are (r * c)^2 x's and each x is a '.' or a number between 1 and (r * c)

    '.' is used over '0' because boards with more digits than 9 need to use 0 as a placeholder
"""

import numpy as np
import itertools
import re
from .board import Board


class GridString:

    def __init__(self, grid_string: str):
        dot_index = grid_string.find('.')
        dot_index2 = dot_index + 1 + grid_string[dot_index + 1:].find('.')
        assert dot_index > 0
        assert dot_index2 > 0
        self.dim_x = int(grid_string[:dot_index])
        self.dim_y = int(grid_string[dot_index+1:dot_index2])
        self.grid = grid_string[dot_index2+1:]

        self._array = None
        self._is_seed = None
        self._board = None

    @property
    def max_digit(self):
        return self.dim_x*self.dim_y


    @property
    def array(self):
        """
        :return: A numpy array of the grid
        """
        if self._array is None:
            self._array = np.array(tuple(self.traverse_grid())).reshape((self.max_digit, self.max_digit))

        return self._array

    @array.setter
    def array(self, value):
        self._array = value
        self.grid = re.sub('[^0-9,\.]', '', np.array_str(self._array).replace('0', '.'))

    @property
    def is_seed(self):
        if self._is_seed is not None:
            return self._is_seed

        top_row = itertools.islice(self.traverse_grid(), self.max_digit)
        for i, digit in zip(range(self.max_digit), top_row):
            if not (digit == i+1 or digit == 0):
                self._is_seed = False
                return False
        self._is_seed = True
        return True

    @property
    def grid_string(self):
        return '.'.join((str(self.dim_x), str(self.dim_y), self.grid))

    @property
    def board(self):
        if self._board is not None:
            return self._board
        self._board = Board(self.dim_x, self.dim_y)
        self._board.board = np.array(self.array)
        self._board.reset_pencil_marks()
        return self._board


    def traverse_grid(self):
        """
        Returns a generator that traverses through each digit in the grid
        :return:
        """
        digit_stride = len(str(self.max_digit))
        grid = self.grid
        while grid:
            if grid[0] == '.':
                yield 0
                grid = grid[1:]
            else:
                yield int(grid[:digit_stride])
                grid = grid[digit_stride:]

    def copy(self):
        clone = GridString(self.grid_string)
        clone._array = np.array(self._array)
        clone._is_seed = self._is_seed
        return clone


    def __repr__(self):
        return self.grid

    def __eq__(self, other):
        return (self.dim_x, self.dim_y, self.grid) == (other.dim_x, other.dim_y, other.grid)

    def __hash__(self):
        return (self.dim_x, self.dim_y, self.grid).__hash__()

def write_solutions_file(filename, puzzles):
    lines = []
    for k, v in puzzles.items():
        lines.append(','.join([k.grid_string, v.grid_string]))
    with open(filename, 'w') as f:
        f.write('\n'.join(lines))

def read_solutions_file(filename):
    with open(filename, encoding='utf8') as f:
        lines = f.read().splitlines()
    puzzles = {}
    unique_solutions = {}
    for line in lines:
        puzzle, solution = line.split(',')
        if solution:
            if solution not in unique_solutions:
                unique_solutions[solution] = GridString(solution)
            puzzles[GridString(puzzle)] = unique_solutions[solution]
        else:
            puzzles[GridString(puzzle)] = None

    return puzzles
