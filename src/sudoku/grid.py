from enum import Enum
import re
import numpy as np
import andrew_utils as utils

HouseType = Enum('HouseType', 'Row, Column, Box')

class House:

    def __init__(self, grid: Grid, type: HouseType, index: int):
        self.grid = grid
        self.type = type
        self.index = index
        self._array = np.zeros(grid.max_digit)


    def get_coordinates(self):
        return utils.get_combinations(range(self.x_min, self.x_min + self.grid.dim_x),
                               range(self.y_min, self.y_min + self.grid.dim_y))

class Grid:

    def __init__(self, dim_x: int, dim_y: int):
        self.dim_x = dim_x
        self.dim_y = dim_y
        self._array = np.zeros((self.max_digit, self.max_digit))
        self._gridstring = '.' * self.max_digit


    @property
    def max_digit(self):
        return self.dim_x * self.dim_y

    @property
    def gridstring(self):
        if not self._gridstring:
            self._gridstring = re.sub('[^0-9]', '', np.array_str(self.board)).replace('0', '.')
        return '.'.join((str(self.dim_x), str(self.dim_y), self._gridstring))

    def traverse_gridstring(self):
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

    def get_string_index(self, x, y):
        return x*self.max_digit + y

    @property
    def array(self):
        if self._array is None:
            self._array = np.array(tuple(self.traverse_gridstring()))
            self._array = self._array.reshape((self.max_digit, self.max_digit))
        return self._array

    def write(self, x: int, y: int, digit: int):
        self._array[x][y] = digit
        self._gridstring[self.get_string_index(x, y)] = str(digit)

    def write_chunk(self, x_min: int, x_max: int, y_min: int, y_max: int, digits):
        self._array[x_min:x_max, y_min:y_max] = digits
        self._gridstring = None

    def write_row(self, x: int, digits):
        self._array[x] = digits
        string_index = self.get_string_index(x, 0)
        self._gridstring[string_index:string_index+self.max_digit] = digits

    def copy(self):
        grid = Grid(self.dim_x, self.dim_y)
        grid._array = self._array
        grid._gridstring = self._gridstring
        return grid

    def __getitem__(self, item):
        return np.array(self.array[item])

    def __repr__(self):
        return self.array.__repr__()