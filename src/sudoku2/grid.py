import numpy as np
from .house import House, HouseType

class Grid:

    def __init__(self, dim_x: int, dim_y: int):
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.max_digit = self.dim_x * self.dim_y
        self.array = np.zeros((self.max_digit, self.max_digit))
        self.rows = tuple([House(self.array, HouseType.Row, i, self.dim_x, self.dim_y)
                        for i in range(self.max_digit)])
        self.columns = tuple([House(self.array, HouseType.Column, i, self.dim_x, self.dim_y)
                        for i in range(self.max_digit)])
        self.boxes = tuple([House(self.array, HouseType.Box, i, self.dim_x, self.dim_y)
                        for i in range(self.max_digit)])

    def __getitem__(self, index):
        return self.array[index]

    def __setitem__(self, key, value):
        self.array[key] = value

    def __repr__(self):
        return self.array.__repr__()


class GridString:

    def __init__(self, dim_x: int, dim_y: int, grid_string: str):
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.grid_string = grid_string

    @property
    def max_digit(self):
        return self.dim_x * self.dim_y

    @property
    def num_hints(self):
        return len(self.grid_string) - self.grid_string.count('.')

    def traverse_grid(self):
        """
        Returns a generator that traverses through each digit in the grid
        :return:
        """
        digit_stride = len(str(self.max_digit))
        grid = self.grid_string
        while grid:
            if grid[0] == '.':
                yield 0
                grid = grid[1:]
            else:
                yield int(grid[:digit_stride])
                grid = grid[digit_stride:]


    @staticmethod
    def load(s: str):
        a = s.split('_')
        dim_x = int(a[0])
        dim_y = int(a[1])
        grid_string = a[2]
        return GridString(dim_x, dim_y, grid_string)
        

    @staticmethod
    def load_old_format(s: str):
        dot_index = s.find('.')
        dot_index2 = dot_index + 1 + s[dot_index + 1:].find('.')
        assert dot_index > 0
        assert dot_index2 > 0

        dim_x = int(s[:dot_index])
        dim_y = int(s[dot_index+1:dot_index2])
        grid_string = s[dot_index2+1:]
        return GridString(dim_x, dim_y, grid_string)

    def __repr__(self):
        return f"{self.dim_x}_{self.dim_y}_{self.grid_string}"
