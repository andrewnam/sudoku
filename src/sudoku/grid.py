import numpy as np
import itertools

class Grid:

    def __init__(self, dim_x: int, dim_y: int):
        self.dim_x = dim_x
        self.dim_y = dim_y

        self._array = None
        self._is_seed = None
        self._grid_string = None

    @staticmethod
    def load_from_string(grid_string: str):
        dot_index = grid_string.find('.')
        dot_index2 = dot_index + 1 + grid_string[dot_index + 1:].find('.')
        assert dot_index > 0
        assert dot_index2 > 0
        dim_x = int(grid_string[:dot_index])
        dim_y = int(grid_string[dot_index + 1:dot_index2])
        grid = grid_string[dot_index2 + 1:]