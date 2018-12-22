import numpy as np
import itertools

class Grid:

    def __init__(self, dim_x: int, dim_y: int):
        self.dim_x = dim_x
        self.dim_y = dim_y

        self._array = None
        self._is_seed = None

        self._grid_string = None
        self._grid_array = None
        self.dirty_grid_string = True
        self.dirty_grid_array = True

    @property
    def grid_string(self):
        return

    @grid_string.setter
    def grid_string(self, value):


    @staticmethod
    def load_from_string(s: str):
        """
        :param s: is form "dim_x.dim_y.###############" where missing cells are represented as '.'
            if dim_x*dim_y > 9, cells are all represented with left-zero-padded numbers to make them uniform
        :return:
        """
        dot_index = s.find('.')
        dot_index2 = dot_index + 1 + s[dot_index + 1:].find('.')
        assert dot_index > 0
        assert dot_index2 > 0
        dim_x = int(s[:dot_index])
        dim_y = int(s[dot_index + 1:dot_index2])
        grid = Grid(dim_x, dim_y)
        grid._grid_string = s[dot_index2 + 1:]
        return grid