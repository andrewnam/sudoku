import numpy as np
import utils
from enum import Enum
HouseType = Enum('HouseType', 'Row, Column, Box')

class House:

    def __init__(self, grid: np.ndarray,
                        type: HouseType,
                        index: int,
                        dim_x=3, dim_y=3):
        self.type = type
        self.index = index
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.max_digit = self.dim_x * self.dim_y

        if self.type == HouseType.Row:
            self.x_min = self.x_max = index
            self.y_min = 0
            self.y_max = self.max_digit-1
            self.array = grid[self.x_min]
        elif self.type == HouseType.Column:
            self.y_min = self.y_max = index
            self.x_min = 0
            self.x_max = self.max_digit-1
            self.array = grid[:,self.y_min]
        elif self.type == HouseType.Box:
            self.x_min = (self.index//self.dim_x)*self.dim_x
            self.y_min = (self.index*self.dim_y)%self.max_digit
            self.x_max = self.x_min + self.dim_x - 1
            self.y_max = self.y_min + self.dim_y - 1
            self.array = grid[self.x_min:self.x_max+1, self.y_min:self.y_max+1]

    def __getitem__(self, index):
        return self.array[index]

    def __setitem__(self, key, value):
        self.array[key] = value

    def set(self, value: np.ndarray):
        assert self.array.shape == value.shape
        self.array[:] = value

    def get_coordinates(self):
        combinations = utils.get_combinations(range(self.x_min, self.x_max + 1), range(self.y_min, self.y_max + 1))
        return {tuple(c) for c in combinations}

    def __repr__(self):
        return "{} {}\n{}".format(self.type, self.index, self.array.__repr__())