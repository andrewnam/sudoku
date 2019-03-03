from grid import Grid, HouseType
import numpy as np
from shuffler import Shuffler

def full_house(x, y, house_type, dim_x=3, dim_y=3):
    grid = Grid(dim_x, dim_y)
    if house_type is HouseType.Row:
        grid[x] = np.roll(np.arange(1, grid.max_digit+1), y)
    elif house_type is HouseType.Column:
        grid[:,y] = np.roll(np.arange(1, grid.max_digit+1), x)
    elif house_type is HouseType.Box:
        pass
    grid[x][y] = 0
    return grid


print(full_house(3, 5, HouseType.Row))