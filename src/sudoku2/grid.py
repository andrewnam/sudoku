import numpy as np
from .house import House, HouseType
from .exceptions import InvalidWriteException
import joblib
import re


class Grid:

    def __init__(self, dim_x: int, dim_y: int):
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.max_digit = self.dim_x * self.dim_y
        self.array = np.zeros((self.max_digit, self.max_digit), dtype=int)
        self.pencil_marks = np.ones((self.max_digit, self.max_digit, self.max_digit))
        self.rows = tuple([House(self.array, self.pencil_marks, HouseType.Row, i, self.dim_x, self.dim_y)
                        for i in range(self.max_digit)])
        self.columns = tuple([House(self.array, self.pencil_marks, HouseType.Column, i, self.dim_x, self.dim_y)
                        for i in range(self.max_digit)])
        self.boxes = tuple([House(self.array, self.pencil_marks, HouseType.Box, i, self.dim_x, self.dim_y)
                        for i in range(self.max_digit)])

    @property
    def num_filled_cells(self):
        return np.sum(self.array != 0)

    @property
    def num_unfilled_cells(self):
        return np.sum(self.array == 0)

    @property
    def complete(self):
        return self.num_unfilled_cells == 0

    @property
    def is_solvable(self):
        """
        Checks if the board has any unfilled cells that have no candidates
        """
        return not np.any((self.array == 0) & (np.sum(self.pencil_marks, axis=2) == 0))

    def to_grid_string(self):
        s = re.sub('[^0-9]', '', np.array_str(self.array))
        return GridString(self.dim_x, self.dim_y, s)

    def copy(self):
        grid = Grid(self.dim_x, self.dim_y)
        np.copyto(grid.array, self.array)
        np.copyto(grid.pencil_marks, self.pencil_marks)
        return grid

    def is_candidate(self, x, y, digit):
        return self.pencil_marks[x][y][digit-1]

    def set_pencil_marks(self, x, y):
        digit = self.array[x][y]
        if digit:
            self.pencil_marks[x][y] = np.zeros(self.max_digit)
            self.rows[x].erase_pencil_marks(digit)
            self.columns[y].erase_pencil_marks(digit)
            self.box_containing(x, y).erase_pencil_marks(digit)

    def write(self, x, y, digit):
        if self.is_candidate(x, y, digit):
            self.array[x][y] = digit
            self.set_pencil_marks(x, y)
        else:
            raise InvalidWriteException(x, y, digit, self.pencil_marks[x][y])

    def contradiction_exists(self, x, y, digit):
        assert self.array[x][y] != digit
        return (digit in self.rows[x]) or (digit in self.columns[y]) or (digit in self.box_containing(x, y))

    def remove(self, x, y, in_place=True):
        grid = self if in_place else self.copy()
        digit = grid.array[x][y]
        assert digit > 0
        grid.array[x][y] = 0

        # for cell at (x, y)
        f = lambda d: not self.contradiction_exists(x, y, d)
        grid.pencil_marks[x][y] = np.vectorize(f)(np.arange(self.max_digit) + 1)

        # for all other cells in neighborhood
        neighbors = self.rows[x].get_coordinates()
        neighbors |= self.columns[y].get_coordinates()
        neighbors |= self.box_containing(x,y).get_coordinates()
        neighbors.remove((x, y))

        for x, y in neighbors:
            grid.pencil_marks[x][y][digit-1] = not self.contradiction_exists(x, y, digit)

        return None if in_place else grid


    def box_containing(self, x, y):
        return self.boxes[(x // self.dim_x) * self.dim_x + (y // self.dim_y)]

    def get_candidates(self, x, y):
        return np.nonzero(self.pencil_marks[x][y])[0] + 1

    def __getitem__(self, index):
        return np.array(self.array[index])

    def __eq__(self, other):
        return np.all(self.array == other.array)

    def __lt__(self, other):
        return self.__repr__() < other.__repr__()

    def __repr__(self):
        return self.array.__repr__()

    def __hash__(self):
        return joblib.hash(self.array).__hash__()


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

    def to_grid(self):
        grid = Grid(self.dim_x, self.dim_y)
        digits =(self.traverse_grid)

        max_digit = self.dim_y * self.dim_y
        i = 0
        for x in range(max_digit):
            for y in range(max_digit):
                grid.array[x][y] = digits[i]
                i += 1
        return grid

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

    def __eq__(self, other):
        return self.dim_x == other.dim_x and self.dim_y == other.dim_y and self.grid_string == other.grid_string

    def __lt__(self, other):
        return self.grid_string < other.grid_string

    def __hash__(self):
        return str(self).__hash__()

    def __repr__(self):
        return f"{self.dim_x}_{self.dim_y}_{self.grid_string}"