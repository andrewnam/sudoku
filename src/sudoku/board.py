from cell import Cell

import numpy as np
import random
import joblib
import re
import utils

np.random.seed(0)


class Board:

    def __init__(self, dim_x, dim_y):
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.max_digit = self.dim_x * self.dim_y
        self.board = np.zeros((self.max_digit, self.max_digit), dtype=np.int8)
        self.pencilMarks = np.ones((self.max_digit, self.max_digit, self.max_digit), dtype=np.int8)

    def __repr__(self):
        return self.board.__repr__()

    def __hash__(self):
        return joblib.hash(self.board).__hash__()

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

    def __getitem__(self, item):
        return self.board.__getitem__(item)

    def copy(self):
        board = Board(self.dim_x, self.dim_y)
        board.board = np.array(self.board)
        board.pencilMarks = np.array(self.pencilMarks)
        return board

    def stringify(self):
        return "{0}.{1}.{2}".format(self.dim_x, self.dim_y,
                                    re.sub('[^0-9]', '', np.array_str(self.board)))

    @staticmethod
    def loadFromString(s: str):
        dim_x, dim_y, digits = s.split('.')
        board = Board(int(dim_x), int(dim_y))
        board.board = (np.fromstring(digits, dtype=np.int8, sep='') - 48)\
            .reshape((board.max_digit, board.max_digit))
        board.resetPencilMarks()
        return board


    def all_filled(self):
        return np.sum(self.board == 0) == 0

    def is_solvable(self):
        """
        Checks if the board has any unfilled cells that have no candidates
        :return: boolean
        """
        return not np.any((self.board == 0) & (np.sum(self.pencilMarks, axis=2) == 0))

    def count_filled_cells(self):
        return np.sum(self.board != 0)

    def count_unfilled_cells(self):
        return np.sum(self.board == 0)

    def is_seed(self) -> bool:
        return (self.board[0] == np.arange(self.board.shape[1]) + 1).all()

    def get_boxes(self):
        return np.array([self.board[x * self.dim_x:(x + 1) * self.dim_x,
                         y * self.dim_y:(y + 1) * self.dim_y]
                         for x, y
                         in utils.get_combinations(np.arange(self.dim_x), np.arange(self.dim_y))])

    def get_cell_possibilities_count(self):
        possibilities = np.sum(self.pencilMarks, axis=2)
        xs, ys = np.nonzero(possibilities)
        return {(x, y): possibilities[x][y] for x, y in zip(xs, ys)}

    def get_possible_digits(self, x, y):
        return np.nonzero(self.pencilMarks[x][y])[0] + 1

    def setPencilMarks(self, x, y):
        digit = self.board[x][y]
        self.pencilMarks[x, y, :] = 0
        self.pencilMarks[x, :, digit - 1] = 0
        self.pencilMarks[:, y, digit - 1] = 0

        box_x_min = self.dim_x * (x // self.dim_x)
        box_x_max = box_x_min + self.dim_x
        box_y_min = self.dim_y * (y // self.dim_y)
        box_y_max = box_y_min + self.dim_y

        self.pencilMarks[box_x_min:box_x_max, box_y_min:box_y_max, digit - 1] = 0

    def write(self, x, y, digit):
        assert bool(self.pencilMarks[x][y][digit-1])
        self.board[x][y] = digit
        self.setPencilMarks(x, y)

    def write_in_box(self, box_number, x, y, digit):
        x = (box_number // self.dim_x) * self.dim_x + x
        y = (box_number % self.dim_y) * self.dim_y + y
        self.write(x, y, digit)

    def resetPencilMarks(self):
        """
        Evaluates the current board state and returns appropriate pencilMarks
        Only useful when a board is newly loaded
        :return:
        """
        self.pencilMarks = np.ones((self.max_digit, self.max_digit, self.max_digit))
        xs, ys = np.nonzero(self.board)
        for x, y in zip(xs, ys):
            self.setPencilMarks(x, y)

    def find_digit_in_column(self, x, digit):
        y = np.where(self.board[x] == digit)[0]
        return Cell(x, y[0], digit) if len(y) > 0 else None

    def find_digit_in_row(self, y, digit):
        x = np.where(self.board[:, y] == digit)[0]
        return Cell(x[0], y, digit) if len(x) > 0 else None

    def find_digit_in_box(self, x, y, digit):
        box_x_min = self.dim_x * (x // self.dim_x)
        box_x_max = box_x_min + self.dim_x
        box_y_min = self.dim_y * (y // self.dim_y)
        box_y_max = box_y_min + self.dim_y
        offset_x, offset_y = np.where(self.board[box_x_min:box_x_max, box_y_min:box_y_max] == digit)
        return Cell(box_x_min + offset_x[0], box_y_min + offset_y[0], digit) if len(offset_x) > 0 else None

    def find_contradictions(self, x, y, digit):
        """
        Given a board, x, y and a digit that is not legal at x, y,
        returns a list of Marks in its row, column, and/or family
        that contributes to its contradiction
        """
        return [self.find_digit_in_column(x, digit),
                self.find_digit_in_row(y, digit),
                self.find_digit_in_box(x, y, digit)]

    def remove(self, x, y, in_place=False):
        """
        Removes the existing digit from the board at (x, y). Fills pencil-marks based on
        :param x:
        :param y:
        :param in_place:
        :return: digit removed from cell
        """
        board = self if in_place else self.copy()
        digit = board.board[x][y]
        assert digit > 0
        board.board[x][y] = 0
        for i in range(board.max_digit):
            if not [1 for cell in board.find_contradictions(x, y, i+1) if cell]: # no contradictions found
                board.pencilMarks[x][y][i] = 1
        return None if in_place else board

    def seekNS(self, pencilMarks):
        xs, ys = np.where(self.board == 0)
        coordinates = list(zip(xs, ys))
        random.shuffle(coordinates)
        for x, y in coordinates:
            possible_digits = np.nonzero(pencilMarks[x][y])[0]
            if len(possible_digits) == 1:
                digit = possible_digits[0] + 1
                return Cell(x, y, digit)

    def seekAllNS(board, pencilMarks):
        marks = []
        xs, ys = np.where(board == 0)
        for x, y in zip(xs, ys):
            possible_digits = np.nonzero(pencilMarks[x][y])[0]
            if len(possible_digits) == 1:
                digit = possible_digits[0] + 1
                marks.append(Cell(x, y, digit))
        return marks