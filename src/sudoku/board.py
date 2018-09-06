from cell import Cell

import numpy as np
import random
import joblib
import re
import utils

np.random.seed(0)

class Box:

    def __init__(self, board, x_min, y_min):
        self.board = board
        self.x_min = x_min
        self.y_min = y_min

        self.box = self.board[x_min:x_min + self.board.dim_x, y_min:y_min + self.board.dim_y]

    def get_coordinates(self):
        return utils.get_combinations(range(self.x_min, self.x_min + self.board.dim_x),
                               range(self.y_min, self.y_min + self.board.dim_y))


class Board:

    def __init__(self, dim_x, dim_y):
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.max_digit = self.dim_x * self.dim_y
        self.board = np.zeros((self.max_digit, self.max_digit), dtype=np.int8)
        self.pencilMarks = np.ones((self.max_digit, self.max_digit, self.max_digit), dtype=np.int8)
        self.boxes = {} # maps (box_x_min, box_y_min) to Box

    def __repr__(self):
        return self.board.__repr__()

    def __hash__(self):
        return joblib.hash(self.board).__hash__()

    def __eq__(self, other):
        return np.all(self.board == other.board)

    def __getitem__(self, item):
        return self.board.__getitem__(item)

    def __lt__(self, other):
        assert type(other) == Board
        return self.stringify() < other.stringify()

    @property
    def T(self):
        transpose = Board(self.dim_x, self.dim_y)
        transpose.board = np.transpose(self.board)
        transpose.pencilMarks = np.transpose(self.pencilMarks, (1, 0, 2))
        return transpose

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

    def is_candidate(self, x, y, digit):
        return self.pencilMarks[x][y][digit-1] > 0

    def get_box_boundaries_x(self, x):
        """
        :param x:
        :return: X Boundaries of the box that x is in, [inclusive, exclusive)
        """
        x_min = (x // self.dim_x) * self.dim_x
        return x_min, x_min + self.dim_x

    def get_box_boundaries_y(self, y):
        """
        :param y:
        :return: Y Boundaries of the box that y is in, [inclusive, exclusive)
        """
        y_min = (y // self.dim_y) * self.dim_y
        return y_min, y_min + self.dim_y

    def get_box(self, x, y):
        box_x_min = self.get_box_boundaries_x(x)[0]
        box_y_min = self.get_box_boundaries_y(y)[0]
        key = (box_x_min, box_y_min)
        if key not in self.boxes:
            self.boxes[key] = Box(self, box_x_min, box_y_min)
        return self.boxes[key]

    # def get_box_index_by_coords(self, x, y):
    #     return (x // self.dim_x) * self.dim_x + y // self.dim_y
	#
    # def get_box_by_index(self, index, copy=True):
    #     x = (index // self.dim_x) * self.dim_x
    #     y = (index % self.dim_y) * self.dim_y
    #     arr = self[x:x + self.dim_x, y:y + self.dim_y]
    #     return np.array(arr) if copy else arr
	#
    # def get_box_by_coords(self, x, y, copy=True):
    #     return self.get_box_by_index(self.get_box_by_coords(x, y), copy)
	#
    # def get_boxes(self):
    #     return np.array([self.board[x * self.dim_x:(x + 1) * self.dim_x,
    #                      y * self.dim_y:(y + 1) * self.dim_y]
    #                      for x, y
    #                      in utils.get_combinations(np.arange(self.dim_x), np.arange(self.dim_y))])

    def get_cell_possibilities_count(self):
        possibilities = np.sum(self.pencilMarks, axis=2)
        xs, ys = np.nonzero(possibilities)
        return {(x, y): possibilities[x][y] for x, y in zip(xs, ys)}

    def get_possible_digits(self, x, y):
        return np.nonzero(self.pencilMarks[x][y])[0] + 1

    def get_digit_pencilmarks(self, digit):
        return self.pencilMarks[:, :, digit-1]

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
        contradictions = [self.find_digit_in_column(x, digit),
                          self.find_digit_in_row(y, digit),
                          self.find_digit_in_box(x, y, digit)]
        return [c for c in contradictions if c]

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

        box_coords = self.get_box(x, y).get_coordinates()
        for i in range(board.max_digit):
            # for cell at (x, y)
            if not board.find_contradictions(x, y, i+1):
                board.pencilMarks[x][y][i] = 1
            # for row x
            if i != x and board[i][y] == 0 and not board.find_contradictions(i, y, digit):
                board.pencilMarks[i][y][digit - 1] = 1
            # for column y
            if i != y and board[x][i] == 0and not board.find_contradictions(x, i, digit):
                board.pencilMarks[x][i][digit - 1] = 1
            # for box
            box_x, box_y = box_coords[i]
            if box_x != x and box_y != y and board[box_x][box_y] == 0 and not board.find_contradictions(box_x, box_y, digit):
                board.pencilMarks[box_x][box_y][digit - 1] = 1

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