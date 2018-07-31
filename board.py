from . import Cell

import numpy as np
import random

np.random.seed(0)

class Board:

	def __init__(self, dim_x, dim_y):
		self.dim_x = dim_x
		self.dim_y = dim_y
		self.max_digit = self.dim_x * self.dim_y
		self.board = np.zeros((self.dim_x**2, self.dim_y**2))
		self.pencilMarks = np.ones((self.dim_x**2, self.dim_y**2, self.max_digit))

	def copy(self):
		board = Board(self.dim_x, self.dim_y)
		board.board = np.array(self.board)
		board.pencilMarks = np.array(self.pencilMarks)
		return board

	def isSolvable(self):
		"""
		Checks if the board has any unfilled cells that have no candidates
		:return: boolean
		"""
		# A cell is empty and it has no candidate pencil marks
		return np.any((self.board == 0) & (np.sum(self.pencilMarks, axis=2) == 0))

	def countFilledCells(self):
		return np.sum(self.board != 0)

	def countUnfilledCells(self):
		return np.sum(self.board == 0)

	def getCellPossibilitiesCount(self):
		possibilities = np.sum(self.pencilMarks, axis=2)
		xs, ys = np.nonzero(possibilities)
		return {(x, y): possibilities[x][y] for x, y in zip(xs, ys)}

	def getPossibleDigits(self, x, y):
		return self.pencilMarks[x][y]

	def write(self, x, y, digit):
		assert digit in self.board[x][y]
		self.board[x][y] = digit
		self.pencilMarks[x, y, :] = 0
		self.pencilMarks[x, :, digit - 1] = 0
		self.pencilMarks[:, y, digit - 1] = 0

		box_x_min = self.dim_x * (x // self.dim_x)
		box_x_max = box_x_min + self.dim_x
		box_y_min = self.dim_y * (y // self.dim_y)
		box_y_max = box_y_min + self.dim_y

		self.pencilMarks[box_x_min:box_x_max, box_y_min:box_y_max, digit-1] = 0

	def findDigitInColumn(self, x, digit):
		y = np.where(self.board[x] == digit)[0]
		return Cell(x, y[0], digit) if len(y) > 0 else -1

	def findDigitInRow(self, y, digit):
		x = np.where(self.board[:, y] == digit)[0]
		return Cell(x[0], y, digit) if len(x) > 0 else -1

	def findDigitInBox(self, x, y, digit):
		box_x_min = self.dim_x * (x // self.dim_x)
		box_x_max = box_x_min + self.dim_x
		box_y_min = self.dim_y * (y // self.dim_y)
		box_y_max = box_y_min + self.dim_y
		offset_x, offset_y = np.where(self.board[box_x_min:box_x_max, box_y_min:box_y_max] == digit)
		return Cell(box_x_min + offset_x[0], box_y_min + offset_y[0], digit) if len(offset_x) > 0 else -1

	def findContradictions(self, x, y, digit):
		"""
		Given a board, x, y and a digit that is not legal at x, y,
		returns a list of Marks in its row, column, and/or family
		that contributes to its contradiction
		"""
		return [self.findDigitInColumn(self, x, digit),
				self.findDigitInRow(self, y, digit),
				self.findDigitInBox(self, x, y, digit)]

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

	def createPencilMarks(self):
		"""
		Evaluates the current board state and returns appropriate pencilMarks
		Only useful when a board is newly loaded
		:return:
		"""
		pencilMarks = np.ones((self.dim_x, self.dim_y, self.max_digit))
		xs, ys = np.nonzero(self.board)
		for x, y in zip(xs, ys):
			digit = board[x][y]
			board, pencilMarks = self.write(board, pencilMarks, x, y, digit)
		return pencilMarks