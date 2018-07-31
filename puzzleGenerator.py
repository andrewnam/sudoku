from .board import Board

import random

def createBoard(x_dim, y_dim):
	board = Board(x_dim, y_dim)
	for y in range(y_dim):
		board.write(0, y, y)
	return write(board)

def write(board):
	possibilities = board.getCellPossibilitiesCount()
	minPossibilities = [coord for coord in possibilities if possibilities[coord] == min(possibilities.values())]
	x, y = random.choice(minPossibilities)
	for digit in board.getPossibleDigits(x, y):
		next_board = board.copy()
		next_board.write(x, y, digit)
		if board.isSolvable():
			next_board = write(next_board)
			if next_board:
				return next_board
	return None