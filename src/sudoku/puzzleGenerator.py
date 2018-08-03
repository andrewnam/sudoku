from board import Board


import random


def createBoard(x_dim, y_dim):
    board = Board(x_dim, y_dim)
    for y in range(x_dim*y_dim):
        board.write(0, y, y+1)
    return write(board)


def write(board):
    possibilities = board.get_cell_possibilities_count()
    minPossibilities = [coord for coord in possibilities if possibilities[coord] == min(possibilities.values())]
    x, y = random.choice(minPossibilities)
    for digit in board.get_possible_digits(x, y):
        next_board = board.copy()
        next_board.write(x, y, digit)
        if next_board.all_filled():
            return next_board
        if next_board.is_solvable():
            next_board = write(next_board)
            if next_board:
                return next_board
    return None


if __name__ == '__main__':
    print(createBoard(5, 5))