SUDOKU_PATH = '/home/ajhnam/sudoku'

import os
import sys
sys.path.append(SUDOKU_PATH + '/src/sudoku')
sys.path.append(SUDOKU_PATH + '/src/misc')

from dataset import Dataset
import rrn_utils

shuffled_puzzles_filename = SUDOKU_PATH + '/data/shuffled_puzzles.txt'

if not os.path.exists('./data'):
    os.makedirs('./data')
puzzles_by_hints = rrn_utils.get_puzzles_by_hints(shuffled_puzzles_filename)

with_top_left = {p: s for p, s in puzzles_by_hints[6].items() if p.grid[0] != '.'}
without_top_left = {p: s for p, s in puzzles_by_hints[6].items() if p.grid[0] == '.'}

print("Number of puzzles in with_top_left set: {}".format(len(with_top_left)))
print("Number of puzzles in without_top_left set: {}".format(len(without_top_left)))

Dataset(with_top_left).save('./data/with_top_left.pkl')
Dataset(without_top_left).save('./data/without_top_left.pkl')