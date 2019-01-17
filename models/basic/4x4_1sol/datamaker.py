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

with_sol_one = {p: s for p, s in puzzles_by_hints[6].items() if s.grid[0] == '1'}
without_sol_one = {p: s for p, s in puzzles_by_hints[6].items() if s.grid[0] != '1'}

print("Number of puzzles in with_sol_one set: {}".format(len(with_sol_one)))
print("Number of puzzles in without_sol_one set: {}".format(len(without_sol_one)))

Dataset(with_sol_one).save('./data/with_sol_one.pkl')
Dataset(without_sol_one).save('./data/without_sol_one.pkl')