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

with_one = {p: s for p, s in puzzles_by_hints[6].items() if '1' in p.grid}
without_one = {p: s for p, s in puzzles_by_hints[6].items() if '1' not in p.grid}

print("Number of puzzles in with_one set: {}".format(len(with_one)))
print("Number of puzzles in without_one set: {}".format(len(without_one)))

Dataset(with_one).save('./data/with_one.pkl')
Dataset(without_one).save('./data/without_one.pkl')