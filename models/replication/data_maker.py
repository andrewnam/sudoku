SUDOKU_PATH = '/home/ajhnam/sudoku'

import os
import sys
sys.path.append(SUDOKU_PATH + '/src/sudoku')

from grid_string import read_solutions_file
from dataset import Dataset

if not os.path.exists('./data'):
    os.makedirs('./data')

train_set = Dataset(read_solutions_file(SUDOKU_PATH + "/data/imported/rrn/train.txt"))
valid_set = Dataset(read_solutions_file(SUDOKU_PATH + "/data/imported/rrn/valid.txt"))
test_set = Dataset(read_solutions_file(SUDOKU_PATH + "/data/imported/rrn/test.txt"))

train_set.save("./data/train_set.pkl")
valid_set.save("./data/valid_set.pkl")
test_set.save("./data/test_set.pkl")