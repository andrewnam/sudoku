SUDOKU_PATH = '/home/ajhnam/sudoku'

import os
import sys
sys.path.append(SUDOKU_PATH + '/src/sudoku')
sys.path.append(SUDOKU_PATH + '/src/misc')

from dataset import Dataset, Datasets
import rrn_utils

shuffled_puzzles_filename = SUDOKU_PATH + '/data/shuffled_puzzles.txt'

if not os.path.exists('./data'):
    os.makedirs('./data')
puzzles_by_hints = rrn_utils.get_puzzles_by_hints(shuffled_puzzles_filename)

solutions = list(puzzles_by_hints[16].values())

sol_4_hint_puzzle_counts = {s: 0 for s in solutions}
for puzzle, solution in puzzles_by_hints[4].items():
    sol_4_hint_puzzle_counts[solution] += 1


# 1536 puzzles with 4 hints in total
train_set_target = 1200 + 112 # 112 for interpolated validation
valid_set_target = 112 # extrapolated validation
test_set_target = 112 # extrapolated test

train_set_sol = set()
train_set_puzzles_total = 0
valid_set_sol = set()
valid_set_puzzles_total = 0
test_set_sol = set()
test_set_puzzles_total = 0
no_puzzles = set()

for sol in sorted(sol_4_hint_puzzle_counts, key=lambda k: sol_4_hint_puzzle_counts[k])[::-1]:
    count = sol_4_hint_puzzle_counts[sol]
    if count == 0:
        no_puzzles.add(sol)
    if train_set_puzzles_total + count <= train_set_target:
        train_set_sol.add(sol)
        train_set_puzzles_total += count
    elif valid_set_puzzles_total + count <= valid_set_target:
        valid_set_sol.add(sol)
        valid_set_puzzles_total += count
    else:
        test_set_sol.add(sol)
        test_set_puzzles_total += count

print("Number of puzzles in training set: {}".format(train_set_target))
print("Number of puzzles in validation set: {}".format(valid_set_target))
print("Number of puzzles in test set: {}".format(test_set_puzzles_total))
print("Number of solutions in training set: {}".format(len(train_set_sol)))
print("Number of solutions in validation set: {}".format(len(valid_set_sol)))
print("Number of solutions in test set: {}".format(len(test_set_sol)))
print("Number of solutions in zero set: {}".format(len(no_puzzles)))

# 1312 train, 112 valid, 112 test for each
# total = 3936 train, 336 valid, 336 test
# 3600 train, 336 interpolated validation, 336 extrapolated validation, 336 test set

num_hints = [4, 5, 6]

_train_datasets = {}
_valid_datasets = {}
_test_datasets = {}

for hints in num_hints:
    train_set = {}
    valid_set = {}
    test_set = {}
    for p, s in puzzles_by_hints[hints].items():
        if s in train_set_sol:
            train_set[p] = s
        elif s in valid_set_sol:
            valid_set[p] = s
        else:
            test_set[p] = s

    print("{} hints, train set = {} puzzles".format(hints, len(train_set)))
    print("{} hints, valid set = {} puzzles".format(hints, len(valid_set)))
    print("{} hints, test set = {} puzzles".format(hints, len(test_set)))
    _train_datasets[hints] = Dataset(train_set)
    _valid_datasets[hints] = Dataset(valid_set)
    _test_datasets[hints] = Dataset(test_set)

train_datasets = Datasets(_train_datasets)
valid_datasets = Datasets(_valid_datasets)
test_datasets = Datasets(_test_datasets)

train_datasets.save('./data/train_datasets.pkl')
valid_datasets.save('./data/ext_valid_datasets.pkl')
test_datasets.save('./data/test_datasets.pkl')