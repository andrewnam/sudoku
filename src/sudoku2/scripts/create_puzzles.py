from sudoku2 import GridSolutions
from tqdm import tqdm
import random

solutions = GridSolutions('2x3solutions.pkl')
solutions.load()

# grids = solutions.verify(3, 3, False)
solutions.save()
solutions.autosave = 5000

seeds = list(solutions.seed_puzzles)
random.shuffle(seeds)
for seed in tqdm(seeds):
    solutions.generate_new_puzzle(seed)

