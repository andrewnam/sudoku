from sudoku2 import Grid, GridString
from sudoku2.solver.csp_solver import solve
from tqdm import tqdm

dim_x = 2
dim_y = 3
num_solutions = 1
filename = f"{dim_x}x{dim_y}solutions.txt"

max_digit = dim_x * dim_y
grid = Grid(dim_x, dim_y)
for i in range(max_digit):
    grid.write(0, i, i+1)

solutions = set()
for i in tqdm(range(num_solutions)):
    solutions.add(solve(grid))

print(len(solutions)) # may want to check

with open(filename, 'w') as f:
    f.write('\n'.join([str(g.to_grid_string()) for g in solutions]))
