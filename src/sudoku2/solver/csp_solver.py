from sudoku2 import Grid
import random

def solve(grid: Grid):
    if grid.complete:
        assert grid.solved
        return grid

    num_candidates = grid.count_candidates()
    min_num_candidates = min(num_candidates.values())
    candidate_cells = [coord for coord in num_candidates if num_candidates[coord] == min_num_candidates]
    x, y = random.choice(candidate_cells)
    for digit in grid.get_candidates(x, y):
        next_grid = grid.copy()
        next_grid.write(x, y, digit)
        if next_grid.complete:
            assert next_grid.solved
            return next_grid
        if next_grid.is_solvable:
            solved = solve(next_grid)
            if solved:
                assert solved.solved
                return solved
