import pickle
import random
import numpy as np
from .grid import Grid, GridString
from andrew_utils import Dict
from .exceptions import SolutionsVerificationException
from sudoku2.solver.csp_solver import solve as csp_solve
import andrew_utils as utils
from tqdm import tqdm

import logging

utils.setup_logging()
logger = logging.getLogger(__name__)


class GridSolutions:

    def __init__(self, filename=None):
        self.puzzles = {}
        self.seed_puzzles = Dict(list) # seed_solution -> puzzle
        self.filename = filename
        self.autosave = 1000 # auto-save every n entries

    def generate_new_puzzle(self, grid):
        while grid:
            last_grid = grid
            grid = self.remove_cell(grid)
        return last_grid

    def remove_cell(self, grid: Grid) -> list:
        """
        Given a board with a unique solution, returns a grid with a cell removed.
        If no cells can be removed without creating multiple solutions, returns None
        :param grid: Grid
        :return: Grid
        """

        xs, ys = np.nonzero(grid.array)
        indices = np.arange(len(xs))
        np.random.shuffle(indices)
        for i in indices:
            x, y = xs[i], ys[i]
            new_grid = grid.copy()
            new_grid.remove(x, y)
            if (new_grid in self and self[new_grid]) or len(self.find_all_solutions(new_grid)) == 1:
                return new_grid
        return None

    def find_all_solutions(self, grid: Grid, short_circuit=True) -> set:
        if short_circuit and grid in self:
            return {self[grid]}
        found_solutions = set()

        num_candidates = grid.count_candidates()
        min_num_candidates = min(num_candidates.values())
        candidate_cells = [coord for coord in num_candidates if num_candidates[coord] == min_num_candidates]
        x, y = random.choice(candidate_cells)

        for digit in grid.get_candidates(x, y):
            next_grid = grid.copy()
            next_grid.array[x, y] = digit # this is faster than write

            if next_grid in self:
                if self[next_grid]:
                    found_solutions.add(self[next_grid])
            elif next_grid.complete:
                # All other cells are filled, so only pencilmark x, y
                next_grid.pencil_marks[x, y] = np.zeros(next_grid.max_digit)
                self[next_grid] = next_grid
                found_solutions.add(next_grid)
            elif next_grid.is_solvable:
                next_grid.array[x, y] = 0
                next_grid.write(x, y, digit)
                found_solutions |= self.find_all_solutions(next_grid, short_circuit=short_circuit)

        if len(found_solutions) == 1:
            self[grid] = list(found_solutions)[0]
        else:
            self[grid] = None
        return found_solutions

    def set_seed_puzzles(self):
        self.seed_puzzles = Dict(list)
        for k, v in self.puzzles.items():
            if v and v.is_seed:
                self.seed_puzzles[v].append(k)

    def verify(self, valid_attempts=3, invalid_attempts=5, log=True):
        """
        Checks for
            1. All puzzles are valid puzzles
            1. All solutions are valid solutions
            2a. All seed solutions are in seed_puzzles
            2b. All seed_puzzle keys exist in puzzles
            2c. All puzzles in seed_puzzles have proper assignment in puzzles
            3. All seed puzzles are assigned to their seed solutions
            4. All valid puzzles have unique solutions
            5. All invalid puzzles have multiple solutions

        For steps 4 and 5, uses a random CSP solver 'attempts' times.
        Raises assertion error if a problem is found
        :return: None
        """
        bad_grids = Dict(list)
        logger.setLevel("DEBUG")
        logger.disabled = not log

        # Check 1: All puzzles are valid puzzles

        logger.info("Solutions verification starting check 1")
        for grid in self.puzzles:
            if not grid.valid:
                bad_grids['1'].append(grid)
                logger.error("Check 1: Invalid puzzle\n" + str(grid))
                # raise SolutionsVerificationException(grid, "Check 1: Invalid puzzle")

        # Check 2: All solutions are valid solutions
        logger.info("Solutions verification starting check 2")
        for grid in self.puzzles.values():
            if grid and not grid.valid:
                bad_grids['2'].append(grid)
                logger.error("Check 2: Invalid solution\n" + str(grid))
                # raise SolutionsVerificationException(grid, "Check 2: Invalid solution")

        # Check 3: All and only seed solutions are in seed puzzles
        logger.info("Solutions verification starting check 3")
        for grid in self.puzzles.values():
            if grid and grid.is_seed and grid not in self.seed_puzzles:
                bad_grids['3a'].append(grid)
                logger.error("Check 3a: Seed solution not in seed_puzzles\n" + str(grid))
                # raise SolutionsVerificationException(grid, "Check 3a: Seed solution not in seed_puzzles")
            if grid and (not grid.is_seed) and grid in self.seed_puzzles:
                bad_grids['3b'].append(grid)
                logger.error("Check 3b: Seed solution not in seed_puzzles\n" + str(grid))
                # raise SolutionsVerificationException(grid, "Check 3b: Non-seed solution in seed_puzzles")

        # Check 4: All seed solutions in seed_puzzles also exist in puzzles
        logger.info("Solutions verification starting check 4")
        all_solutions = set(self.puzzles.values())
        for grid in self.seed_puzzles:
            if grid not in all_solutions:
                bad_grids['4'].append(grid)
                logger.error("Check 4: Seed solution in seed_puzzles but not in puzzles\n" + str(grid))
                # raise SolutionsVerificationException(grid, "Check 4: Seed solution in seed_puzzles but not in puzzles")

        # Check 5: All puzzles in seed_puzzles also exist in puzzles and are correctly linked
        logger.info("Solutions verification starting check 5")
        for sol, puzzles in self.seed_puzzles.items():
            for grid in puzzles:
                if grid not in self.puzzles:
                    bad_grids['5a'].append(grid)
                    logger.error("Check 5a: Puzzle in seed_puzzles but not in puzzles\n" + str(grid))
                    # raise SolutionsVerificationException(grid,
                    #          "Check 5a: Puzzle in seed_puzzles but not in puzzles")
                if self.puzzles[grid] != sol:
                    bad_grids['5b'].append(grid)
                    logger.error("Check 5b: Puzzle in seed_puzzles not correctly linked in puzzles\n" + str(grid))
                    # raise SolutionsVerificationException(grid,
                    #          "Check 5b: Puzzle in seed_puzzles not correctly linked in puzzles")

        # Check 6 All valid puzzles have unique solutions
        # Check 7 All invalid puzzles have multiple solutions
        logger.info("Solutions verification starting checks 6 and 7")
        for k, v in tqdm(list(self.puzzles.items())):
            found_solutions = set()
            for i in range(valid_attempts if v else invalid_attempts):
                found_solutions.add(csp_solve(k))
            if v:
                if len(found_solutions) != 1:
                    bad_grids['6a'].append(grid)
                    logger.error("Check 6a: Valid puzzle has multiple solutions\n" + str(grid))
                    # raise SolutionsVerificationException(k, "Check 6a: Valid puzzle has multiple solutions")
                elif list(found_solutions)[0] != v:
                    bad_grids['6b'].append(grid)
                    logger.error("Check 6b: Valid puzzle's solution not same as in puzzles\n" + str(grid))
                    # raise SolutionsVerificationException(k, "Check 6b: Valid puzzle's solution not same as in puzzles")
            if v is None and len(found_solutions) == 1:
                bad_grids['7'].append(grid)
                logger.error("Check 7: Invalid puzzle only has 1 solution\n" + str(grid))
                # raise SolutionsVerificationException(k, "Check 7: Invalid puzzle only has 1 solution")


        logger.info("Validation complete!")
        logger.disabled = False
        logger.setLevel("INFO")
        return bad_grids

    def to_grid_string_solutions(self):
        solutions = GridStringSolutions(self.filename)
        solutions.puzzles = {k.to_grid_string(): v.to_grid_string() if v else "" for k, v in self.puzzles.items()}
        return solutions

    def save(self):
        s = self.to_grid_string_solutions()
        s.save()

    def load(self):
        s = GridStringSolutions(self.filename)
        s.load()
        self.puzzles = s.to_grid_solutions().puzzles
        self.set_seed_puzzles()

    # region __ functions

    def __getitem__(self, item):
        return self.puzzles[item]

    def __setitem__(self, key: Grid, value: Grid):
        assert type(key) == Grid
        assert (type(value) == Grid and value.complete) or value is None
        self.puzzles[key] = value

        if value and value.is_seed:
            self.seed_puzzles[value].append(key)

        if len(self.puzzles) % self.autosave == 0:
            self.save()

    def __contains__(self, item):
        return item in self.puzzles

    def __iter__(self):
        return iter(self.puzzles)

    def keys(self):
        return self.puzzles.keys()

    def values(self):
        return self.puzzles.values()

    def items(self):
        return self.puzzles.items()

    def __len__(self):
        return len(self.puzzles)

    # endregion


class GridStringSolutions:

    def __init__(self, filename=None):
        self.puzzles = {}
        self.filename = filename

    def to_grid_solutions(self):
        solutions = GridSolutions(self.filename)
        solutions.puzzles = {k.to_grid(): v.to_grid() if v else None for k, v in self.puzzles.items()}
        return solutions

    def save(self):
        with open(self.filename, 'wb') as f:
            pickle.dump(self.puzzles, f)

    def load(self):
        with open(self.filename, 'rb') as f:
            self.puzzles = pickle.load(f)

    def __getitem__(self, item):
        return self.puzzles[item]

    def __len__(self):
        return len(self.puzzles)

    def keys(self):
        return self.puzzles.keys()

    def values(self):
        return self.puzzles.values()

    def items(self):
        return self.puzzles.items()