from board import Board
import random
import numpy as np
from tqdm import tqdm
from grid_string import GridString
from Dict import Dict
import utils

class Solutions:
    MEMORY_ONLY = "__MEMORY_ONLY__"

    def __init__(self, filename, solutions=None):
        self.solutions = {}
        self.filename = filename

        self.solution_boards = set()
        self.seed_solutions = set()

        if solutions:
            self.solutions = solutions

        self.refresh_solution_boards()
        self.refresh_seed_solutions()

        self.load()
        print("Successfully loaded from {}".format(filename))
        # try:
        #     self.load()
        #     print("Successfully loaded from {}".format(filename))
        # except:
        #     pass

    def __setitem__(self, key: Board, item: Board):
        assert type(key) == Board
        assert (type(item) == Board and item.all_filled()) or item is None
        self.solutions[key] = item
        if item:
            self.solution_boards.add(item)
            if np.all(item[0] == np.arange(item.max_digit) + 1):
                self.seed_solutions.add(item)

        # Auto-save every 1000 entries
        if len(self.solutions)%2500 == 0:
            self.save()

    def __getitem__(self, key):
        assert type(key) == Board
        return self.solutions[key]

    def __len__(self):
        return len(self.solutions)

    def __contains__(self, item):
        return item in self.solutions

    def __iter__(self):
        return iter(self.solutions)

    def has_key(self, k):
        return k in self.solutions

    def keys(self):
        return self.solutions.keys()

    def values(self):
        return self.solutions.values()

    def items(self):
        return self.solutions.items()

    def refresh_solution_boards(self):
        self.solution_boards = set(self.solutions.values())
        if None in self.solution_boards:
            self.solution_boards.remove(None)

    def refresh_seed_solutions(self):
        self.seed_solutions = {board for board in self.solutions.values() if board and board.is_seed()}

    def save(self):
        if self.filename is Solutions.MEMORY_ONLY:
            return

        lines = []
        for k, v in self.solutions.items():
            k_str = k.stringify()
            v_str = ''
            if v is not None:
                v_str = v.stringify()
            lines.append(','.join([k_str, v_str]))
        with open(self.filename, 'w') as f:
            f.write('\n'.join(lines))

    def load(self):
        if self.filename is Solutions.MEMORY_ONLY:
            return

        with open(self.filename) as f:
            lines = f.read().splitlines()

        solution_boards = {}
        puzzle_boards = {}

        for line in lines:
            k, v = line.split(',')
            puzzle_boards[k] = v
            if k == v:
                solution_boards[k] = GridString(k).board

        for k, v in puzzle_boards.items():
            if v == '':
                self.solutions[GridString(k).board] = None
            else:
                self.solutions[GridString(k).board] = solution_boards[v]
        self.refresh_solution_boards()
        self.refresh_seed_solutions()
        return self

    def count_puzzles_per_solution(self):
        occurrences = {}
        for k, v in [(k, v) for k, v in self.solutions.items() if v and v.is_seed()]:
            if v not in occurrences:
                occurrences[v] = 0
            occurrences[v] += 1
        return occurrences

    def get_random_seed_solution(self, inverse_weight=False):
        if inverse_weight:
            seeds = list(self.seed_solutions)
            weights = {k: 1/v for k, v in self.count_puzzles_per_solution().items()}
            weight_list = np.array([weights[seeds[i]] for i in range(len(seeds))])
            total_weight = np.sum(weight_list)
            weight_list /= total_weight
            randint = np.random.choice(np.arange(len(seeds)), p=weight_list)
            return seeds[randint]
        return random.sample(self.seed_solutions, 1)[0]

    def get_num_puzzles_per_hint(self):
        """
        :return: A dictionary
         { seed_solution: {num_hint: num_puzzles} }
        """
        counts = Dict(int, 2)
        for p, s in self.solutions.items():
            if s in self.seed_solutions:
                num_hints = p.count_filled_cells()
                counts[s][num_hints] += 1
        return counts

    def get_min_puzzle_seed_solution(self):
        counts = self.get_num_puzzles_per_hint()
        min_hints = min(utils.flatten([v.keys() for v in counts.values()]))
        min_hint_counts = {k: v[min_hints] for k, v in counts.items()}
        min_board = min(min_hint_counts, key=lambda k: min_hint_counts[k])
        return min_board


    def find_all_solutions(self, board: Board) -> set:
        """
        Returns a set of Boards that contain all possible solutions to the input board
        :param board:
        :param solutions: dictionary of board K -> board V where V is the solution to K. V may be None
        :return: set of Boards
        """
        if board in self:
            return {self[board]}

        found_solutions = set()
        possibilities = board.get_cell_possibilities_count()
        min_possibilities = [coord for coord in possibilities if possibilities[coord] == min(possibilities.values())]
        x, y = random.choice(min_possibilities)
        for digit in board.get_possible_digits(x, y):
            next_board = board.copy()
            next_board.write(x, y, digit)
            if next_board in self:
                found_solutions.add(self[next_board])
            elif next_board.all_filled():
                self[next_board] = next_board
                found_solutions.add(self[next_board])
            if next_board.is_solvable():
                found_solutions |= self.find_all_solutions(next_board)
        if len(found_solutions) == 1:
            self[board] = list(found_solutions)[0]
        else:
            self[board] = None
        return found_solutions

    def check_valid_boards(self):
        violations = []
        for board, sol in tqdm([(b, s) for b, s in self.items() if s]):
            if sol and board != sol:
                all_sols = list(Solutions(Solutions.MEMORY_ONLY).find_all_solutions(board))
                if not len(all_sols) is 1 and all_sols[0] == sol:
                    violations.append((board, all_sols))
        return violations

    def check_invalid_boards(self):
        """
        Checks that boards that map to no solution actually do not have a unique solution.
        Takes much longer than checking for valid boards.
        :return:
        """
        violations = []
        for board, sol in tqdm([(b, s) for b, s in self.items() if not s]):
            if not len(Solutions(Solutions.MEMORY_ONLY).find_all_solutions(board)) != 1:
                violations.append((board, sol))
        return violations