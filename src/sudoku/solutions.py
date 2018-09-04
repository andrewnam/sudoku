from board import Board
import random
import numpy as np


class Solutions:

    def __init__(self, solutions=None):
        self.solutions = {}
        self.solution_boards = set()
        self.seed_solutions = set()

        if solutions:
            self.solutions = solutions

        self.refresh_solution_boards()
        self.refresh_seed_solutions()

    def __setitem__(self, key: Board, item: Board):
        assert type(key) == Board
        assert (type(item) == Board and item.all_filled()) or item is None
        self.solutions[key] = item
        if item:
            self.solution_boards.add(item)
            if np.all(item[0] == np.arange(item.max_digit) + 1):
                self.seed_solutions.add(item)

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

    def save(self, filename):
        lines = []
        for k, v in self.solutions.items():
            k_str = k.stringify()
            v_str = ''
            if v is not None:
                v_str = v.stringify()
            lines.append(','.join([k_str, v_str]))
        with open(filename, 'w') as f:
            f.write('\n'.join(lines))

    def load(self, filename):
        with open(filename) as f:
            lines = f.read().splitlines()

        solution_boards = {}
        puzzle_boards = {}

        for line in lines:
            k, v = line.split(',')
            puzzle_boards[k] = v
            if k == v:
                solution_boards[k] = Board.loadFromString(k)

        for k, v in puzzle_boards.items():
            if v == '':
                self.solutions[Board.loadFromString(k)] = None
            else:
                self.solutions[Board.loadFromString(k)] = solution_boards[v]
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

