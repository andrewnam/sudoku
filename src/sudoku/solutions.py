from board import Board


class Solutions:

    def __init__(self, solutions=None):
        self.solutions = {}
        if solutions:
            self.solutions = solutions

    def __setitem__(self, key: Board, item: Board):
        assert type(key) == Board
        assert (type(item) == Board and item.all_filled()) or item is None
        self.solutions[key] = item

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
        return self

