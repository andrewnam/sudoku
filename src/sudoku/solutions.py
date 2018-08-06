from board import Board

class Solutions:
	"""
	Maps each board to its solution in a space-efficient manner.
	A key board can map to a solution board or a None if no solution exists
	"""


	def __init__(self):
		self.solutionHashes = {}
		self.solutionBoards = {}

	def __setitem__(self, key: Board, item: Board):
		assert type(key) == Board
		assert (type(item) == Board and item.all_filled()) or item is None
		key_hash = key.boardHash()
		item_hash = item.boardHash()
		self.solutionHashes[key_hash] = item_hash
		self.solutionBoards[item_hash] = item

	def __getitem__(self, key):
		assert type(key) == Board
		solution_hash = self.solutionHashes[key.boardHash()]
		return self.solutionBoards[solution_hash] if solution_hash else None

	def __len__(self):
		return len(self.solutionHashes)

	def len_solutions(self):
		return len(self.solutionBoards)

	def __contains__(self, item):
		return item in self.solutionHashes

	def __iter__(self):
		return iter(self.solutionHashes)

	def has_key(self, k):
		return k in self.solutionHashes

	def keys(self):
		return self.solutionHashes.keys()

	def values(self):
		return self.solutionBoards.values()

