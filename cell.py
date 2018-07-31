class Cell:
	def __init__(self, x, y, digit):
		self.x = x
		self.y = y
		self.digit = digit

	def __repr__(self):
		return str((self.x, self.y, self.digit))