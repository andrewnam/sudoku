"""
ShuffledGrid allows creates a transformed Sudoku grid by exploiting symmetrical structures
Each ShuffledGrid is an instance of a parent seed grid and a set of transformations,
making it easy to track exactly how this grid was generated

A useful feature is mirroring the transformations of one ShuffledGrid onto another so that
a puzzle and solution grid can be tracked together
"""

from grid import Grid
from enum import Enum
import numpy as np


Rotations = Enum('Rotation', '0 90 180 270')


class ShuffledGrid:
    def __init__(self, parent_grid: Grid,
                 labels=None,
                 stacks=None,
                 bands=None,
                 columns=None,
                 rows=None,
                 reflection=False,
                 rotation=None):

        self.parent_grid = parent_grid
        self.labels = labels
        self.stacks = stacks
        self.bands = bands
        self.columns = columns
        self.rows = rows
        self.reflection = reflection
        self.rotation = rotation

        self._grid = None

    @property
    def grid(self):
        if self._grid is None:
            self._grid = self._grid.copy()

            if self.labels:
                self.relabel()

        return self._grid

    def mirror(self, other):
        self.labels = other.labels
        self.stacks = other.stacks
        self.bands = other.bands
        self.columns = other.columns
        self.rows = other.rows
        self.reflection = other.reflection
        self.rotation = other.rotation
        self._grid_string = None
        return self

    def relabel(self):
        assert type(self.labels) is dict
        self.labels[0] = 0
        assert set(self.labels) == set(self.labels.values()) == set(range(len(self.labels)))

        self.grid._array = np.vectorize(lambda x: self.labels[x])(self.parent_grid_string.array)