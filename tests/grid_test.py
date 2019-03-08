from sudoku2 import Grid, InvalidWriteException
import pytest
import numpy as np

def test_write():
    g = Grid(3, 3)
    g.write(3, 3, 4)
    with pytest.raises(InvalidWriteException):
        g.write(3, 3, 3)
    with pytest.raises(InvalidWriteException):
        g.write(0, 3, 4)
    with pytest.raises(InvalidWriteException):
        g.write(3, 0, 4)
    with pytest.raises(InvalidWriteException):
        g.write(4, 4, 4)

    g.write(3, 2, 3)
    g.write(5, 5, 1)

    assert g[3][3] == 4
    assert g[3][2] == 3
    assert g[5][5] == 1
    assert sum(g.pencil_marks[3][3]) == 0
    assert not g.pencil_marks[3][4][0]
    assert g.pencil_marks[3][4][1]
    assert not g.pencil_marks[3][4][2]
    assert not g.pencil_marks[3][4][3]
    assert g.pencil_marks[3][4][4]
    assert g.pencil_marks[3][4][5]
    assert g.pencil_marks[3][4][6]
    assert g.pencil_marks[3][4][7]
    assert g.pencil_marks[3][4][8]

def test_remove():
    g = Grid(3, 3)
    g.write(3, 3, 4)
    g.write(3, 4, 5)
    g.write(5, 5, 6)
    g.write(1, 1, 3)
    g.write(2, 2, 4)
    g.write(0, 8, 4)
    g.write(4, 7, 4)

    assert sum(g.pencil_marks[3][3]) == 0
    assert sum(g.rows[3].digit_pencil_marks(g[3][3])) == 0
    assert sum(g.columns[3].digit_pencil_marks(g[3][3])) == 0
    assert sum(g.box_containing(3, 3).digit_pencil_marks(g[3][3])) == 0

    g.remove(3, 3)
    pencilmarks = np.ones(g.max_digit)
    pencilmarks[4] = False
    pencilmarks[5] = False
    assert (g.pencil_marks[3][3] == pencilmarks).all()
    assert (g.pencil_marks[5][4] == pencilmarks).all()
    pencilmarks[3] = False
    assert not (g.pencil_marks[5][4] == pencilmarks).all()
    assert (g.pencil_marks[4][4] == pencilmarks).all()
    pencilmarks[5] = True

    assert (g.pencil_marks[3][2] == pencilmarks).all()
    assert (g.pencil_marks[3][6] == pencilmarks).all()
    assert (g.pencil_marks[3][7] == pencilmarks).all()
    assert (g.pencil_marks[3][8] == pencilmarks).all()


    pencilmarks = np.ones(g.max_digit)
    pencilmarks[2] = False
    pencilmarks[4] = False

    assert (g.pencil_marks[3][1] == pencilmarks).all()