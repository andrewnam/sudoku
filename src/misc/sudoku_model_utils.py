import torch
import itertools

def collect_batches(outputs, iters):
    """
    Collects parallel-gpu batches from running forward on a DataParallel RRN model.
    If x and y are both of shape (n, c) where n = number of puzzles and c = number of cells,
     a forward pass with iters iterations will return a tensor of shape (iters, n, c, max_digit).
    If using DataParallel with d devices, the shape will instead be (iters*d, n/d, c, max_digit).
    This function takes the latter and returns the former and is safe to use even when d is 1.
    Therefore, it is advised to always call collect_batches(model(x, iters), iters).
    This cannot be put into the forward function since it needs to wrap around the DataParallel abstraction.

    :param outputs: Tensor of shape (iters*d, n/d, c, max_digit)
    :param iters: Number of iters when running forward
    :return: Tensor of shape (iters, n, c, max_digit)
    """
    return torch.cat([outputs[i:i + iters] for i in range(0, len(outputs), iters)], dim=1)


def determine_edges(dim_x, dim_y):
    """
    Returns a 2-d array of (max_digit**2, n) where the i_th entry is a list of
        other cells' indices that cell i shares a house with
    """
    max_digit = dim_x * dim_y
    edges = []
    for row in range(max_digit):
        row_edges = []
        for col in range(max_digit):
            # row & column
            col_edges = {(row, i) for i in range(max_digit)}
            col_edges |= {(i, col) for i in range(max_digit)}

            # box
            x_min = (row // dim_x) * dim_x
            y_min = (col // dim_y) * dim_y
            col_edges |= set(itertools.product(range(x_min, x_min + dim_x), range(y_min, y_min + dim_y)))

            # removing self
            col_edges -= {(row, col)}
            col_edges = [row * max_digit + col for row, col in col_edges]
            row_edges.append(sorted(col_edges))
        edges.append(row_edges)
    edges = torch.tensor(edges)
    shape = edges.shape
    return edges.view(max_digit ** 2, shape[2])