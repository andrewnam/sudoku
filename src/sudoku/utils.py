import numpy as np


def flatten(lst):
    return [item for sublist in lst for item in sublist]


def get_combinations(a, b):
    return np.array(np.meshgrid(a, b)).T.reshape(-1, 2)


def datetime_to_str(datetime):
    return datetime.strftime("%Y-%m-%d_%H:%M:%S")
