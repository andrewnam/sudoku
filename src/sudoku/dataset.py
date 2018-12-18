import numpy as np
import pickle

class Dataset:
    def __init__(self, data: dict, split_boundaries: tuple, split_data=None):
        """
        data: a non-nested dictionary of primitive key-value pairs where
            the key is the input and
            value is the target output
        """
        assert split_boundaries == sorted(split_boundaries)
        assert (split_boundaries[0] > 0 and split_boundaries[-1] < 1) or \
               split_boundaries[0] >= 1
        assert len(data) > len(split_boundaries)

        self.data = data
        self.split_boundaries = split_boundaries
        if split_data:
            self.split_data = split_data
        else:
            self.split_data = self.create_split_data()

    def create_split_data(self):
        fractions = self.split_boundaries[0] < 1

        inputs = list(self.data)
        np.random.shuffle(inputs)

        split = []
        last_boundary = 0

        if fractions:
            boundaries = [int(len(inputs) * boundary) for boundary in self.split_boundaries + [1]]
        else:
            boundaries = self.split_boundaries

        for boundary in boundaries:
            split.append(inputs[last_boundary:boundary])
            last_boundary = boundary
        return split

    def get_input_data(self, index=None):
        if index is None:
            return list(self.split_data)
        else:
            return self.split_data[index]

    def get_output_data(self, index=None):
        if index is None:
            return [[self.data[k] for k in inputs] for inputs in self.split_data]
        else:
            return [self.data[k] for k in self.split_data[index]]

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump({'data': self.data,
                         'split_boundaries': self.split_boundaries,
                         'split_data': self.split_data}, f)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            raw = pickle.load(f)
        return Dataset(raw['data'], raw['split_boundaries'], raw['split_data'])