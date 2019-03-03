import numpy as np
import pickle
import utils


class Dataset:
    def __init__(self, data: dict):
        """
        data: a non-nested dictionary of primitive key-value pairs where
            the key is the input and
            value is the target output
        """
        self.data = data
        self.input_order = list(self.data)
        np.random.shuffle(self.input_order)

    def split_data(self, boundaries):
        assert boundaries == sorted(boundaries)
        assert (boundaries[0] > 0 and boundaries[-1] < 1) or \
               boundaries[0] >= 1
        assert len(self.data) > len(boundaries)

        if boundaries[-1] < 1:
            boundaries = [int(len(self.input_order) * boundary) for boundary in boundaries + [1]]
        else:
            boundaries = boundaries + [len(self.input_order)]

        inputs = []
        last_boundary = 0
        for boundary in boundaries:
            inputs.append(self.input_order[last_boundary:boundary])
            last_boundary = boundary
        return inputs, [[self.data[k] for k in s] for s in inputs]

    def get_input_data(self, boundary_a=None, boundary_b=None):
        if boundary_a == boundary_b == None:
            return self.input_order[:]
        elif boundary_a > 0 and boundary_b is None:
            return self.input_order[:boundary_a]
        elif boundary_b > boundary_a:
            return self.input_order[boundary_a:boundary_b]
        assert False

    def get_output_data(self, boundary_a=None, boundary_b=None):
        inputs = self.get_input_data(boundary_a, boundary_b)
        return [self.data[k] for k in inputs]

    def __len__(self):
        return len(self.data)

    def keys(self):
        return self.get_input_data()

    def values(self):
        return self.get_output_data()

    def items(self):
        return zip(self.get_input_data(), self.get_output_data())

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump({'data': self.data,
                         'input_order': self.input_order}, f)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        dst = Dataset(data['data'])
        dst.input_order = data['input_order']
        return dst

class Datasets:
    """
    Allows operations to be performed across datasets and return a new dataset
    """

    def __init__(self, datasets: dict):
        """
        :param datasets: dict of datasets, key = dataset name, value = Dataset
        """
        self.datasets = datasets

    def split_data(self, boundaries):
        num_bins = len(boundaries) + 1
        all_inputs, all_outputs = [[] for i in range(num_bins)], [[] for i in range(num_bins)]
        for dst in self.datasets.values():
            inputs, outputs = dst.split_data(boundaries)
            for i in range(len(inputs)):
                all_inputs[i] += inputs[i]
                all_outputs[i] += outputs[i]
        return all_inputs, all_outputs

    def __len__(self):
        return len(self.datasets)

    def keys(self):
        return self.datasets.keys()

    def values(self):
        return self.datasets.values()

    def items(self):
        return zip(self.keys(), [self.datasets[k] for k in self.keys()])

    def data_keys(self):
        return utils.flatten([self.datasets[k].keys() for k in sorted(self.datasets)])

    def data_values(self):
        return utils.flatten([self.datasets[k].values() for k in sorted(self.datasets)])

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump({ name: {
                                'data': dst.data,
                                'input_order': dst.input_order
                                }
                          for name, dst in self.datasets.items()
                         }, f)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        datasets = {}
        for name in data:
            dst = Dataset(data[name]['data'])
            dst.input_order = data[name]['input_order']
            datasets[name] = dst
        return Datasets(datasets)
