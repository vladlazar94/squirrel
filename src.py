import pickle
import gzip
import numpy as np


def vectorized_result(i):

    output = np.zeros((10, 1))
    output[i] = 1.0

    return output


def load_data():

    f = gzip.open("./data/mnist.pkl.gz")
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    f.close()

    training_inputs = [np.reshape(x, (784, 1)) for x in train_set[0]]
    training_results = [vectorized_result(y) for y in train_set[1]]
    training_data = zip(training_inputs, training_results)

    validation_inputs = [np.reshape(x, (784, 1)) for x in valid_set[0]]
    validation_data = zip(validation_inputs, valid_set[1])

    test_inputs = [np.reshape(x, (784, 1)) for x in test_set[0]]
    test_data = zip(test_inputs, test_set[1])

    return training_data, validation_data, test_data


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class Network(object):

    def __init__(self, sizes):

        self.num_layers = len(sizes)

        self.sizes = sizes

        self.biases = [np.random.randn(x, 1) for x in sizes[1:]]

        self.weights = []

        for index in range(0, len(sizes) - 1):

            self.weights.append(np.random.randn(sizes[index + 1], sizes[index]))

    def feed_forward(self, a):

        for w, b in zip(self.weights, self.biases):

            a = sigmoid(np.dot(w, a) + b)

        return a


net = Network([3, 5, 2])