import random

import numpy as np


def sigmoid(value):
    return 1.0 / (1.0 + np.exp(-value))


def sigmoid_prime(value):
    return sigmoid(value)*(1-sigmoid(value))


class SequentialNet:

    def __init__(self, sizes):

        self.num_layers = len(sizes)

        self.sizes = sizes

        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feed_forward(self, a):

        for w, b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a) + b)

        return a

    def backprop(self, x, y):

        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        activation = x
        activations = [x]
        zs = []

        for w, b in zip(self.weights, self.biases):

            z = np.dot(w, activation) + b
            zs.append(z)

            activation = sigmoid(z)
            activations.append(activation)

        delta = 2 * (activations[-1] - y) * sigmoid_prime(zs[-1])

        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        nabla_b[-1] = delta

        for layer_no in range(2, self.num_layers):

            z = zs[-layer_no]
            delta = np.dot(self.weights[-layer_no + 1].transpose(), delta) * sigmoid_prime(z)

            nabla_b[-layer_no] = delta
            nabla_w[-layer_no] = np.dot(delta, activations[-layer_no - 1].transpose())

        return nabla_w, nabla_b

    def process_batch(self, batch, learn_rate):

        nabla_w = [np.zeros(w_mat.shape) for w_mat in self.weights]
        nabla_b = [np.zeros(b_vec.shape) for b_vec in self.biases]

        for x, y in batch:

            part_w_grad, part_b_grad = self.backprop(x, y)

            nabla_w = [nab_w + part_w for nab_w, part_w in zip(nabla_w, part_w_grad)]
            nabla_b = [nab_b + part_b for nab_b, part_b in zip(nabla_b, part_b_grad)]

        scale_factor = -learn_rate / len(batch)

        self.weights = [w_mat + scale_factor * nab_w
                        for w_mat, nab_w in zip(self.weights, nabla_w)]

        self.biases = [b_vec + scale_factor * nab_b
                       for b_vec, nab_b in zip(self.biases, nabla_b)]

    def test_network(self, test_data):

        test_results = [(np.argmax(self.feed_forward(x)), np.argmax(y)) for x, y in test_data]

        return sum([int(x == y) for x, y in test_results])

    def stoc_grad_desc_learn(self, training_data, epochs, batch_size, learn_rate, test_data=None):

        for epoch_no in range(epochs):

            random.shuffle(training_data)

            batches = [training_data[n: n + batch_size]
                       for n in range(0, len(training_data), batch_size)]

            for batch in batches:
                self.process_batch(batch, learn_rate)

            if test_data:
                print("Epoch {0}: {1} / {2}".format(
                    epoch_no, self.test_network(test_data), len(test_data)))
            else:
                print("Epoch {0} complete".format(epoch_no))
