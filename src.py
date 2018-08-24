import pickle
import gzip
import numpy as np


def load_data():

    def vectorized_result(i):
        output = np.zeros((10, 1))
        output[i] = 1.0
        return output

    f = gzip.open("./data/mnist.pkl.gz")
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    f.close()

    training_inputs = [np.reshape(x, (784, 1)) for x in train_set[0]]
    training_results = [vectorized_result(y) for y in train_set[1]]
    training_data = zip(training_inputs, training_results)

    validation_inputs = [np.reshape(x, (784, 1)) for x in valid_set[0]]
    validation_data = zip(validation_inputs, valid_set[1])

    test_inputs = [np.reshape(x, (784, 1)) for x in test_set[0]]
    test_results = [vectorized_result(y) for y in test_set[1]]
    test_data = zip(test_inputs, test_results)

    return training_data, validation_data, test_data


class Network(object):

    def __init__(self, sizes):

        self.num_layers = len(sizes)

        self.sizes = sizes

        self.biases = [np.random.randn(x, 1) for x in sizes[1:]]

        self.weights = []

        for index in range(0, len(sizes) - 1):

            self.weights.append(np.random.randn(sizes[index + 1], sizes[index]))

    def feed_forward(self, a):

        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-x))

        for w, b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def grad_desc_learn(self, input_set, output_set, cycles_no, step_size):

        def compute_cost():
            cost = 0

            for real_input, expected_output in zip(input_set, output_set):
                net_output = self.feed_forward(real_input)
                local_cost = np.linalg.norm(net_output - expected_output)
                cost += local_cost

            return cost

        def bias_derivative(i, j, initial_cost):

            delta = 0.00001
            self.biases[i][j, 0] += delta
            new_cost = compute_cost()
            self.biases[i][j, 0] -= delta
            derivative = (new_cost - initial_cost) / delta

            return derivative

        for cycle in range(7):

            initial_cost = compute_cost()
            new_biases = [np.zeros((x, 1)) for x in self.sizes[1:]]

            for i in range(len(self.biases)):
                for j in range(len(self.biases[i])):

                    new_biases[i][j, 0] = self.biases[i][j, 0] - step_size * bias_derivative(i, j, initial_cost)

            self.biases = new_biases

            print("Cycle " + str(cycle) + ": cost function --- " + str(initial_cost))
























train, valid, test = load_data()
train_in, train_out = zip(*train)
test_in, test_out = zip(*valid)

net = Network([784, 5, 10])
net.grad_desc_learn(train_in, train_out, 10, 0.0001)



