import pickle
import gzip
import numpy as np


def load_data():

    def vectorized_result(i):
        output = np.zeros((10, 1))
        output[i] = 1.0
        return output

    f = gzip.open("../data/mnist.pkl.gz")
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

    return list(training_data), list(validation_data), list(test_data)