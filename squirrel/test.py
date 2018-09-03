from squirrel.loader import load_data
from squirrel.seqnet import SequentialNet


training_data, validation_data, test_data = load_data()

seq_net = SequentialNet([784, 30, 10])

seq_net.stoc_grad_desc_learn(training_data=training_data,
                             epochs=30,
                             batch_size=10,
                             learn_rate=3.0,
                             test_data=test_data)

