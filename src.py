import loader
from seqnet import SequentialNet


training_data, validation_data, test_data = loader.load_data()

net = SequentialNet([784, 30, 10])

net.stoc_grad_desc_learn(training_data=training_data,
                         epochs=30,
                         batch_size=10,
                         learn_rate=3.0,
                         test_data=test_data)


