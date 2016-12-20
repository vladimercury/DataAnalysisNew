import numpy as np

np.seterr(over='ignore')


class NeuralNetwork:
    def __init__(self, inputs, hidden, outputs, seed=1):
        self.inputs = inputs
        self.hidden = hidden
        self.outputs = outputs

        self.cycles = 0
        self.step = 1
        self.stats = []

        np.random.seed(seed)

        self.input_synapse = 2 * np.random.random((inputs, hidden)) - 1
        self.output_synapse = 2 * np.random.random((hidden, outputs)) - 1

        self.hidden_layer = np.asarray([[]])

    def predict(self, data):
        self.hidden_layer = self._sigmoid(np.dot(data, self.input_synapse))
        return self._sigmoid(np.dot(self.hidden_layer, self.output_synapse))

    def train(self, data, labels, a=1):
        predicted = self.predict(data)
        output_error = labels - predicted
        self.cycles += 1
        if self.cycles % self.step == 0:
            self.stats.append(np.mean(np.abs(output_error)))
        output_delta = output_error * self._sigmoid_deriv(predicted) * a
        hidden_error = np.dot(output_delta, self.output_synapse.T)
        hidden_delta = hidden_error * self._sigmoid_deriv(self.hidden_layer) * a
        self.input_synapse += np.dot(np.transpose(data), hidden_delta)
        self.output_synapse += np.dot(np.transpose(self.hidden_layer), output_delta)

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def _sigmoid_deriv(x):
        sigmoid = NeuralNetwork._sigmoid(x)
        return sigmoid * (1 - sigmoid)