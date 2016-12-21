import numpy as np
import sys

np.seterr(over='ignore')


class NeuralNetwork:
    def __init__(self, inputs, hidden, outputs, seed=1, layers=1):
        self.inputs = inputs
        self.hidden = hidden
        self.outputs = outputs

        self.nlayers = layers
        self.layers = [np.asarray([[]]) for i in range(layers)]

        self.cycles = 0
        self.step = 20
        self.stats = []

        np.random.seed(seed)

        self.input_synapse = 2 * np.random.random((inputs, hidden)) - 1
        self.hidden_synapse = [2 * np.random.random((hidden, hidden)) - 1 for i in range(layers-1)]
        self.output_synapse = 2 * np.random.random((hidden, outputs)) - 1

    def predict(self, data):
        self.layers[0] = self._sigmoid(np.dot(data, self.input_synapse))
        for i in range(1, self.nlayers):
            self.layers[i] = self._sigmoid(np.dot(self.layers[i-1], self.hidden_synapse[i-1]))
        return self._sigmoid(np.dot(self.layers[-1], self.output_synapse))

    def train(self, data, labels, a=1):
        predicted = self.predict(data)
        output_error = labels - predicted
        output_delta = output_error * self._sigmoid_deriv(predicted) * a
        hidden_delta = [0 for i in range(self.nlayers)]
        hidden_error = np.dot(output_delta, self.output_synapse.T)
        hidden_delta[-1] = hidden_error * self._sigmoid_deriv(self.layers[-1]) * a
        for i in range(2, self.nlayers + 1):
            hidden_error = np.dot(hidden_delta[-i + 1], self.hidden_synapse[-i + 1].T)
            hidden_delta[-i] = hidden_error * self._sigmoid_deriv(self.layers[-i]) * a
        self.input_synapse += np.dot(np.transpose(data), hidden_delta[0])
        for i in range(1, self.nlayers):
            self.hidden_synapse[i - 1] += np.dot(np.transpose(self.layers[i]), hidden_delta[i])
        self.output_synapse += np.dot(np.transpose(self.layers[-1]), output_delta)
        self.cycles += 1
        if self.cycles % self.step == 0:
            error = np.mean(np.abs(labels - self.predict(data)))
            self.stats.append(error)

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def _sigmoid_deriv(x):
        sigmoid = NeuralNetwork._sigmoid(x)
        return sigmoid * (1 - sigmoid)