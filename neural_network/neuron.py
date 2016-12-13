from random import randint
from numpy import asarray


class Neuron:
    def __init__(self, row, col):
        self.minimum = 50
        self.row = row
        self.col = col
        self.size = row * col
        self.weight = asarray([randint(0, 10) for j in range(self.size)])

    def transfer_hard(self, input_bytes):
        return 1 if self.transfer(input_bytes) >= self.minimum else 0

    def transfer(self, input_bytes):
        return sum([self.weight[i] * input_bytes[i] for i in range(self.size)])

    def change_weights(self, input_bytes, coef):
        for i in range(self.size):
            self.weight[i] += coef * input_bytes[i]


class NeuralNetwork:
    def __init__(self, size, row, col):
        self.size = size
        self.row = row
        self.col = col
        self.neurons = [Neuron(row, col) for i in range(size)]

    def handle_hard(self, input_bytes):
        return [self.neurons[i].transfer_hard(input_bytes) for i in range(self.size)]

    def handle(self, input_bytes):
        return [self.neurons[i].transfer(input_bytes) for i in range(self.size)]

    def get_answer(self, input_bytes):
        output = self.handle(input_bytes)
        return max(range(len(output)), key=lambda x: output[x])

    def study(self, input_bytes, label):
        correct = [0] * self.size
        correct[label] = 1
        output = self.handle_hard(input_bytes)
        while not correct== output:
            for i in range(self.size):
                diff = correct[i] - output[i]
                self.neurons[i].change_weights(input_bytes, diff)
            output = self.handle_hard(input_bytes)