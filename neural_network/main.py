import neural_network.reader as reader
from neural_network.neuron import NeuralNetwork
from util.frame import progress
from sklearn.metrics import classification_report
from util.dump import dump_object, load_object
from sys import stdout

NETWORK_DUMPED = True
NEED_CLASSIFY = False

train_labels = []
train_images = []
test_labels = []
test_images = [0, (0, 0)]
image_size = (28, 28)
if not NETWORK_DUMPED:
    stdout.write('Loading Train data...')
    train_labels = reader.read_labels('mnist/train-labels-idx1-ubyte')
    train_images = reader.read_images('mnist/train-images-idx3-ubyte')
    print('DONE')
    image_size = train_images[1]
if NEED_CLASSIFY:
    stdout.write('Loading Test data...')
    test_labels = reader.read_labels('mnist/t10k-labels-idx1-ubyte')
    test_images = reader.read_images('mnist/t10k-images-idx3-ubyte')
    print('DONE')
    image_size = test_images[1]


def to_int_vector(byte_vector):
    return [byte_vector[i] for i in range(len(byte_vector))]

network = NeuralNetwork(1, 1, 1)
if NETWORK_DUMPED:
    network = load_object('network.dump')
else:
    network = NeuralNetwork(10, image_size[0], image_size[1])
    images = train_images[2]
    labels = train_labels[1]
    rang = len(images)
    print('Training...')
    for i in range(rang):
        network.study(images[i], labels[i])
        if i % 10 == 0 or i == rang - 1:
            progress((i + 1) / rang)
    print()
    dump_object(network, 'network.dump')

if NEED_CLASSIFY:
    images = test_images[2]
    labels = test_labels[1]
    rang = len(images)
    predicted = []
    print('Classification...')
    for i in range(rang):
        predicted.append(network.get_answer(images[i]))
        if i % 10 == 0 or i == rang - 1:
            progress((i + 1) / rang)
    print()
    print(classification_report(labels[:rang], predicted))

import pylab as pt
import numpy as np
pt.pcolor(np.asarray(network.neurons[6].weight).reshape(image_size))
pt.colorbar()
pt.show()