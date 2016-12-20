import neural_network.reader as reader
from neural_network.network import NeuralNetwork
from util.frame import progress
from sklearn.metrics import f1_score, classification_report
from util.dump import dump_object, load_object
from sys import stdout
from util.timer import Timer
import numpy as np
import warnings
import pylab as pt
warnings.filterwarnings('ignore')

def images_to_np_array(image_data):
    return np.asarray([np.fromstring(i, dtype=np.uint8) / 256 for i in image_data])


def labels_to_np_array(labels_data):
    x = np.zeros((len(labels_data), 10))
    for i in range(len(labels_data)):
        x[i][labels_data[i]] = 1
    return x


def get_predicted(predict_data):
    return [max(range(len(i)), key=lambda x: i[x]) for i in predict_data]


def classify():
    predicted = network.predict(test_data)
    predicted = get_predicted(predicted)
    return f1_score(test_labels_raw, predicted)

DUMPED = False
CONTINUE = True
timer = Timer()

stats = []
if not DUMPED:
    image_size = load_object('fs_size.dump')
    train_data = load_object('fs_train_data.dump')
    train_labels_raw = load_object('fs_train_labels.dump')
    train_labels = labels_to_np_array(train_labels_raw)
    test_data = load_object('fs_test_data.dump')
    test_labels_raw = load_object('fs_test_labels.dump')
    test_labels = labels_to_np_array(test_labels_raw)

    network = NeuralNetwork(image_size, 30, 10)
    if CONTINUE:
        network = load_object('fs_network.dump')
        stats = load_object('fs_stats.dump')
    print('Training...')
    cycles = 30
    timer = Timer()
    progress(0)
    for i in range(cycles):
        network.train(train_data, train_labels)
        progress((i+1) / cycles)
        stats.append(classify())
    print(' DONE in ', timer.get_diff_str())
    dump_object(network, 'fs_network.dump')
    dump_object(stats, 'fs_stats.dump')
else:
    stats = load_object('fs_stats.dump')
pt.plot(range(len(stats)), stats)
pt.show()