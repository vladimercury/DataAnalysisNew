import neural_network.reader as reader
from neural_network.network import NeuralNetwork
from util.frame import progress
from sklearn.metrics import f1_score, classification_report, accuracy_score
from util.dump import dump_object, load_object
from sys import stdout
from util.timer import Timer
import numpy as np
import warnings
import pylab as pt

warnings.filterwarnings('ignore')

DUMPED = False
CONTINUE = False


def images_to_np_array(image_data):
    return np.asarray([np.fromstring(i, dtype=np.uint8) / 256 for i in image_data])


def labels_to_np_array(labels_data):
    x = np.zeros((len(labels_data), 10))
    for i in range(len(labels_data)):
        x[i][labels_data[i]] = 1
    return x


def get_predicted(predict_data):
    return [max(range(len(i)), key=lambda x: i[x]) for i in predict_data]


stats_x, stats_y = [], []
if CONTINUE or DUMPED:
    stats_x, stats_y = load_object('stoch-n-images-stat.dump')
if not DUMPED or (DUMPED and CONTINUE):
    train_labels = []
    train_images = []
    image_size = (28, 28)
    timer = Timer()
    stdout.write('Loading Train data...')
    timer.set_new()
    train_labels = reader.read_labels('mnist/train-labels-idx1-ubyte')
    train_images = reader.read_images('mnist/train-images-idx3-ubyte')
    print('DONE in ' + timer.get_diff_str())
    image_size = train_images[1]

    stdout.write('Loading Test data...')
    timer.set_new()
    test_labels = reader.read_labels('mnist/t10k-labels-idx1-ubyte')
    test_images = reader.read_images('mnist/t10k-images-idx3-ubyte')
    print('DONE in ' + timer.get_diff_str())
    image_size = test_images[1]

    images_test = images_to_np_array(test_images[2])
    labels_test = labels_to_np_array(test_labels[1])
    rang_test = len(images_test)


    def classify():
        predicted = network.predict(images_test)
        predicted = get_predicted(predicted)
        return accuracy_score(test_labels[1], predicted)

    network = NeuralNetwork(1, 1, 1)
    images_train = images_to_np_array(train_images[2])
    labels_train = labels_to_np_array(train_labels[1])

    cycles = 1000
    print('Training...')
    progress(0)
    timer = Timer()
    rang = list(range(30, 70))
    for j in range(len(rang)):
        if not rang[j] in stats_x:
            np.random.seed(1)
            network = NeuralNetwork(image_size[0] * image_size[1], 10, 10)
            for i in range(cycles):
                randoms = np.random.randint(0, 60000, rang[j])
                network.train(images_train[randoms], labels_train[randoms])
                if i % 10 == 0:
                    progress((j * cycles + i + 1) / (cycles * len(rang)))
            stats_x.append(rang[j])
            stats_y.append(classify())
    progress(1)
    dump_object((stats_x, stats_y), 'stoch-n-images-stat.dump')
    print(' DONE in ', timer.get_diff_str())
pt.plot(stats_x, stats_y)
pt.show()