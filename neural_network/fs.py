from sys import stdout
from util.timer import Timer
from util.dump import load_object, dump_object
import neural_network.reader as reader
from feature_selection.spearman import spearman
from feature_selection.pearson import pearson
from feature_selection.information_gain import information_gain
import numpy as np


def images_to_np_array(image_data):
    return np.asarray([np.fromstring(i, dtype=np.uint8) / 256 for i in image_data])

timer = Timer()
FS_DUMPED = False
if not FS_DUMPED:
    stdout.write('Loading Train data...')
    timer.set_new()
    train_labels_file = reader.read_labels('mnist/train-labels-idx1-ubyte')
    train_images_file = reader.read_images('mnist/train-images-idx3-ubyte')
    train_data = images_to_np_array(train_images_file[2])
    train_labels = np.asarray(train_labels_file[1])
    print('DONE in ' + timer.get_diff_str())

    stdout.write('Loading Test data...')
    timer.set_new()
    test_labels_file = reader.read_labels('mnist/t10k-labels-idx1-ubyte')
    test_images_file = reader.read_images('mnist/t10k-images-idx3-ubyte')
    test_data = images_to_np_array(test_images_file[2])
    test_labels = np.asarray(test_labels_file[1])
    print('DONE in ' + timer.get_diff_str())
    # timer.set_new()
    # coef = information_gain(train_data, train_labels)
    # print(' DONE in ' + timer.get_diff_str())
    # dump_object(coef, 'spearman.dump')
    import pylab as pt

    ig = [x[1] for x in sorted(load_object('ig.dump'))]

    y = np.zeros((28, 28, 3))
    n = 150
    features = ig[-n:]
    for i in features:
        y[i // 28][i % 28] = [1, 1, 1]
    pt.imshow(y)
    pt.show()

    fs_data = train_data.T[features].T
    fs_labels = train_labels

    fs_test_data = test_data.T[features].T
    fs_test_labels = test_labels

    dump_object(n, 'fs_size.dump')
    dump_object(fs_data, 'fs_train_data.dump')
    dump_object(fs_labels, 'fs_train_labels.dump')
    dump_object(fs_test_data, 'fs_test_data.dump')
    dump_object(fs_test_labels, 'fs_test_labels.dump')