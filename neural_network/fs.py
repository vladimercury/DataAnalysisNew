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
    # stdout.write('Loading Train data...')
    # timer.set_new()
    # train_labels_file = reader.read_labels('mnist/train-labels-idx1-ubyte')
    # train_images_file = reader.read_images('mnist/train-images-idx3-ubyte')
    # print('DONE in ' + timer.get_diff_str())
    #
    # train_data = images_to_np_array(train_images_file[2])
    # train_labels = train_labels_file[1]

    # timer.set_new()
    # coef = spearman(train_data, train_labels)
    # print(' DONE in ' + timer.get_diff_str())
    # dump_object(coef, 'spearman.dump')
    import pylab as pt

    ig = [x[1] for x in sorted(load_object('ig.dump'))]

    y = np.zeros((28, 28, 3))
    n = 256
    for j in range(n, 0, -1):
        features = ig[-j:]
        for i in features:
            y[i // 28][i % 28] = [(n-j)/n, (n-j)/n, (n-j)/n]
    pt.imshow(y)
    pt.show()