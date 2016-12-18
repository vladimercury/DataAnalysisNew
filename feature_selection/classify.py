import util.dump as dump
import util.frame as frame
import feature_selection.trim as trim
import sklearn.metrics as metrics
import numpy as np
import warnings
from util.timer import Timer
warnings.filterwarnings('ignore')

timer = Timer()

INFO_GAIN = True
PEARSON = True
SPEARMAN = True
OVERALL= True

IG_NBEST = True
PEARSON_NBEST = True
SPEARMAN_NBEST = True

def run_classifier(train_data, train_labels, test_data, classifier):
    if train_data.shape[1] == 0:
        return np.asarray([0] * train_data.shape[0])
    classifier.fit(train_data, train_labels)
    return classifier.predict(test_data)


def classify(x, x_val, y):
    import sklearn.svm as svm
    predict = run_classifier(x, y, x_val, svm.LinearSVC())
    return predict

data = dump.load_object('data.dump')
data_valid = dump.load_object('data_valid.dump')
labels = dump.load_object('labels.dump')
labels_valid = dump.load_object('labels_valid.dump')


score = metrics.f1_score(labels_valid, classify(data, data_valid, labels))
print(score)
print()
dump.dump_object(score, 'score.dump')

# INFO GAIN
if INFO_GAIN:
    ig = dump.load_object('ig/ig.dump')
    ig_coefs = np.arange(0.1, 0.91, 0.01)
    ig_f1 = []
    ig_n_feat = []
    print('Information Gain: classifying on different coefficients')
    timer.set_new()
    for i in range(len(ig_coefs)):
        frame.progress((i + 1) / len(ig_coefs))
        trimmed_ig = [x for x in ig if x[0] > ig_coefs[i]]
        indexes_ig = [x[1] for x in trimmed_ig]
        ig_data = trim.trim_data(data, indexes_ig)
        ig_data_valid = trim.trim_data(data_valid, indexes_ig)
        ig_f1.append(metrics.f1_score(labels_valid, classify(ig_data, ig_data_valid, labels)))
        ig_n_feat.append(len(indexes_ig))
    print(' DONE in ' + timer.get_diff_str())
    dump.dump_object(ig_coefs, 'ig/svm/coefs.dump')
    dump.dump_object(ig_f1, 'ig/svm/f1.dump')
    dump.dump_object(ig_n_feat, 'ig/svm/feat.dump')

    ig_cls = [(ig_coefs[i], ig_f1[i]) for i in range(len(ig_coefs))]
    ig_coef_max = max(ig_cls, key=lambda x: x[1])[0]
    indexes_ig = [x[1] for x in [y for y in ig if y[0] > ig_coef_max]]  # to eiler's diagram
    dump.dump_object(indexes_ig, 'ig/max/indexes.dump')

# PEARSON
if PEARSON:
    pearson = dump.load_object('pearson/p.dump')
    pearson_coefs = np.arange(0.1, 0.91, 0.01)
    pearson_f1 = []
    pearson_n_feat = []
    print('Pearson: classifying on different coefficients')
    timer.set_new()
    for i in range(len(pearson_coefs)):
        frame.progress((i + 1) / len(pearson_coefs))
        trimmed_pearson = [x for x in pearson if x[0] > pearson_coefs[i]]
        indexes_pearson = [x[1] for x in trimmed_pearson]
        pearson_data = trim.trim_data(data, indexes_pearson)
        pearson_data_valid = trim.trim_data(data_valid, indexes_pearson)
        pearson_f1.append(metrics.f1_score(labels_valid, classify(pearson_data, pearson_data_valid, labels)))
        pearson_n_feat.append(len(indexes_pearson))
    print(' DONE in ' + timer.get_diff_str())
    dump.dump_object(pearson_coefs, 'pearson/svm/coefs.dump')
    dump.dump_object(pearson_f1, 'pearson/svm/f1.dump')
    dump.dump_object(pearson_n_feat, 'pearson/svm/feat.dump')

    pearson_cls = [(pearson_coefs[i], pearson_f1[i]) for i in range(len(pearson_coefs))]
    pearson_coef_max = max(pearson_cls, key=lambda x: x[1])[0]
    indexes_pearson = [x[1] for x in [y for y in pearson if y[0] > pearson_coef_max]]  # to eiler's diagram
    dump.dump_object(indexes_pearson, 'pearson/max/indexes.dump')

# SPEARMAN
if SPEARMAN:
    spearman = dump.load_object('spearman/p.dump')
    spearman_coefs = np.arange(0.1, 0.91, 0.01)
    spearman_f1 = []
    spearman_n_feat = []
    print('Spearman: classifying on different coefficients')
    timer.set_new()
    for i in range(len(spearman_coefs)):
        frame.progress((i + 1) / len(spearman_coefs))
        trimmed_spearman = [x for x in spearman if x[0] > spearman_coefs[i]]
        indexes_spearman = [x[1] for x in trimmed_spearman]
        spearman_data = trim.trim_data(data, indexes_spearman)
        spearman_data_valid = trim.trim_data(data_valid, indexes_spearman)
        spearman_f1.append(metrics.f1_score(labels_valid, classify(spearman_data, spearman_data_valid, labels)))
        spearman_n_feat.append(len(indexes_spearman))
    print(' DONE in ' + timer.get_diff_str())
    dump.dump_object(spearman_coefs, 'spearman/svm/coefs.dump')
    dump.dump_object(spearman_f1, 'spearman/svm/f1.dump')
    dump.dump_object(spearman_n_feat, 'spearman/svm/feat.dump')

    spearman_cls = [(spearman_coefs[i], spearman_f1[i]) for i in range(len(spearman_coefs))]
    spearman_coef_max = max(spearman_cls, key=lambda x: x[1])[0]
    indexes_spearman = [x[1] for x in [y for y in spearman if y[0] > spearman_coef_max]]  # to eiler's diagram
    dump.dump_object(indexes_spearman, 'spearman/max/indexes.dump')


def f1_score(indexes_list):
    cross_data = trim.trim_data(data, indexes_list)
    cross_data_valid = trim.trim_data(data_valid, indexes_list)
    return metrics.f1_score(labels_valid, classify(cross_data, cross_data_valid, labels))



def nbest(metric, folder):
    nbest_coefs = np.arange(500, 4999, 100)
    metric = sorted(metric, key=lambda x: x[0])
    metric_f1 = []
    metric_n_feat = []
    print(folder + ': classifying N BEST')
    timer.set_new()
    for i in range(len(nbest_coefs)):
        frame.progress((i + 1) / len(nbest_coefs))
        indexes_metric = [x[1] for x in metric[-nbest_coefs[i]:]]
        metric_data = trim.trim_data(data, indexes_metric)
        metric_data_valid = trim.trim_data(data_valid, indexes_metric)
        metric_f1.append(metrics.f1_score(labels_valid, classify(metric_data, metric_data_valid, labels)))
        metric_n_feat.append(len(indexes_metric))
    print(' DONE in ' + timer.get_diff_str())
    dump.dump_object(nbest_coefs, folder + '/nbest/svm/coefs.dump')
    dump.dump_object(metric_f1, folder + '/nbest/svm/f1.dump')
    dump.dump_object(metric_n_feat, folder + '/nbest/svm/feat.dump')

    metric_cls = [(nbest_coefs[i], metric_f1[i]) for i in range(len(nbest_coefs))]
    metric_coef_max = max(metric_cls, key=lambda x: x[1])[0]
    indexes_metric = [x[1] for x in metric[-metric_coef_max:]]  # to eiler's diagram
    dump.dump_object(indexes_metric, folder + '/nbest/max/indexes.dump')

if IG_NBEST:
    nbest(dump.load_object('ig/ig.dump'), 'ig')
if PEARSON_NBEST:
    nbest(dump.load_object('pearson/p.dump'), 'pearson')
if SPEARMAN_NBEST:
    nbest(dump.load_object('spearman/p.dump'), 'spearman')



# OVERALL
if OVERALL:
    indexes_ig = set(dump.load_object('ig/max/indexes.dump'))
    indexes_pearson = set(dump.load_object('pearson/max/indexes.dump'))
    indexes_spearman = set(dump.load_object('spearman/max/indexes.dump'))

    indexes_ig_nbest = set(dump.load_object('ig/nbest/max/indexes.dump'))
    indexes_pearson_nbest = set(dump.load_object('pearson/nbest/max/indexes.dump'))
    indexes_spearman_nbest = set(dump.load_object('spearman/nbest/max/indexes.dump'))

    print(f1_score(list(indexes_ig & indexes_pearson)))
    print(f1_score(list(indexes_ig & indexes_spearman)))
    print(f1_score(list(indexes_pearson & indexes_spearman)))
    print(f1_score(list(indexes_ig & indexes_pearson & indexes_spearman)))
    print()
    print(f1_score(list(indexes_ig_nbest & indexes_pearson_nbest)))
    print(f1_score(list(indexes_ig_nbest & indexes_spearman_nbest)))
    print(f1_score(list(indexes_pearson_nbest & indexes_spearman_nbest)))
    print(f1_score(list(indexes_ig_nbest & indexes_pearson_nbest & indexes_spearman_nbest)))