import util.dump as dump
import util.frame as frame
import feature_selection.trim as trim
import matplotlib.pyplot as pt
import sklearn.metrics as metrics
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def run_classifier(train_data, train_labels, test_data, classifier):
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

ig = dump.load_object('ig/ig.dump')
ig_dumped = True
ig_graph = False

score = metrics.f1_score(labels_valid, classify(data, data_valid, labels))

# INFO GAIN
ig_coefs = np.arange(0.1, 0.91, 0.01)
ig_f1 = []
ig_n_feat = []
if not ig_dumped:
    print('Information Gain: classifying on different coefficients')
    for i in range(len(ig_coefs)):
        frame.progress((i + 1) / len(ig_coefs))
        trimmed_ig = [x for x in ig if x[0] > ig_coefs[i]]
        indexes_ig = [x[1] for x in trimmed_ig]
        ig_data = trim.trim_data(data, indexes_ig)
        ig_data_valid = trim.trim_data(data_valid, indexes_ig)
        ig_f1.append(metrics.f1_score(labels_valid, classify(ig_data, ig_data_valid, labels)))
        ig_n_feat.append(len(indexes_ig))
    print()
    dump.dump_object(ig_coefs, 'ig/svm/coefs.dump')
    dump.dump_object(ig_f1, 'ig/svm/f1.dump')
    dump.dump_object(ig_n_feat, 'ig/svm/feat.dump')
else:
    ig_coefs = dump.load_object('ig/svm/coefs.dump')
    ig_f1 = dump.load_object('ig/svm/f1.dump')
    ig_n_feat = dump.load_object('ig/svm/feat.dump')

if ig_graph:
    pt.title('Information Gain: F1')
    pt.plot(ig_coefs, ig_f1)
    pt.plot(ig_coefs, [score] * len(ig_coefs), color='red')
    pt.figure()
    pt.title('Infromation Gain: N Features')
    pt.plot(ig_coefs, ig_n_feat)
    pt.plot(ig_coefs, [len(data[0])] * len(ig_coefs), color='red')
    pt.show()

ig_cls = [(ig_coefs[i], ig_f1[i]) for i in range(len(ig_coefs))]
coef_max = max(ig_cls, key=lambda x: x[1])[0]
indexes_ig = [x[1] for x in [y for y in ig if y[0] > coef_max]]  # to eiler's diagram
