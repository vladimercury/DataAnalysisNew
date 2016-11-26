import util.dump as dump
import warnings
warnings.filterwarnings('ignore')


def run_classifier(x, y, classifier):
    classifier.fit(x, y)
    return classifier.predict(x)


def classify(x, x1, y):
    import sklearn.svm as svm
    import sklearn.naive_bayes as nb
    import sklearn.metrics as met
    classifiers = [
        nb.GaussianNB(),
        nb.MultinomialNB(),
        nb.BernoulliNB(),
        svm.LinearSVC()
    ]
    for classifier in classifiers:
        print(type(classifier))
        print(met.f1_score(y, run_classifier(x, y, classifier)))
        print(met.f1_score(y, run_classifier(x1, y, classifier)))

data = dump.load_object('data.dump')
labels = dump.load_object('labels.dump')
new_data = dump.load_object('newdata.dump')

print(data.shape)
print(new_data.shape)

classify(data, new_data, labels)