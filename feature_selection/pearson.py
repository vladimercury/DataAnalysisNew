def pearson(data, labels, dumped=False):
    import numpy as np
    import util.dump as dump
    import math
    import warnings
    import scipy.stats as stats
    warnings.filterwarnings('ignore')

    def get_features(data_set):
        n = len(data[0])
        return [[i[j] for i in data_set] for j in range(n)]

    def feature_correlation(x, y):
        n = range(len(x))
        x_avg = sum(x) / len(x)
        y_avg = sum(y) / len(y)
        cov = sum([(x[i] - x_avg) * (y[i] - y_avg) for i in n])
        x_dev = math.sqrt(sum([(x[i] - x_avg) ** 2 for i in n]))
        y_dev = math.sqrt(sum([(y[i] - y_avg) ** 2 for i in n]))
        return cov / (x_dev * y_dev)

    def correlation(x, y):
        from util.frame import progress
        print('Pearson: computing corellation coefficients:')
        feat_len = len(x)
        result = []
        for i in range(feat_len):
            result.append(feature_correlation(x[i], y))
            if i % 10 == 0:
                progress((i + 1) / feat_len)
        progress(1)
        return np.asarray(result)

    features = get_features(data)
    ro = []
    if not dumped:
        ro = correlation(features, labels)
        dump.dump_object(ro, 'pearson/ro.dump')
    else:
        ro = dump.load_object('pearson/ro.dump')
    v = len(labels) - 2
    p = []
    for i in range(len(ro)):
        t = ro[i] * math.sqrt(v) / math.sqrt(1 - ro[i] ** 2)
        p.append((stats.t.sf(np.abs(t), v) * 2, i))
    return p
