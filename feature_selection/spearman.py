def spearman(data, labels, dumped=False):
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
        n = len(x)
        rank_x = np.asarray(stats.rankdata(x, method='max'))
        rank_y = np.asarray(stats.rankdata(y, method='max'))
        sum_d_2 = sum((rank_x - rank_y) ** 2)
        return 1 - 6 * sum_d_2 / (n * (n ** 2 - 1))

    def correlation(x, y):
        from util.frame import progress
        print('Spearman: computing corellation coefficients:')
        feat_len = len(x)
        result = []
        for i in range(feat_len):
            result.append(feature_correlation(x[i], y))
            progress((i + 1) / feat_len)
        print()
        return np.asarray(result)

    features = get_features(data)
    ro = []
    if not dumped:
        ro = correlation(features, labels)
        dump.dump_object(ro, 'spearman/ro.dump')
    else:
        ro = dump.load_object('spearman/ro.dump')
    n = len(labels)
    v = n - 2
    p = []
    for i in range(len(ro)):
        t = ro[i] * math.sqrt(v) / math.sqrt(1 - ro[i] ** 2)
        p.append((stats.t.sf(np.abs(t), v) * 2, i))
    return p
