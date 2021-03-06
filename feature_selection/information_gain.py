def information_gain(data, labels, dumped=False):
    import numpy as np

    def get_features(data_set):
        n = len(data[0])
        return [[i[j] for i in data_set] for j in range(n)]

    def possibilities(feature):
        counts = np.bincount(feature)
        return np.asarray(counts[np.nonzero(counts)] / float(len(feature)))

    def entropy(feature):
        p = possibilities(feature)
        return -np.sum(p * np.log2(p))

    def spec_cond_entropy(x, y, xi):
        new_y = [y[i] for i in range(len(y)) if x[i] == xi]
        return entropy(new_y)

    def cond_entropy(x, y):
        p = possibilities(x)
        return sum([p[xi] * spec_cond_entropy(x, y, xi) for xi in range(len(p))])

    def cond_entropy_full(x, y):
        from util.frame import progress
        print('Information gain: computing conditional entropy:')
        feat_len = len(x)
        result = []
        for i in range(feat_len):
            result.append(cond_entropy(x[i], y))
            if i % 10 == 0:
                progress((i + 1) / feat_len)
        progress(1)
        return np.asarray(result)

    import util.dump as dump
    features = get_features(data)
    h_y_x = []
    if not dumped:
        h_y_x = cond_entropy_full(features, labels)
        dump.dump_object(h_y_x, 'ig/hyx.dump')
    else:
        h_y_x = dump.load_object('ig/hyx.dump')
    info_gain = entropy(labels) - h_y_x
    result = [(info_gain[i], i) for i in range(len(info_gain))]
    return result
