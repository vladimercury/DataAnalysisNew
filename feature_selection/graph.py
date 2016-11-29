import util.dump as dump
import matplotlib.pyplot as pt

INFO_GAIN = True
PEARSON = True

VENN = True

score = dump.load_object('score.dump')

# IG
if INFO_GAIN:
    ig_coefs = dump.load_object('ig/svm/coefs.dump')
    ig_f1 = dump.load_object('ig/svm/f1.dump')
    ig_n_feat = dump.load_object('ig/svm/feat.dump')

    pt.title('Information Gain: F1')
    pt.plot(ig_coefs, ig_f1)
    pt.plot(ig_coefs, [score] * len(ig_coefs), color='red')
    pt.figure()
    pt.title('Infromation Gain: N Features')
    pt.plot(ig_coefs, ig_n_feat)
    pt.show()

# PEARSON
if PEARSON:
    pearson_coefs = dump.load_object('pearson/svm/coefs.dump')
    pearson_f1 = dump.load_object('pearson/svm/f1.dump')
    pearson_n_feat = dump.load_object('pearson/svm/feat.dump')

    pt.title('Pearson: F1')
    pt.plot(pearson_coefs, pearson_f1)
    pt.plot(pearson_coefs, [score] * len(pearson_coefs), color='red')
    pt.figure()
    pt.title('Pearson: N Features')
    pt.plot(pearson_coefs, pearson_n_feat)
    pt.show()

# VENN
if VENN:
    indexes_ig = set(dump.load_object('ig/max/indexes.dump'))
    indexes_pearson = set(dump.load_object('pearson/max/indexes.dump'))
    import matplotlib_venn as venn
    venn.venn2([indexes_ig, indexes_pearson], ('IG', 'Pearson'))
    pt.show()