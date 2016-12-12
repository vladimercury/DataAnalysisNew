import util.dump as dump
import matplotlib.pyplot as pt

INFO_GAIN = True
PEARSON = True
SPEARMAN = True

IG_NBEST = True
PEARSON_NBEST = True
SPEARMAN_NBEST = True

VENN = True
VENN_NBEST = True

score = dump.load_object('score.dump')


def draw_plot(metric):
    metric_coefs = dump.load_object(metric + '/svm/coefs.dump')
    metric_f1 = dump.load_object(metric + '/svm/f1.dump')
    metric_n_feat = dump.load_object(metric +'/svm/feat.dump')

    pt.title(metric + ': F1')
    pt.plot(metric_coefs, metric_f1)
    pt.plot(metric_coefs, [score] * len(metric_coefs), color='red')
    pt.figure()
    pt.title(metric + ': N Features')
    pt.plot(metric_coefs, metric_n_feat)
    pt.show()


# IG
if INFO_GAIN:
    draw_plot('ig')
if PEARSON:
    draw_plot('pearson')
if SPEARMAN:
    draw_plot('spearman')
if IG_NBEST:
    draw_plot('ig/nbest')
if PEARSON_NBEST:
    draw_plot('pearson/nbest')
if SPEARMAN_NBEST:
    draw_plot('spearman/nbest')


import matplotlib_venn as venn
# VENN
if VENN:
    indexes_ig = set(dump.load_object('ig/max/indexes.dump'))
    indexes_pearson = set(dump.load_object('pearson/max/indexes.dump'))
    indexes_spearman = set(dump.load_object('spearman/max/indexes.dump'))
    venn.venn3([indexes_ig, indexes_pearson, indexes_spearman], ('IG', 'Pearson', 'Spearman'))
    pt.show()
if VENN_NBEST:
    indexes_ig = set(dump.load_object('ig/nbest/max/indexes.dump'))
    indexes_pearson = set(dump.load_object('pearson/nbest/max/indexes.dump'))
    indexes_spearman = set(dump.load_object('spearman/nbest/max/indexes.dump'))
    venn.venn3([indexes_ig, indexes_pearson, indexes_spearman], ('IG', 'Pearson', 'Spearman'))
    pt.title('N BEST')
    pt.show()