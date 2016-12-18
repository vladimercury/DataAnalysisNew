import feature_selection.reader as reader
import feature_selection.information_gain as info_gain
import feature_selection.pearson as pearson
import feature_selection.spearman as spearman
import util.dump as dump
import numpy as np
from util.timer import Timer

data, labels = reader.read_data('data/arcene_train')
data, labels = np.asarray(data), np.asarray(labels)

# valid_data, valid_labels = reader.read_data('data/arcene_valid')
# valid_data, valid_labels = np.asarray(valid_data), np.asarray(valid_labels)

# dump.dump_object(data, 'data.dump')
# dump.dump_object(valid_data, 'data_valid.dump')
# dump.dump_object(labels, 'labels.dump')
# dump.dump_object(valid_labels, 'labels_valid.dump')

# IG
timer = Timer()
ig = info_gain.information_gain(data, labels, dumped=False)
print('DONE in ' + timer.get_diff_str())
# dump.dump_object(ig, 'ig/ig.dump')

# Pearson
timer.set_new()
p = pearson.pearson(data, labels, dumped=False)
print('DONE in ' + timer.get_diff_str())
# dump.dump_object(p, 'pearson/p.dump')

# Spearman
timer.set_new()
s = spearman.spearman(data, labels, dumped=False)
print('DONE in ' + timer.get_diff_str())
# dump.dump_object(s, 'spearman/p.dump')