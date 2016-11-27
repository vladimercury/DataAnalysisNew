import feature_selection.reader as reader
import feature_selection.information_gain as info_gain
import feature_selection.trim as trim
import util.dump as dump
import numpy as np

data, labels = reader.read_data('data/arcene_train')
valid_data, valid_labels = reader.read_data('data/arcene_valid')
data, labels = np.asarray(data), np.asarray(labels)
valid_data, valid_labels = np.asarray(valid_data), np.asarray(valid_labels)


# dump.dump_object(data, 'data.dump')
# dump.dump_object(valid_data, 'data_valid.dump')
# dump.dump_object(labels, 'labels.dump')
# dump.dump_object(valid_labels, 'labels_valid.dump')

# IG
ig = info_gain.information_gain(data, labels, dumped=True)
dump.dump_object(ig, 'ig/ig.dump')