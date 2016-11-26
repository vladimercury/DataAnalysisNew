import feature_selection.reader as reader
import feature_selection.information_gain as info_gain
import feature_selection.trim as trim
import util.dump as dump
import numpy as np

data, labels = reader.read_data('data/arcene_train')
data, labels = np.asarray(data), np.asarray(labels)
ig = info_gain.information_gain(data, labels, dumped=True)
trimmed_ig = [x for x in ig if x[0] < 0.3]
indexes = [x[1] for x in trimmed_ig]
new_data = trim.trim_data(data, indexes)
dump.dump_object(data, 'data.dump')
dump.dump_object(labels, 'labels.dump')
dump.dump_object(new_data, 'newdata.dump')
