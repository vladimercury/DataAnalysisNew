import feature_selection.reader as reader
import feature_selection.information_gain as info_gain
import feature_selection.trim as trim

data, labels = reader.read_data('data/arcene_train')
ig = info_gain.information_gain(data, labels, dumped=True)
trimmed_ig = [x for x in ig if x[0] > 0.9]
indexes = [x[1] for x in trimmed_ig]
new_data = trim.trim_data(data, indexes)