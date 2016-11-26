def read_data(prefix):
    data_file_content = [x for x in open(prefix + '.data').read().splitlines()]
    labels_file_content = [x for x in open(prefix + '.labels').read().splitlines()]
    data = [[int(y) for y in x.split()] for x in data_file_content]
    labels = [int(x) for x in labels_file_content]
    labels = [0 if x < 0 else x for x in labels]
    return data, labels
