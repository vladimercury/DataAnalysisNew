import neural_network.reader as reader

train_labels = reader.read_labels('mnist/train-labels-idx1-ubyte')
train_images = reader.read_images('mnist/train-images-idx3-ubyte')
print(train_images[0])
print(train_labels[0])

