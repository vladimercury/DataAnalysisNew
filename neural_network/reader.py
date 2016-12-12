def read_labels(file_name):
    file = open(file_name, 'rb')
    endian = 'big'
    magic_number = int.from_bytes(file.read(4), endian)
    number = int.from_bytes(file.read(4), endian)
    labels = [int.from_bytes(file.read(1), endian) for i in range(number)]
    file.close()
    return (magic_number, number), labels


def read_images(file_name):
    file = open(file_name, 'rb')
    endian = 'big'
    magic_number = int.from_bytes(file.read(4), endian)
    number = int.from_bytes(file.read(4), endian)
    rows = int.from_bytes(file.read(4), endian)
    cols = int.from_bytes(file.read(4), endian)
    images = [file.read(cols * rows) for i in range(number)]
    file.close()
    return (magic_number, number), (rows, cols), images

