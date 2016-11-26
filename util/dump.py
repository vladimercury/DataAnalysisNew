def dump_object(obj, file_name):
    from pickle import dump
    file = open('dump/' + file_name, 'wb')
    dump(obj, file)
    file.close()


def load_object(file_name):
    from pickle import load
    file = open('dump/' + file_name, 'rb')
    entry = load(file)
    file.close()
    return entry
