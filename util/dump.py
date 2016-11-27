def dump_object(obj, file_name):
    from pickle import dump
    import os
    path = 'dump/' + file_name
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    file = open(path, 'wb')
    dump(obj, file)
    file.close()


def load_object(file_name):
    from pickle import load
    file = open('dump/' + file_name, 'rb')
    entry = load(file)
    file.close()
    return entry
