def trim_data(data, indexes):
    import numpy as np
    return np.asarray([np.asarray(x)[indexes] for x in data])