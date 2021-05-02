def unbatch(x):
    return x.reshape((-1,) + x.shape[2:])


def batch(x, batchsize=None):
    # batchsize=None -> add just a dummy batch dimension, same as np.expand_dims(x, 0)
    n = x.shape[0]
    if batchsize is None:
        batchsize = n
    else:
        assert (n % batchsize) == 0
    return x.reshape((n // batchsize, batchsize) + x.shape[1:])


def rebatch(x, batchsize):
    return batch(unbatch(x), batchsize)
