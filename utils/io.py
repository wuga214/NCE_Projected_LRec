import mxnet as mx
from scipy.sparse import save_npz, load_npz
from scipy.sparse import csr_matrix
import numpy as np


def save_csr(matrix, path, name, format="MXNET"):
    print(name+" Shape: {0}".format(matrix.shape))
    if format == "MXNET":
        if not isinstance(matrix, np.ndarray):
            matrix = matrix.todense()
        mx_array = mx.nd.array(matrix)
        mx.nd.save(path+name, mx_array)
    else:
        save_npz(path + name, matrix)


def load_csr(path, name, shape=(1010000, 2262292)):
    #NYZ or Sparse CSV
    if name.endswith('.npz'):
        return load_npz(path+name).tocsr()
    else:
        data = np.genfromtxt(path+name, delimiter=',')
        matrix = csr_matrix((data[:, 2], (data[:, 0], data[:, 1])), shape=shape)
        # create npz for later convenience
        # save_npz(path + "rating.npz", matrix)
        return matrix