import mxnet as mx
from scipy.sparse import save_npz, load_npz
from scipy.sparse import csr_matrix
from numpy import genfromtxt


def save_csr(matrix, path, name, format="MXNET"):
    print(name+" Shape: {0}".format(matrix.shape))
    if format == "MXNET":
        matrix = matrix.todense()
        mx_array = mx.nd.array(matrix)
        mx.nd.save(path+name, mx_array)
    else:
        save_npz(path + name, matrix)


def load_csr(path, name, format="NYZ", shape=(1010000, 2262292)):
    #NYZ or Sparse CSV
    if format == "NYZ":
        return load_npz(path+name)
    else:
        data = genfromtxt(path+name, delimiter=',')
        return csr_matrix((data[:, 2], (data[:, 0], data[:, 1])), shape=shape)