import mxnet as mx
from scipy.sparse import save_npz


def save_csr(matrix, path, name, format="MXNET"):

    matrix = matrix.todense()
    if format == "MXNET":
        mx_array = mx.nd.array(matrix)
        mx.nd.save(path+name, mx_array)
    else:
        save_npz(path + name, matrix)