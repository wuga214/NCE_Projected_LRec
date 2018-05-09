import mxnet as mx
from scipy.sparse import save_npz, load_npz


def save_csr(matrix, path, name, format="MXNET"):
    print(name+" Shape: {0}".format(matrix.shape))
    matrix = matrix.todense()
    if format == "MXNET":
        mx_array = mx.nd.array(matrix)
        mx.nd.save(path+name, mx_array)
    else:
        save_npz(path + name, matrix)


def load_csr(path, name, format="MXNET"):
    if format == "MXNET":
        return mx.nd.load(path+name)
    else:
        return load_npz(path+name)

# def save_csr(matrix, path, name, format="MXNET"):
#     print(name+" Shape: {0}".format(matrix.shape))
#     matrix = matrix.todense()
#     if format == "MXNET":
#         SPLIT = 100
#         _, dim = matrix.shape
#         batch_size = dim/SPLIT
#         mx_array = mx.nd.array(matrix)
#         map = dict()
#         for i in xrange(batch_size):
#             map[str(i)] = mx_array[:, SPLIT*i:SPLIT*(i+1)]
#
#         mx.nd.save(path+name, map)
#     else:
#         save_npz(path + name, matrix)