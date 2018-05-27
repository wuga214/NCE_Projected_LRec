import mxnet as mx
from scipy.sparse import save_npz, load_npz
from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd


def save_mxnet(matrix, path, name):
    if not isinstance(matrix, np.ndarray):
        matrix = matrix.todense()
    mx_array = mx.nd.array(matrix)
    mx.nd.save(path + name, mx_array)


def save_dataframe_csv(df, path, name):
    df.to_csv(path+name)


def save_datafram_latex(df, path, model):
    with open('{0}{1}_parameter_tuning.tex'.format(path, model), 'w') as handle:
        handle.write(df.to_latex(index=False))


def save_numpy(matrix, path, model):
    save_npz('{0}{1}_parameter_tuning.csv'.format(path, model), matrix)


def load_numpy(path, name):
    return load_npz(path+name).tocsr()


def load_pandas(path, name, row_name='userId', col_name='movieId',
                value_name='rating', shape=(138494, 131263), sep=','):
    df = pd.read_csv(path + name, sep=sep)
    rows = df[row_name]
    cols = df[col_name]
    values = df[value_name]
    return csr_matrix((values,(rows, cols)), shape=shape)


def load_csv(path, name, shape=(1010000, 2262292)):
    data = np.genfromtxt(path + name, delimiter=',')
    matrix = csr_matrix((data[:, 2], (data[:, 0], data[:, 1])), shape=shape)
    # create npz for later convenience
    save_npz(path + "rating.npz", matrix)
    return matrix