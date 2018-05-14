import numpy as np


def r_precision(vector_u, matrix_V, vector_train, vector_true, k=500):
    train_index = vector_train.nonzeros[1]
    vector_predict = matrix_V.dot(vector_u).argsort()[-(k+len(train_index)):][::-1]
    vector_predict = np.delete(vector_predict, np.isin(vector_predict, train_index).nonzero()[0])
    vector_true = vector_true.nonzeros[1]
    hits = len(np.isin(vector_true, vector_predict).nonzero()[0])
    return float(hits)/len(vector_true)
