import numpy as np
import scipy.sparse as sparse


def sub_routine(vector_u, matrix_V, vector_train, vector_true, k=500):
    train_index = vector_train.nonzero()[1]
    vector_predict = matrix_V.dot(vector_u).argsort()[-(k+len(train_index)):][::-1]
    vector_predict = np.delete(vector_predict, np.isin(vector_predict, train_index).nonzero()[0])
    vector_true = vector_true.nonzero()[1]
    return vector_predict, vector_true


def r_precision(vector_true_dense, vector_predict):
    hits = len(np.isin(vector_true_dense, vector_predict).nonzero()[0])
    return float(hits)/len(vector_true_dense)


def ndcg(vector_true_dense, vector_predict):
    pass

vector_u = np.random.rand(10)
matrix_V = sparse.rand(5000, 10).tocsr()
vector_train = sparse.rand(1, 1000).tocsr()
vector_true = sparse.rand(1, 1000).tocsr()


vector_true_dense, vector_predict = sub_routine(vector_u, matrix_V, vector_train, vector_true, k=500)
print r_precision(vector_true_dense, vector_predict)