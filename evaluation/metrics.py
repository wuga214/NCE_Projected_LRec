import numpy as np
import scipy.sparse as sparse


def sub_routine(vector_u, matrix_V, vector_train, vector_true, k=500):
    train_index = vector_train.nonzero()[1]
    vector_predict = matrix_V.dot(vector_u).argsort()[-(k+len(train_index)):][::-1]
    vector_predict = np.delete(vector_predict, np.isin(vector_predict, train_index).nonzero()[0])
    vector_true_dense = vector_true.nonzero()[1]
    #import ipdb;ipdb.set_trace()
    return vector_predict[:k], vector_true_dense, np.isin(vector_predict[:k], vector_true_dense)


def r_precision(vector_true_dense, hits, **unused):
    hits = len(hits.nonzero()[0])
    return float(hits)/len(vector_true_dense)


def _dcg_support(size):
    arr = np.arange(1, size+1)+1
    return 1./np.log2(arr)


def ndcg(vector_true_dense, vector_predict, hits):
    idcg = np.sum(_dcg_support(len(vector_true_dense)))
    dcg_base = _dcg_support(len(vector_predict))
    #import ipdb; ipdb.set_trace()
    dcg_base[np.logical_not(hits)] = 0
    dcg = np.sum(dcg_base)
    return dcg/idcg

def click(hits, **unused):
    first_hit = next((i for i, x in enumerate(hits) if x), None)
    return first_hit/10


vector_u = np.random.rand(10)
matrix_V = sparse.rand(5000, 10).tocsr()
vector_train = sparse.rand(1, 1000).tocsr()
vector_true = sparse.rand(1, 1000).tocsr()


vector_predict, vector_true_dense, hits = sub_routine(vector_u, matrix_V, vector_train, vector_true, k=500)
print r_precision(vector_true_dense=vector_true_dense, hits=hits)
print ndcg(vector_true_dense=vector_true_dense, vector_predict=vector_predict, hits=hits)
print click(hits=hits)