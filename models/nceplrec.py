import numpy as np
import scipy.sparse as sparse
from scipy.sparse import vstack, hstack
from scipy.sparse.linalg import inv
from sklearn.utils.extmath import randomized_svd
from utils.progress import WorkSplitter, inhour
import time
from tqdm import tqdm


def get_pmi_matrix(matrix, root):

    rows, cols = matrix.shape
    item_rated = matrix.sum(axis=0)
    # user_rated = matrix.sum(axis=1)
    nnz = matrix.nnz
    pmi_matrix = []
    for i in tqdm(xrange(rows)):
        row_index, col_index = matrix[i].nonzero()
        if len(row_index) > 0:
            # import ipdb; ipdb.set_trace()
            # values = np.asarray(user_rated[i].dot(item_rated)[:, col_index]).flatten()
            values = np.asarray(item_rated[:, col_index]).flatten()
            values = np.maximum(np.log(rows/np.power(values, root)), 0)
            pmi_matrix.append(sparse.coo_matrix((values, (row_index, col_index)), shape=(1, cols)))
        else:
            pmi_matrix.append(sparse.coo_matrix((1, cols)))

    return sparse.vstack(pmi_matrix)


def get_pmi_matrix_gpu(matrix, root):
    import cupy as cp
    rows, cols = matrix.shape
    item_rated = cp.array(matrix.sum(axis=0))
    pmi_matrix = []
    nnz = matrix.nnz
    for i in tqdm(xrange(rows)):
        row_index, col_index = matrix[i].nonzero()
        if len(row_index) > 0:
            values = cp.asarray(item_rated[:, col_index]).flatten()
            values = cp.maximum(cp.log(rows/cp.power(values, root)), 0)
            pmi_matrix.append(sparse.coo_matrix((cp.asnumpy(values), (row_index, col_index)), shape=(1, cols)))
        else:
            pmi_matrix.append(sparse.coo_matrix((1, cols)))
    return sparse.vstack(pmi_matrix)


def nceplrec(matrix_train, embeded_matrix=np.empty((0)), iteration=4,
             lam=80, rank=200, seed=1, root=1.1, gpu_on=True, **unused):
    """
    Function used to achieve generalized projected lrec w/o item-attribute embedding
    :param matrix_train: user-item matrix with shape m*n
    :param embeded_matrix: item-attribute matrix with length n (each row represents one item)
    :param lam: parameter of penalty
    :param k_factor: ratio of the latent dimension/number of items
    :return: prediction in sparse matrix
    """
    progress = WorkSplitter()
    matrix_input = matrix_train
    if embeded_matrix.shape[0] > 0:
        matrix_input = vstack((matrix_input, embeded_matrix.T))

    progress.subsection("Create PMI matrix")
    if gpu_on:
        pmi_matrix = get_pmi_matrix_gpu(matrix_input, root)
    else:
        pmi_matrix = get_pmi_matrix(matrix_input, root)

    progress.subsection("Randomized SVD")
    start_time = time.time()
    P, sigma, Qt = randomized_svd(pmi_matrix,
                                  n_components=rank,
                                  n_iter=iteration,
                                  random_state=seed)
    # Plain
    # RQ = matrix_input.dot(sparse.csc_matrix(Qt).T)

    # sqrt sigma injection
    RQ = matrix_input.dot(sparse.csc_matrix(Qt.T*np.sqrt(sigma)))

    print("Elapsed: {0}".format(inhour(time.time() - start_time)))

    progress.subsection("Closed-Form Linear Optimization")
    start_time = time.time()
    pre_inv = RQ.T.dot(RQ) + lam * sparse.identity(rank, dtype=np.float32)
    inverse = inv(pre_inv)
    Y = inverse.dot(RQ.T).dot(matrix_input)
    print("Elapsed: {0}".format(inhour(time.time() - start_time)))
    return np.array(RQ.todense()), np.array(Y.todense()), None