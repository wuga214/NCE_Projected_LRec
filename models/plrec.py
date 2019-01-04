import numpy as np
import scipy.sparse as sparse
from scipy.sparse import vstack, hstack
from scipy.sparse.linalg import inv
from sklearn.utils.extmath import randomized_svd
from utils.progress import WorkSplitter, inhour
import time


def plrec(matrix_train, embeded_matrix=np.empty((0)), iteration=4, lam=80, rank=200, seed=1, **unused):
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

    progress.subsection("Randomized SVD")
    start_time = time.time()
    P, sigma, Qt = randomized_svd(matrix_input,
                                  n_components=rank,
                                  n_iter=iteration,
                                  random_state=seed)

    RQ = matrix_input.dot(sparse.csc_matrix(Qt.T*np.sqrt(sigma)))

    print("Elapsed: {0}".format(inhour(time.time() - start_time)))

    progress.subsection("Closed-Form Linear Optimization")
    start_time = time.time()
    pre_inv = RQ.T.dot(RQ) + lam * sparse.identity(rank, dtype=np.float32)
    inverse = inv(pre_inv)
    Y = inverse.dot(RQ.T).dot(matrix_input)
    print("Elapsed: {0}".format(inhour(time.time() - start_time)))
    return np.array(RQ.todense()), np.array(Y.todense()), None