import numpy as np
import scipy.sparse as sparse
from scipy.sparse import vstack, hstack
from scipy.sparse.linalg import inv
from sklearn.utils.extmath import randomized_svd
from utils.progress import WorkSplitter, inhour
import time


def embedded_lrec_items(matrix_train, embeded_matrix=np.empty((0)), iteration=4, lam=80, rank=200, **unused):
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

    progress.section("Randomized SVD")
    start_time = time.time()
    P, sigma, Qt = randomized_svd(matrix_input,
                                  n_components=rank,
                                  n_iter=iteration,
                                  random_state=42)
    RQ = matrix_input * sparse.csc_matrix(Qt).T
    print "Elapsed: {0}".format(inhour(time.time() - start_time))

    progress.section("Closed-Form Linear Optimization")
    start_time = time.time()
    pre_inv = RQ.T.dot(RQ) + lam * sparse.identity(rank)
    inverse = inv(pre_inv)
    Y = inverse.dot(RQ.T).dot(matrix_input)
    print "Elapsed: {0}".format(inhour(time.time() - start_time))
    return np.array(RQ.todense()), np.array(Y.todense())


# def embedded_lrec_users(matrix_train, embeded_matrix=np.empty((0)), iteration=4, lam=80, rank=200):
#     """
#     Function used to achieve generalized projected lrec w/o item-attribute embedding
#     :param matrix_train: user-item matrix with shape m*n
#     :param embeded_matrix: user-attribute matrix with length m (each row represents one user)
#     :param lam: parameter of penalty
#     :param k_factor: ratio of the latent dimension/number of items
#     :return: prediction in sparse matrix
#     """
#     progress = WorkSplitter()
#     matrix_nonzero = matrix_train.nonzero()
#     matrix_input = matrix_train
#     if embeded_matrix.shape[0] > 0:
#         matrix_input = hstack((matrix_input, embeded_matrix))
#     progress.section("Randomized SVD")
#     start_time = time.time()
#     P, sigma, Qt = randomized_svd(matrix_input,
#                                   n_components=rank,
#                                   n_iter=iteration,
#                                   random_state=None)
#     print "Elapsed: {0}".format(inhour(time.time() - start_time))
#     PtR = sparse.csr_matrix(P).T.dot(matrix_input)
#
#     progress.section("Closed-Form Linear Optimization")
#     start_time = time.time()
#     pre_inv = PtR.dot(PtR.T) + lam * sparse.identity(rank)
#     inverse = sparse.csr_matrix(inv(pre_inv))
#     Y = matrix_input.dot(PtR.T).dot(inverse)
#     print "Elapsed: {0}".format(inhour(time.time() - start_time))
#     return PtR, Y
