import numpy as np
import scipy.sparse as sparse
from scipy.sparse import vstack
from sklearn.utils.extmath import randomized_svd
from fbpca import pca
from utils.progress import WorkSplitter, inhour
import time


def pure_svd(matrix_train, embeded_matrix=np.empty((0)),
             iteration=4, rank=200, fb=True, **unused):
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
    if fb:
        P, sigma, Qt = pca(matrix_input,
                           k=rank,
                           n_iter=iteration,
                           raw=True)
    else:
        P, sigma, Qt = randomized_svd(matrix_input,
                                      n_components=rank,
                                      n_iter=iteration,
                                      power_iteration_normalizer='QR',
                                      random_state=None)

    RQ = matrix_input * sparse.csr_matrix(Qt).T
    print "Elapsed: {0}".format(inhour(time.time() - start_time))

    return RQ, Qt


# boost pureSVD with multiplying Sigma for multiple times. Let the similarity matrix to be sharp!
def eigen_boosted_pure_svd(matrix_train, embeded_matrix=np.empty((0)),
                           iteration=4, rank=200, fb=True, alpha=1, **unused):
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
    if fb:
        P, sigma, Qt = pca(matrix_input,
                           k=rank,
                           n_iter=iteration,
                           raw=True)
    else:
        P, sigma, Qt = randomized_svd(matrix_input,
                                      n_components=rank,
                                      n_iter=iteration,
                                      power_iteration_normalizer='QR',
                                      random_state=None)

    PS = P*(sigma*alpha)
    print "Elapsed: {0}".format(inhour(time.time() - start_time))

    return PS, Qt