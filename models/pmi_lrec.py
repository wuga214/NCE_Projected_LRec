import numpy as np
import scipy.sparse as sparse
from scipy.sparse import vstack, hstack
from scipy.sparse.linalg import inv
from sklearn.utils.extmath import randomized_svd
from utils.progress import WorkSplitter, inhour
import time
from tqdm import tqdm


def get_pmi_matrix(matrix):

    size = matrix.shape[0]
    user_rated = matrix.sum(axis=0)
    item_rated = matrix.sum(axis=1)
    pmi_matrix = sparse.csr_matrix(matrix.shape)
    for i in tqdm(xrange(size)):
        row_index, col_index = matrix[i].nonzero()
        values = np.asarray(item_rated[i].dot(user_rated)[:, col_index]).flatten()

        # PMI
        values = size / values

        # SPPMI
        # values = np.maximum(np.log(size/values)-np.log(5), 0)
        row_index.fill(i)
        pmi_matrix = pmi_matrix + sparse.csr_matrix((values, (row_index, col_index)), shape=matrix.shape)

    return pmi_matrix




def pmi_lrec_items(matrix_train, embeded_matrix=np.empty((0)), iteration=4, lam=80, rank=200, seed=1, **unused):
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
    pmi_matrix = get_pmi_matrix(matrix_input)
    #import ipdb; ipdb.set_trace()

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

    print "Elapsed: {0}".format(inhour(time.time() - start_time))

    progress.subsection("Closed-Form Linear Optimization")
    start_time = time.time()
    pre_inv = RQ.T.dot(RQ) + lam * sparse.identity(rank)
    inverse = inv(pre_inv)
    Y = inverse.dot(RQ.T).dot(matrix_input)
    print "Elapsed: {0}".format(inhour(time.time() - start_time))
    return np.array(RQ.todense()), np.array(Y.todense())