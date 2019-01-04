import numpy as np
import scipy.sparse as sparse
from scipy.sparse import vstack
from sklearn.utils.extmath import randomized_svd
from fbpca import pca
from utils.progress import WorkSplitter, inhour
import time


def puresvd(matrix_train, embeded_matrix=np.empty((0)),
             iteration=4, rank=200, fb=False, seed=1, **unused):
    """
    PureSVD algorithm
    :param matrix_train: rating matrix
    :param embeded_matrix: item or user embedding matrix(side info)
    :param iteration: number of random SVD iterations
    :param rank: SVD top K eigenvalue ranks
    :param fb: facebook package or sklearn package. boolean
    :param seed: Random initialization seed
    :param unused: args that not applicable for this algorithm
    :return:
    """
    progress = WorkSplitter()
    matrix_input = matrix_train
    if embeded_matrix.shape[0] > 0:
        matrix_input = vstack((matrix_input, embeded_matrix.T))

    progress.subsection("Randomized SVD")
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
                                      random_state=seed)

    RQ = matrix_input.dot(sparse.csc_matrix(Qt).T)
    print("Elapsed: {0}".format(inhour(time.time() - start_time)))

    return np.array(RQ.todense()), Qt, None