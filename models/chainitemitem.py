import numpy as np
import scipy.sparse as sparse
from scipy.sparse import vstack
from sklearn.utils.extmath import randomized_svd
from fbpca import pca
from utils.progress import WorkSplitter, inhour
import time


def chain_item_item(matrix_train, embeded_matrix=np.empty((0)),
                    iteration=7, rank=200, fb=True, seed=1, chain=1, **unused):

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

    RQ = matrix_input.dot(sparse.csc_matrix(Qt).T).toarray()
    PS = P*sigma
    SPPS = PS.T.dot(PS)

    HRQ = RQ.dot(SPPS)

    if chain > 1:
        QTQ = Qt.dot(Qt.T)

    for i in range(1, chain):
        HRQ = HRQ.dot(QTQ).dot(SPPS)

    print("Elapsed: {0}".format(inhour(time.time() - start_time)))

    return HRQ, Qt
