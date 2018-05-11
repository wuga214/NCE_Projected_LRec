import numpy as np
import scipy.sparse as sparse
from scipy.sparse import vstack, hstack
from scipy.linalg import inv
from sklearn.utils.extmath import randomized_svd
from utils.progress import WorkSplitter, inhour
from tqdm import tqdm
import time


def per_item(matrix_A, matrix_B, matrix_BT, vector_c, vector_r):
    denominator = inv(matrix_A+(matrix_BT*vector_c).dot(matrix_B))
    return (denominator.dot(matrix_BT))*(vector_c*vector_r+vector_r)



def weighted_lrec_items(matrix_train, embeded_matrix=np.empty((0)), iteration=4, lam=80, rank=200, alpha=100):
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
    P, sigma, Qt = randomized_svd(matrix_input, n_components=rank, n_iter=iteration, random_state=None)
    print "Elapsed: {0}".format(inhour(time.time() - start_time))

    start_time = time.time()
    progress.section("Create Cacheable Matrices")
    matrix_A = sparse.diags(sigma*sigma-lam)#Sigma.T.dot(Sigma) - lam*sparse.identity(rank)
    matrix_B = P*sigma
    matrix_BT = matrix_B.T
    print "Elapsed: {0}".format(inhour(time.time() - start_time))


    progress.section("Item-wised Optimization")
    start_time = time.time()

    m, n = matrix_train.shape

    for i in tqdm(range(1)): #change back to n!!!
        vector_r = matrix_train[:, i].toarray().ravel()
        vector_c = alpha*vector_r
        vector_y = per_item(matrix_A, matrix_B, matrix_BT, vector_c, vector_r)
        print(vector_y.shape)
    print "Elapsed: {0}".format(inhour(time.time() - start_time))