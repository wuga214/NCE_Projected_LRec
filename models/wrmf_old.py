import numpy as np
from scipy.sparse import vstack
import scipy.sparse as sparse
from scipy.linalg import inv as cpu_inv
from tqdm import tqdm

from utils.progress import WorkSplitter


def per_item_cpu(vector_r, matrix_A, matrix_B, matrix_BT, alpha):
    vector_r_index = vector_r.nonzero()[0]
    vector_r_small = vector_r.data
    vector_c_small = alpha * vector_r_small
    matrix_B_small = np.take(matrix_B, vector_r_index, axis=0)
    matrix_BT_small = np.take(matrix_BT, vector_r_index, axis=1)
    denominator = cpu_inv(matrix_A+(matrix_BT_small*vector_c_small).dot(matrix_B_small))
    return (denominator.dot(matrix_BT_small)).dot((vector_c_small*vector_r_small+vector_r_small).T).flatten()


def per_item_gpu(vector_r, matrix_A, matrix_B, matrix_BT, alpha):
    import cupy as cp
    from cupy.linalg import inv as gpu_inv
    vector_r_index = vector_r.nonzero()[0]
    vector_r_small = cp.array(vector_r.data)
    vector_c_small = alpha * vector_r_small
    matrix_B_small = cp.take(matrix_B, vector_r_index, axis=0)
    matrix_BT_small = cp.take(matrix_BT, vector_r_index, axis=1)

    denominator = gpu_inv(matrix_A+(matrix_BT_small*vector_c_small).dot(matrix_B_small))
    return (denominator.dot(matrix_BT_small)).dot((vector_c_small*vector_r_small+vector_r_small)).flatten()


def solve(R, X, H, lam, rank, alpha, gpu):
    """
    Linear function solver, in the form R = XH^T with weighted loss
    """

    if gpu:
        import cupy as cp
        H = cp.array(H)
        HT = H.T
        matrix_A = HT.dot(H) + cp.array((lam * sparse.identity(rank, dtype=np.float32)).toarray())

        for i in tqdm(xrange(R.shape[1])):
            vector_r = R[:, i]
            vector_x = per_item_gpu(vector_r, matrix_A, H, HT, alpha)
            y_i_gpu = cp.asnumpy(vector_x)
            y_i_cpu = np.copy(y_i_gpu)
            X[i] = y_i_cpu


    else:
        HT = H.T
        matrix_A = HT.dot(H) + (lam * sparse.identity(rank, dtype=np.float32)).toarray()

        for i in tqdm(xrange(R.shape[1])):
            vector_r = R[:, i]
            vector_x = per_item_cpu(vector_r, matrix_A, H, HT, alpha)
            y_i_cpu = vector_x
            X[i] = y_i_cpu

def get_cold(matrix_csr):
    matrix_coo = matrix_csr.tocoo()
    m, n = matrix_csr.shape
    warm_rows = np.unique(matrix_coo.row)
    warm_cols = np.unique(matrix_coo.col)

    mask = np.ones(m, np.bool)
    mask[warm_rows] = 0
    cold_rows = np.nonzero(mask)

    mask = np.ones(n, np.bool)
    mask[warm_cols] = 0
    cold_cols = np.nonzero(mask)

    return cold_rows, cold_cols


def als(matrix_train,
        embeded_matrix=np.empty((0)),
        iteration=4,
        lam=80,
        rank=200,
        alpha=100,
        gpu_on=True,
        seed=1,
        **unused):
    """
    :param matrix_train: rating matrix
    :param embeded_matrix: item or user embedding matrix(side info)
    :param iteration: number of alternative solving
    :param lam: regularization parameter
    :param rank: SVD top K eigenvalue ranks
    :param alpha: re-weighting parameter
    :param gpu: GPU computation or CPU computation. GPU usually does 2X speed of CPU
    :param seed: Random initialization seed
    :return:
    """

    progress = WorkSplitter()
    progress.subsection("Alternative Item-wised Optimization")
    matrix_input = matrix_train
    if embeded_matrix.shape[0] > 0:
        matrix_input = vstack((matrix_input, embeded_matrix.T))

    m, n =matrix_train.shape

    np.random.seed(1)
    U = np.random.normal(0, 0.01, size=(m, rank)).astype(np.float32)
    V = np.random.normal(0, 0.01, size=(n, rank)).astype(np.float32)

    cold_rows, cold_cols = get_cold(matrix_train)
    U[cold_rows] = 0
    V[cold_cols] = 0

    for i in xrange(iteration):
        progress.subsubsection("Iteration: {0}".format(i))
        solve(matrix_input.T, U, V, lam=lam, rank=rank, alpha=alpha, gpu=gpu_on)
        solve(matrix_input, V, U, lam=lam, rank=rank, alpha=alpha, gpu=gpu_on)

    return U, V.T, None
