import numpy as np
from scipy.sparse import vstack
from tqdm import tqdm
import torch

from utils.progress import WorkSplitter


def als(matrix_train,
        embeded_matrix=np.empty((0)),
        iteration=4,
        lam=80,
        rank=200,
        alpha=100,
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

    matrix_coo = matrix_train.tocoo()

    cold_rows, cold_cols = get_cold(matrix_coo, m, n)

    np.random.seed(1)
    U = torch.tensor(np.random.normal(0, 0.01, size=(m, rank)).astype(np.float32)).float()
    V = torch.tensor(np.random.normal(0, 0.01, size=(n, rank)).astype(np.float32)).float()

    U[cold_rows] = 0
    V[cold_cols] = 0

    for i in xrange(iteration):
        progress.subsubsection("Iteration: {0}".format(i))
        solve(matrix_input.T, U, V, lam=lam, rank=rank, alpha=alpha)
        solve(matrix_input, V, U, lam=lam, rank=rank, alpha=alpha)

    return U.numpy(), V.numpy().T, None


def get_cold(matrix_coo, m, n):
    warm_rows = np.unique(matrix_coo.row)
    warm_cols = np.unique(matrix_coo.col)

    mask = np.ones(m, np.bool)
    mask[warm_rows] = 0
    cold_rows = np.nonzero(mask)

    mask = np.ones(n, np.bool)
    mask[warm_cols] = 0
    cold_cols = np.nonzero(mask)

    return cold_rows, cold_cols


def solve(R, X, H, lam, rank, alpha):
    """
    Linear function solver, in the form R = XH^T with weighted loss
    """
    HT = torch.transpose(H, 0, 1)
    matrix_A = torch.mm(HT, H) + torch.eye(rank)*lam

    for i in tqdm(xrange(R.shape[1])):
        vector_r = R[:, i]
        vector_x = per_item(vector_r, matrix_A, H, alpha)
        X[i] = vector_x


def per_item(vector_r, matrix_A, matrix_B, alpha):
    vector_r_index = torch.tensor(vector_r.nonzero()[0]).type(torch.long)
    vector_r_small = torch.tensor(vector_r.data).float()
    vector_c_small = alpha * vector_r_small
    matrix_B_small = matrix_B[vector_r_index]
    matrix_BT_small = torch.transpose(matrix_B_small, 0, 1)
    denominator = torch.inverse(matrix_A+torch.mm((torch.mul(matrix_BT_small, vector_c_small)), matrix_B_small))
    return torch.flatten(torch.mv(torch.mm(denominator, matrix_BT_small), torch.mul(vector_c_small, vector_r_small)+vector_r_small))