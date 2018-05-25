import scipy.sparse as sparse
import numpy as np
from numpy.random import permutation
from tqdm import tqdm


def split(matrix, ratio=0.3, implicit=True, random=True):
    if implicit:
        matrix[matrix.nonzero()] = 1

    user_num, item_num = matrix.shape

    rtrain = sparse.csr_matrix((user_num, item_num))
    rvalid = sparse.csr_matrix((user_num, item_num))

    for i in tqdm(xrange(user_num)):
        item_indexes = matrix[i].nonzero()[1]
        data = matrix[i].data
        num_nonzeros = len(item_indexes)
        if num_nonzeros >= 1:
            num_valid = int(num_nonzeros*ratio)
            num_train = num_nonzeros - num_valid
            rtrain_row = sparse.csr_matrix((user_num, item_num))
            rvalid_row = sparse.csr_matrix((user_num, item_num))
            if random:
                perm = permutation(num_nonzeros)
                item_indexes = item_indexes[perm]
                data = data[perm]

            rtrain = rtrain + sparse.csr_matrix((data[:num_train], (np.full(num_train, i), item_indexes[:num_train])),
                                                shape=(user_num, item_num))
            rvalid = rvalid + sparse.csr_matrix((data[num_train:], (np.full(num_valid, i), item_indexes[num_train:])),
                                                shape=(user_num, item_num))

    return rtrain, rvalid


