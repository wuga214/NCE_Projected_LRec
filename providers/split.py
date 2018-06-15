import scipy.sparse as sparse
import numpy as np
from tqdm import tqdm


def time_ordered_split(rating_matrix, timestamp_matrix, ratio=[0.5, 0.2, 0.3], implicit=True, remove_empty=True):
    if implicit:
        rating_matrix[rating_matrix.nonzero()] = 1

    nonzero_index = None

    if remove_empty:
        # Remove empty columns. record original item index
        nonzero_index = np.unique(rating_matrix.nonzero()[1])
        rating_matrix = rating_matrix[:, nonzero_index]
        timestamp_matrix = timestamp_matrix[:, nonzero_index]

    user_num, item_num = rating_matrix.shape

    rtrain = sparse.csr_matrix((user_num, item_num))
    rvalid = sparse.csr_matrix((user_num, item_num))
    rtest = sparse.csr_matrix((user_num, item_num))

    for i in tqdm(xrange(user_num)):
        item_indexes = rating_matrix[i].nonzero()[1]
        data = rating_matrix[i].data
        timestamp = timestamp_matrix[i].data
        num_nonzeros = len(item_indexes)
        if num_nonzeros >= 1:
            num_test = int(num_nonzeros*ratio[2])
            num_valid = int(num_nonzeros*(ratio[1]+ratio[2]))

            valid_offset = num_nonzeros - num_valid
            test_offset = num_nonzeros - num_test

            argsort = np.argsort(timestamp)
            data = data[argsort]
            item_indexes = item_indexes[argsort]

            rtrain = rtrain + sparse.csr_matrix((data[:valid_offset], (np.full(valid_offset, i),
                                                                       item_indexes[:valid_offset])),
                                                shape=(user_num, item_num))
            rvalid = rvalid + sparse.csr_matrix((data[valid_offset:test_offset],
                                                 (np.full(test_offset-valid_offset, i),
                                                  item_indexes[valid_offset:test_offset])),
                                                shape=(user_num, item_num))
            rtest = rtest + sparse.csr_matrix((data[test_offset:],
                                               (np.full(num_test, i),
                                                item_indexes[test_offset:])),
                                              shape=(user_num, item_num))

    return rtrain, rvalid, rtest, nonzero_index

