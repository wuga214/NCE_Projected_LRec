import scipy.sparse as sparse
import numpy as np


def get_popular_predictions(predict_matrix, shape):

    predict = sparse.csr_matrix(shape)

    k = predict_matrix.shape[1]

    for i, user in enumerate(predict_matrix):
        predict = predict + sparse.csr_matrix((np.ones(k), (np.full(k, i), user)), shape)

    return np.sort(np.sum(predict, axis = 0))




# def popular_overlapping(train_matrix, predict_matrix, names):
#
