import scipy.sparse as sparse
import numpy as np


def get_popular_predictions(predict_matrix, shape):

    predict = sparse.csr_matrix(shape)

    k = predict_matrix.shape[1]

    for i, user in enumerate(predict_matrix):
        predict = predict + sparse.csr_matrix((np.ones(k), (np.full(k, i), user)), shape)

    item_popularity = np.sum(predict, axis=0)

    item_popularity_rated = np.array(item_popularity/np.sum(item_popularity)).flatten()

    order = np.argsort(item_popularity_rated)
    return order, item_popularity_rated


def popular_overlapping(train_matrix, predict_matrix, k=20):
    train_popularity = np.sum(train_matrix, axis=0)
    train_popularity_rated = np.array(train_popularity/np.sum(train_popularity)).flatten()
    order_train = np.argsort(train_popularity_rated)
    order_predict, predict_popularity_rated = get_popular_predictions(predict_matrix, train_matrix.shape)
    import ipdb; ipdb.set_trace()

