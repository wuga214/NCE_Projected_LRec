import numpy as np
from tqdm import tqdm


def r_precision(vector_true_dense, hits, **unused):
    hits = len(hits.nonzero()[0])
    return float(hits)/len(vector_true_dense)


def _dcg_support(size):
    arr = np.arange(1, size+1)+1
    return 1./np.log2(arr)


def ndcg(vector_true_dense, vector_predict, hits):
    idcg = np.sum(_dcg_support(len(vector_true_dense)))
    dcg_base = _dcg_support(len(vector_predict))
    dcg_base[np.logical_not(hits)] = 0
    dcg = np.sum(dcg_base)
    return dcg/idcg


def click(hits, **unused):
    first_hit = next((i for i, x in enumerate(hits) if x), None)
    if first_hit is None:
        return 5
    else:
        return first_hit/10


def evaluate(matrix_Predict, matrix_Test, metric_names, atK):
    """

    :param matrix_U: Latent representations of users, for LRecs it is RQ, for ALSs it is U
    :param matrix_V: Latent representations of items, for LRecs it is Q, for ALSs it is V
    :param matrix_Train: Rating matrix for training, features.
    :param matrix_Test: Rating matrix for evaluation, true labels.
    :param k: Top K retrieval
    :param metric_names: Evaluation metrics
    :return:
    """
    metrics = {
        "R-Precision": r_precision,
        "NDCG": ndcg,
        "Clicks": click
    }

    output = dict()

    for k in atK:
        results = {name: [] for name in metric_names}

        topK_Predict = matrix_Predict[:, :k]

        for user_index in tqdm(range(topK_Predict.shape[0])):
            vector_predict = topK_Predict[user_index]
            if len(vector_predict.nonzero()[0]) > 0:
                vector_true = matrix_Test[user_index]
                vector_true_dense = vector_true.nonzero()[1]
                hits = np.isin(vector_predict, vector_true_dense)

                if vector_true_dense.size > 0:
                    for name in metric_names:
                        results[name] = metrics[name](vector_true_dense=vector_true_dense,
                                                      vector_predict=vector_predict,
                                                      hits=hits)

        results_summary = dict()
        for name in metric_names:
            results_summary[name] = np.average(results[name])

        output[str(k)] = results_summary
    return output




# Test Code
# vector_u = np.random.rand(10)
# matrix_V = sparse.rand(5000, 10).tocsr()
# vector_train = sparse.rand(1, 1000).tocsr()
# vector_true = sparse.rand(1, 1000).tocsr()
#
#
# vector_predict, vector_true_dense, hits = sub_routine(vector_u, matrix_V, vector_train, vector_true, k=500)
# print r_precision(vector_true_dense=vector_true_dense, hits=hits)
# print ndcg(vector_true_dense=vector_true_dense, vector_predict=vector_predict, hits=hits)
# print click(hits=hits)