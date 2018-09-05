import numpy as np
from utils.progress import WorkSplitter, inhour
import argparse
import time
from utils.io import save_mxnet, load_numpy, load_pandas, load_csv
from utils.argument import check_float_positive, check_int_positive, shape
from models.lrec import embedded_lrec_items_analysis
from models.pure_svd import pure_svd_analysis
from models.pmi_lrec import pmi_lrec_items_analysis
from models.predictor import predict,predict_batch
from evaluation.metrics import evaluate_analysis
import scipy.sparse as sparse
import pandas as pd
import json
from plot.plot import pandas_scatter_plot


models = {
    "PLRec": embedded_lrec_items_analysis,
    "NCE-PLRec": pmi_lrec_items_analysis,
}


def main(args):
    # Progress bar
    progress = WorkSplitter()

    # Show hyper parameter settings
    progress.section("Parameter Setting")
    print("Data Path: {0}".format(args.path))
    print("Train File Name: {0}".format(args.train))
    print("Alpha: {0}".format(args.alpha))
    print("Rank: {0}".format(args.rank))
    print("Lambda: {0}".format(args.lamb))
    print("SVD/Alter Iteration: {0}".format(args.iter))
    print("Evaluation Ranking Topk: {0}".format(args.topk))

    # Load Data
    progress.section("Loading Data")
    start_time = time.time()
    R_train = load_numpy(path=args.path, name=args.train)
    R_valid = load_numpy(path=args.path, name=args.valid)

    print "Elapsed: {0}".format(inhour(time.time() - start_time))

    # hold out users
    m, n = R_train.shape
    holdout_index = np.random.choice(m, m/20)
    training_index = sorted(list(set(range(m)) - set(holdout_index)))
    Train = R_train[training_index]
    Cold = R_train[holdout_index]
    Cold_valid = R_valid[holdout_index]

    results = dict()

    def getReport(model_name, lam):

        Q, Yt = models[model_name](Train, embeded_matrix=np.empty((0)),
                                   iteration=args.iter, rank=args.rank,
                                   lam=lam, alpha=args.alpha, seed=args.seed, root=args.root)
        Y = Yt.T

        RQ = Cold.dot(Q)

        progress.section("Predict")
        prediction = predict(matrix_U=RQ,
                             matrix_V=Y,
                             bias=None,
                             topK=args.topk,
                             matrix_Train=Cold,
                             measure="Cosine",
                             gpu=True)

        metric_names = ['R-Precision', 'NDCG', 'Recall', 'Precision']

        result = evaluate_analysis(prediction, Cold_valid, metric_names, [args.topk])
        results[model_name] = result

    getReport('PLRec', 100)
    getReport('NCE-PLRec', 100)

    evaluated_metrics = results['NCE-PLRec'].keys()

    candid1 = results['NCE-PLRec']
    candid2 = results['PLRec']

    for metric in evaluated_metrics:
        x = np.array(candid1[metric]).astype(np.float16)
        y = np.array(candid2[metric]).astype(np.float16)

        useless_index = np.intersect1d(np.where(x == 0)[0],np.where(y == 0)[0])

        x = np.delete(x, useless_index)
        y = np.delete(y, useless_index)

        max1 = np.max(x)
        max2 = np.max(y)
        max = np.minimum(max1, max2)

        positive_percentage = ((x - y) > 0).sum()
        negative_percentage = ((x - y) < 0).sum()

        size = np.abs(x-y).astype(np.float16)
        df = pd.DataFrame({'x': x, 'y': y, 'diff': x-y, 'Diverge': size})
        pandas_scatter_plot(df,
                            'NCE-PLRec',
                            'PLRec',
                            metric,
                            positive_percentage,
                            negative_percentage,
                            max, folder='figures/cold_start', save=True)



if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="LRec")

    parser.add_argument('-i', dest='iter', type=check_int_positive, default=1)
    parser.add_argument('-a', dest='alpha', type=check_float_positive, default=100.0)
    parser.add_argument('-l', dest='lamb', type=check_float_positive, default=100)
    parser.add_argument('-r', dest='rank', type=check_int_positive, default=100)
    parser.add_argument('-f', dest='root', type=check_float_positive, default=1)
    parser.add_argument('-s', dest='seed', type=check_int_positive, default=1)
    parser.add_argument('-d', dest='path', default="datax/")
    parser.add_argument('-k', dest='topk', type=check_int_positive, default=50)
    parser.add_argument('-t', dest='train', default='Rtrain.npz')
    parser.add_argument('-v', dest='valid', default='Rvalid.npz')
    parser.add_argument('-x', dest='index', default='Index.npy')
    parser.add_argument('-n', dest='side', default='ml-1m/movies.dat')
    parser.add_argument('-sample', dest='sample_size', type=check_int_positive, default=5)
    args = parser.parse_args()

    main(args)