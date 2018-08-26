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
from evaluation.metrics import evaluate
import scipy.sparse as sparse
import pandas as pd
import json


models = {
    "PLRec": embedded_lrec_items_analysis,
    "PmiPLRec": pmi_lrec_items_analysis,
    "PureSVD": pure_svd_analysis,
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
    Index = np.load(args.path + args.index)
    Side_info = pd.read_csv(args.path + args.side,
                            delimiter='::', names=['index', 'name', 'type'], encoding='utf-8')
    Side_info = Side_info[Side_info['index'].isin(Index)].reset_index(drop=True)
    Side_info['popularity'] = np.asarray(np.sum(R_train, axis=0)).reshape(-1)
    Side_info['notes'] = Side_info['name'] + '<br>' + Side_info['type'] + '<br>Popularity:' + Side_info[
        'popularity'].astype(str)

    Item_names = Side_info['notes'].str.decode('iso-8859-1').str.encode('utf-8').tolist()

    print "Elapsed: {0}".format(inhour(time.time() - start_time))

    new_users = sparse.random(args.sample_size, R_train.shape[1], density=0.001).tocsr()
    new_users[new_users.nonzero()] = 1


    def getReport(model_name, lam):

        Q, Yt = models[model_name](R_train, embeded_matrix=np.empty((0)),
                                   iteration=args.iter, rank=args.rank,
                                   lam=lam, alpha=args.alpha, seed=args.seed, root=args.root)
        Y = Yt.T

        RQ = new_users.dot(Q)

        progress.section("Predict")
        prediction = predict(matrix_U=RQ,
                             matrix_V=Y,
                             bias=None,
                             topK=args.topk,
                             matrix_Train=new_users,
                             measure="Cosine",
                             gpu=True)

        results =  []
        for i in range(args.sample_size):
            query = [Item_names[n] for n in new_users[i].nonzero()[1]]
            answer = [Item_names[int(n)] for n in prediction[i]]
            results.append({'algorithm': model_name, 'query': query, 'retrieval': answer})

        return pd.DataFrame(results)

    df1 = getReport('PLRec', 100)
    df2 = getReport('PmiPLRec', 1000000)
    df3 = getReport('PureSVD', 100)

    frames = [df1, df2, df3]
    report = pd.concat(frames)
    report = report.sort_values(by=['query'])
    report.to_csv('tables/cold_start.csv')



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
    parser.add_argument('-k', dest='topk', type=check_int_positive, default=3)
    parser.add_argument('-t', dest='train', default='Rtrain.npz')
    parser.add_argument('-x', dest='index', default='Index.npy')
    parser.add_argument('-n', dest='side', default='ml-1m/movies.dat')
    parser.add_argument('-sample', dest='sample_size', type=check_int_positive, default=5)
    args = parser.parse_args()

    main(args)