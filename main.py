import numpy as np
from utils.progress import WorkSplitter, inhour
import argparse
import time
from utils.io import save_mxnet, load_numpy, load_pandas
from utils.argument import check_float_positive, check_int_positive, shape
from models.lrec import embedded_lrec_items
from models.weighted_lrec import weighted_lrec_items
from models.pure_svd import pure_svd, eigen_boosted_pure_svd
from models.als import als
from models.pmi_lrec import pmi_lrec_items
from evaluation.metrics import evaluate


models = {
    "PLRec": embedded_lrec_items,
    "WPLRec": weighted_lrec_items,
    "PmiPLRec": pmi_lrec_items,
    "PureSVD": pure_svd,
    "EBPureSVD": eigen_boosted_pure_svd,
    "ALS": als
}


def main(args):
    # Progress bar
    progress = WorkSplitter()

    # Show hyper parameter settings
    progress.section("Parameter Setting")
    print("Data Path: {0}".format(args.path))
    print("Train File Name: {0}".format(args.train))
    if args.validation:
        print("Valid File Name: {0}".format(args.valid))
    print("Algorithm: {0}".format(args.model))
    if args.item == True:
        mode = "Item-based"
    else:
        mode = "User-based"
    print("Mode: {0}".format(mode))
    print("Alpha: {0}".format(args.alpha))
    print("Rank: {0}".format(args.rank))
    print("Lambda: {0}".format(args.lamb))
    print("SVD/Alter Iteration: {0}".format(args.iter))
    print("Evaluation Ranking Topk: {0}".format(args.topk))

    # Load Data
    progress.section("Loading Data")
    start_time = time.time()
    if args.shape is None:
        R_train = load_numpy(path=args.path, name=args.train)
    else:
        R_train = load_pandas(path=args.path, name=args.train, shape=args.shape)
    print "Elapsed: {0}".format(inhour(time.time() - start_time))

    print("Train U-I Dimensions: {0}".format(R_train.shape))

    # Item-Item or User-User
    if args.item == True:
        RQ, Yt = models[args.model](R_train, embeded_matrix=np.empty((0)),
                                    iteration=args.iter, rank=args.rank,
                                    lam=args.lamb, alpha=args.alpha, seed=args.seed)
        Y = Yt.T
    else:
        Y, RQt = models[args.model](R_train.T, embeded_matrix=np.empty((0)),
                                    iteration=args.iter, rank=args.rank,
                                    lam=args.lamb, alpha=args.alpha, seed=args.seed)
        RQ = RQt.T

    # Save Files
    progress.section("Save U-V Matrix")
    start_time = time.time()
    save_mxnet(matrix=RQ, path=args.path+mode+'/',
               name='U_{0}_{1}_{2}'.format(args.rank, args.lamb, args.model))
    save_mxnet(matrix=Y, path=args.path+mode+'/',
               name='V_{0}_{1}_{2}'.format(args.rank, args.lamb, args.model))
    print "Elapsed: {0}".format(inhour(time.time() - start_time))

    if args.validation:
        progress.section("Create Metrics")
        start_time = time.time()

        metric_names = ['R-Precision', 'NDCG', 'Clicks']
        R_valid = load_numpy(path=args.path, name=args.valid)
        result = evaluate(RQ, Y, R_train, R_valid, args.topk, metric_names)
        print("-")
        for key in result.keys():
            print("{0} :{1}".format(key, result[key]))
        print "Elapsed: {0}".format(inhour(time.time() - start_time))


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="LRec")

    parser.add_argument('--disable-item-item', dest='item', action='store_false')
    parser.add_argument('--disable-validation', dest='validation', action='store_false')
    parser.add_argument('-i', dest='iter', type=check_int_positive, default=1)
    parser.add_argument('-a', dest='alpha', type=check_float_positive, default=100.0)
    parser.add_argument('-l', dest='lamb', type=check_float_positive, default=100.0)
    parser.add_argument('-r', dest='rank', type=check_int_positive, default=100)
    parser.add_argument('-s', dest='seed', type=check_int_positive, default=1)
    parser.add_argument('-m', dest='model', default="PLRec")
    parser.add_argument('-d', dest='path', default="/media/wuga/Storage/python_project/lrec/data/")
    parser.add_argument('-t', dest='train', default='R_train.npz')
    parser.add_argument('-v', dest='valid', default='R_valid.npz')
    parser.add_argument('-k', dest='topk', type=check_int_positive, default=50)
    parser.add_argument('--shape', help="CSR Shape", dest="shape", type=shape, nargs=2)
    args = parser.parse_args()

    main(args)