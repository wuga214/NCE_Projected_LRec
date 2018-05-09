import numpy as np
from scipy.sparse import load_npz, save_npz
from models.lrec import embedded_lirec_items, embedded_lirec_users
from utils.progress import WorkSplitter, inhour
import argparse
import time
from utils.io import save_csr


# Commandline parameter constrains
def check_int_positive(value):
    ivalue = int(value)
    if ivalue <= 0:
         raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue


def check_float_positive(value):
    ivalue = float(value)
    if ivalue <= 0:
         raise argparse.ArgumentTypeError("%s is an invalid positive float value" % value)
    return ivalue


def main(args):
    # Progress bar
    progress = WorkSplitter()

    # Constants
    train_npy = args.path + 'R_train.npz'
    valid_npy = args.path + 'R_valid.npz'
    rows_npy = args.path + 'validRows.npy'

    # Show hyper parameter settings
    progress.section("Parameter Setting")
    print("Data Path: {0}".format(args.path))
    print("Rank: {0}".format(args.rank))
    print("Lambda: {0}".format(args.lamb))
    if args.item == True:
        mode = "Item-based"
    else:
        mode = "User-based"
    print("Mode: {0}".format(mode))
    print("SVD Iteration: {0}".format(args.iter))

    # Load Data
    progress.section("Loading Data")
    start_time = time.time()
    R_train = load_npz(train_npy).tocsr()
    R_valid = load_npz(valid_npy).tocsr()
    valid_rows = np.load(rows_npy)
    print "Elapsed: {0}".format(inhour(time.time() - start_time))

    print("Train U-I Dimensions: {0}".format(R_train.shape))
    print("Valid U-I Dimensions: {0}".format(R_valid.shape))

    # Item-Item or User-User
    if args.item == True:
        RQ, Y = embedded_lirec_items(R_train, embeded_matrix=np.empty((0)),
                                     iteration=args.iter, lam=args.lamb, rank=args.rank)

        # Save Files
        progress.section("Save U-V Matrix")
        start_time = time.time()
        save_csr(matrix=RQ, path=args.path+mode, name='U_{0}_{1}'.format(args.rank, args.lamb), format='MXNET')
        save_csr(matrix=Y.T, path=args.path+mode, name='V_{0}_{1}'.format(args.rank, args.lamb), format='MXNET')
        print "Elapsed: {0}".format(inhour(time.time() - start_time))
    else:
        PtR, Y = embedded_lirec_users(R_train, embeded_matrix=np.empty((0)),
                                      iteration=args.iter, lam=args.lamb, rank=args.rank)

        # Save Files
        progress.section("Save U-V Matrix")
        start_time = time.time()
        save_csr(matrix=Y, path=args.path+mode, name='U_{0}_{1}'.format(args.rank, args.lamb), format='MXNET')
        save_csr(matrix=PtR.T, path=args.path+mode, name='V_{0}_{1}'.format(args.rank, args.lamb), format='MXNET')
        print "Elapsed: {0}".format(inhour(time.time() - start_time))


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="Projected LRec")

    parser.add_argument('--disable-item-item', dest='item', action='store_false')
    parser.add_argument('-i', dest='iter', type=check_int_positive, default=1)
    parser.add_argument('-l', dest='lamb', type=check_float_positive, default=100.0)
    parser.add_argument('-r', dest='rank', type=check_int_positive, default=100)
    parser.add_argument('-d', dest='path', default="/media/wuga/Storage/python_project/lrec/data/")
    args = parser.parse_args()

    main(args)