import numpy as np
from models.lrec import embedded_lrec_items
from utils.progress import WorkSplitter, inhour
import argparse
import time
from utils.io import save_csr, load_csr
from models.weighted_lrec import weighted_lrec_items


models = {
    "PLRec": embedded_lrec_items,
    "WPLRec": weighted_lrec_items,
}


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

def shape(s):
    try:
        num = int(s)
        return num
    except:
        raise argparse.ArgumentTypeError("Sparse matrix shape must be integer")


def main(args):
    # Progress bar
    progress = WorkSplitter()

    # Show hyper parameter settings
    progress.section("Parameter Setting")
    print("Data Path: {0}".format(args.path))
    print("Data Name: {0}".format(args.name))
    print("Algorithm: {0}".format(args.model))
    if args.item == True:
        mode = "Item-based"
    else:
        mode = "User-based"
    print("Mode: {0}".format(mode))
    print("Alpha: {0}".format(args.alpha))
    print("Rank: {0}".format(args.rank))
    print("Lambda: {0}".format(args.lamb))
    print("SVD Iteration: {0}".format(args.iter))

    # Load Data
    progress.section("Loading Data")
    start_time = time.time()
    if args.shape is None:
        R_train = load_csr(path=args.path, name=args.name)
    else:
        R_train = load_csr(path=args.path, name=args.name, shape=args.shape)
    print "Elapsed: {0}".format(inhour(time.time() - start_time))

    print("Train U-I Dimensions: {0}".format(R_train.shape))

    # Item-Item or User-User
    if args.item == True:
        RQ, Yt = models[args.model](R_train, embeded_matrix=np.empty((0)),
                                    iteration=args.iter, lam=args.lamb, rank=args.rank, alpha=args.alpha)
        Y = Yt.T
    else:
        Y, RQt = models[args.model](R_train.T, embeded_matrix=np.empty((0)),
                                    iteration=args.iter, lam=args.lamb, rank=args.rank, alpha=args.alpha)
        RQ = RQt.T


    # Save Files
    progress.section("Save U-V Matrix")
    start_time = time.time()
    save_csr(matrix=RQ, path=args.path+mode+'/',
             name='U_{0}_{1}_{2}'.format(args.rank, args.lamb, args.model), format='MXNET')
    save_csr(matrix=Y, path=args.path+mode+'/',
             name='V_{0}_{1}_{2}'.format(args.rank, args.lamb, args.model), format='MXNET')
    print "Elapsed: {0}".format(inhour(time.time() - start_time))


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="LRec")

    parser.add_argument('--disable-item-item', dest='item', action='store_false')
    parser.add_argument('-i', dest='iter', type=check_int_positive, default=1)
    parser.add_argument('-a', dest='alpha', type=check_int_positive, default=100.0)
    parser.add_argument('-l', dest='lamb', type=check_float_positive, default=100.0)
    parser.add_argument('-r', dest='rank', type=check_int_positive, default=100)
    parser.add_argument('-m', dest='model', default="PLRec")
    parser.add_argument('-d', dest='path', default="/media/wuga/Storage/python_project/lrec/data/")
    parser.add_argument('-n', dest='name', default='R_train.npz')
    parser.add_argument('--shape', help="CSR Shape", dest="shape", type=shape, nargs=2)
    args = parser.parse_args()

    main(args)