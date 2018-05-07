import numpy as np
from scipy.sparse import load_npz, save_npz
from models.lrec import embedded_lirec_items, embedded_lirec_users
from utils.progress import WorkSplitter, inhour
import argparse
import time


PATH = '/media/wuga/Storage/python_project/lrec/data/'
TRAIN_NPY = PATH+'R_train.npz'
VALID_NPY = PATH+'R_valid.npz'
ROWS_NPY = PATH+'validRows.npy'
RANK = 200


def check_positive(value):
    ivalue = int(value)
    if ivalue <= 0:
         raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue


def main(args):
    progress = WorkSplitter()

    progress.section("Parameter Setting")
    print("Rank: {0}".format(RANK))
    if args.item == True:
        mode = "Item based"
    else:
        mode = "User based"
    print("Mode: {0}".format(mode))
    print("SVD Iteration: {0}".format(args.iter))

    progress.section("Loading Data")
    start_time = time.time()
    R_train = load_npz(TRAIN_NPY).tocsr()
    R_valid = load_npz(VALID_NPY).tocsr()
    valid_rows = np.load(ROWS_NPY)
    print "Elapsed: {0}".format(inhour(time.time() - start_time))

    print("Train U-I Dimensions: {0}".format(R_train.shape))
    print("Valid U-I Dimensions: {0}".format(R_valid.shape))

    if args.item == True:
        RQ, Y = embedded_lirec_items(R_train, embeded_matrix=np.empty((0)), iteration=args.iter, lam=80, rank=RANK)
        print type(RQ)
        print RQ.shape
        print type(Y)
        print Y.shape
        progress.section("Selected Save U-V Matrix")
        start_time = time.time()
        save_npz(PATH + 'U_{0}.npy'.format(RANK), RQ)
        save_npz(PATH + 'V_{0}.npy'.format(RANK), Y)
        print "Elapsed: {0}".format(inhour(time.time() - start_time))
    else:
        PtR, Y = embedded_lirec_users(R_train, embeded_matrix=np.empty((0)), iteration=args.iter, lam=80, rank=RANK)
        print type(PtR)
        print PtR.shape
        print type(Y)
        print Y.shape
        progress.section("Save U-V Matrix")
        start_time = time.time()
        save_npz(PATH + 'U_{0}.npy'.format(RANK), Y)
        save_npz(PATH + 'V_{0}.npy'.format(RANK), PtR)
        print "Elapsed: {0}".format(inhour(time.time() - start_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Projected LRec")

    parser.add_argument('--disable-item-item', dest='item', action='store_false')
    parser.add_argument('-i', dest='iter', type=check_positive)
    args = parser.parse_args()

    main(args)