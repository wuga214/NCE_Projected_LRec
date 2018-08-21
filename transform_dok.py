import argparse
from utils.io import load_numpy
import numpy as np
import pyximport; pyximport.install()
from utils.cython.negative_sampler import get_negative_samples
import scipy.sparse as sparse
from utils.progress import WorkSplitter, inhour


def main(args):
    progress = WorkSplitter()

    progress.section("Load Matrix")
    print("Data Path: {0}".format(args.path))
    print("Train File Name: {0}".format(args.matrix))
    matrix = load_numpy(path=args.path, name=args.matrix)

    print("Matrix Shape: {0}".format(matrix.shape))

    user_item_pair = np.asarray(matrix.nonzero())
    implicit_rating = np.ones((1, user_item_pair.shape[1]))
    data = np.vstack((user_item_pair, implicit_rating))

    progress.section("Save Positive Samples")
    print("Positive Sample Shape: {0}".format(data.T.shape))
    np.savetxt(fname='{0}/{1}.csv'.format(args.path, args.matrix), X=data.T, delimiter=',')

    if args.shape:
        np.savetxt(fname='{0}/shape.csv'.format(args.path), X=matrix.shape, delimiter=',')

    if args.neg:
        user_to_positive_set = {u: set(row) for u, row in enumerate(sparse.lil_matrix(matrix).rows)}

        neg_samples = np.random.randint(0,
                                        matrix.shape[1],
                                        size=(user_item_pair.shape[1], 4))

        neg_samples = get_negative_samples(user_to_positive_set, user_item_pair[0], matrix.shape[1], neg_samples)
        neg_data = np.vstack((np.repeat(user_item_pair[0], 4, axis=0),
                              neg_samples.flatten(),
                              np.zeros((1, 4*user_item_pair.shape[1]))))

        progress.section("Save Positive Samples")
        print("Negative Sample Shape: {0}".format(neg_data.T.shape))
        np.savetxt(fname='{0}/{1}.csv'.format(args.path, "negative"), X=neg_data.T, delimiter=',')


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="Function Caller")
    parser.add_argument('--negative-sampling', dest='neg', action='store_true')
    parser.add_argument('--output-shape', dest='shape', action='store_true')
    parser.add_argument('-d', dest='path', default="datax/")
    parser.add_argument('-m', dest='matrix', default='Rtrain.npz')
    args = parser.parse_args()

    main(args)