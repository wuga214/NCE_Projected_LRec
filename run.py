import argparse
from utils.io import load_numpy
import numpy as np
from experiment.latent_analysis import latent_analysis
from experiment.popular_analysis import popular_overlapping
from utils.argument import shape
from utils.argument import check_float_positive, check_int_positive, shape
from models.predictor import predict



def main(args):
    R_train = load_numpy(path=args.path, name=args.train)
    import ipdb;ipdb.set_trace()
    latent_analysis(R_train)


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="Function Caller")
    parser.add_argument('-d', dest='path', default="datax/")
    parser.add_argument('-t', dest='train', default='Rtrain.npz')
    parser.add_argument('-v', dest='valid', default='Rvalid.npz')
    parser.add_argument('--shape', help="CSR Shape", dest="shape", type=shape, nargs=2)
    parser.add_argument('-i', dest='iter', type=check_int_positive, default=4)
    parser.add_argument('-a', dest='alpha', type=check_float_positive, default=1.0)
    parser.add_argument('-l', dest='lamb', type=check_float_positive, default=1.0)
    parser.add_argument('-r', dest='rank', type=check_int_positive, default=100)
    parser.add_argument('-f', dest='root', type=check_float_positive, default=1)
    parser.add_argument('-s', dest='seed', type=check_int_positive, default=1)
    args = parser.parse_args()

    main(args)