from utils.io import save_numpy, load_pandas
from utils.argument import shape, check_float_positive
from providers.split import split
import argparse
from utils.progress import WorkSplitter, inhour


def main(args):
    progress = WorkSplitter()

    progress.section("Load Raw Data")
    matrix = load_pandas(path=args.path, name=args.name, shape=args.shape)
    progress.section("Split CSR Matrices")
    rtrain, rvalid = split(matrix=matrix, ratio=args.ratio, implicit=args.implicit, random=args.random)
    progress.section("Save NPZ")
    save_numpy(rtrain, args.path, "Rtrain")
    save_numpy(rvalid, args.path, "Rvalid")

if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="Data Preprocess")
    parser.add_argument('--implicit', dest='implicit', action='store_true')
    parser.add_argument('--random-split', dest='random', action='store_true')
    parser.add_argument('-r', dest='ratio', type=check_float_positive, default=0.3)
    parser.add_argument('-d', dest='path',
                        default="/media/wuga/Experiments/Recsys-18/IMPLEMENTATION_Projected_LRec/data/movielens/")
    parser.add_argument('-n', dest='name', default='ml-20m/ratings.csv')
    parser.add_argument('--shape', help="CSR Shape", dest="shape", type=shape, nargs=2)
    args = parser.parse_args()

    main(args)