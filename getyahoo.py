from utils.io import save_numpy, load_pandas, save_array, load_yahoo
from utils.argument import shape, ratio
from providers.split import time_ordered_split, split_seed_randomly
import argparse
from utils.progress import WorkSplitter

import sys

def main(args):
    progress = WorkSplitter()
    progress.section("Load Raw Data")
    #rating_matrix = load_pandas(path=args.path, name=args.name, shape=args.shape)
    rating_matrix = load_yahoo(path=args.path, name=args.name, shape=args.shape)
    #timestamp_matrix = load_pandas(path=args.path, value_name='timestamp', name=args.name, shape=args.shape)
    progress.section("Split CSR Matrices")
    #rtrain, rvalid, rtest, nonzero_index = time_ordered_split(rating_matrix=rating_matrix,                                                          ratio=args.ratio, implicit=args.implicit)
    rtrain, rvalid, rtest, nonzero_index = split_seed_randomly(rating_matrix=rating_matrix, ratio=args.ratio, implicit=args.implicit)
    print("Done splitting Yahoo dataset")
    progress.section("Save NPZ")
    save_numpy(rtrain, args.path, "Rtrain")
    save_numpy(rvalid, args.path, "Rvalid")
    save_numpy(rtest, args.path, "Rtest")
    save_array(nonzero_index, args.path, "Index")
    print("Done saving data for yahoo after splitting")

if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="Data Preprocess")
    #parser.add_argument('--implicit', dest='implicit', action='store_true')
    parser.add_argument('--implicit', dest='implicit', default=True)
    parser.add_argument('-r', dest='ratio', type=ratio, default='0.5,0.2,0.3')
    pathToYahoo = "/root/Github/RecommendationSystems/sclrecommender/data/"
    parser.add_argument('-d', dest='path',
                        default=pathToYahoo)
    # Note: Modified the file to change negative itemId to positive manually using replace all in vim
    parser.add_argument('-n', dest='name', default='yahoo/ydata-ymusic-user-artist-ratings-v1_0modified.txt')
    parser.add_argument('--shape', help="CSR Shape", dest="shape", type=shape, nargs=2)
    args = parser.parse_args()
    main(args)
