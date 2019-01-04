from utils.io import save_numpy, save_array, load_pandas_without_names
from utils.argcheck import shape, ratio
from providers.split import time_ordered_split, split_seed_randomly
import argparse
from utils.progress import WorkSplitter

import sys

def main(args):
    progress = WorkSplitter()
    progress.section("Load Raw Data")
    rating_matrix = load_pandas_without_names(path=args.path, name=args.name, row_name='userId', sep='\t',
                                              col_name='trackId', value_name='rating', shape=args.shape,
                                              names=['userId', 'trackId', 'rating'])
    progress.section("Split CSR Matrices")
    rtrain, rvalid, rtest, nonzero_index = split_seed_randomly(rating_matrix=rating_matrix,
                                                               ratio=args.ratio,
                                                               threshold=80,
                                                               implicit=args.implicit,
                                                               sampling=True,
                                                               percentage=0.2)
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
    parser.add_argument('--implicit', dest='implicit', action='store_true')
    parser.add_argument('-r', dest='ratio', type=ratio, default='0.5,0.2,0.3')
    parser.add_argument('-d', dest='path', default="data/yahoo/")
    # Note: Modified the file to change negative itemId to positive manually using replace all in vim
    parser.add_argument('-n', dest='name', default='ydata-ymusic-user-artist-ratings-v1_0modified.txt')
    parser.add_argument('--shape', help="CSR Shape", dest="shape", type=shape, nargs=2)
    args = parser.parse_args()
    main(args)