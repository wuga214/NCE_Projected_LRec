from utils.io import save_numpy, load_pandas, save_array
from utils.argument import shape, ratio
from providers.split import time_ordered_split
import argparse
from utils.progress import WorkSplitter
import pandas as pd
from scipy.sparse import csr_matrix
import numpy as np

def main(args):
    progress = WorkSplitter()

    raw = pd.read_csv(args.path+args.name, names=['user', 'item', 'rating', 'timestamp'])

    raw['userID'] = pd.factorize(raw.user)[0]
    raw['itemID'] = pd.factorize(raw.item)[0]


    progress.section("Load Raw Data")
    rating_matrix = getSparseMatrix(raw, row_name='userID', col_name='itemID', value_name='rating')
    timestamp_matrix = getSparseMatrix(raw, row_name='userID', col_name='itemID', value_name='timestamp')
    progress.section("Split CSR Matrices")
    rtrain, rvalid, rtest, nonzero_index, rtime = time_ordered_split(rating_matrix=rating_matrix,
                                                                     timestamp_matrix=timestamp_matrix,
                                                                     ratio=args.ratio,
                                                                     implicit=args.implicit)
    progress.section("Save NPZ")
    save_numpy(rtrain, args.path, "Rtrain")
    save_numpy(rvalid, args.path, "Rvalid")
    save_numpy(rtest, args.path, "Rtest")
    save_numpy(rtime, args.path, "Rtime")
    save_array(nonzero_index, args.path, "Index")


def getSparseMatrix(df, row_name, col_name, value_name):
    rows = df[row_name]
    cols = df[col_name]
    values = df[value_name]
    return csr_matrix((values, (rows, cols)), shape=(np.amax(rows)+1, np.amax(cols)+1))


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="Data Preprocess")
    parser.add_argument('--implicit', dest='implicit', action='store_true')
    parser.add_argument('-r', dest='ratio', type=ratio, default='0.5,0.2,0.3')
    parser.add_argument('-d', dest='path',
                        default="data/amazon/")
    parser.add_argument('-n', dest='name', default='raw/ratings_Automotive.csv')
    args = parser.parse_args()

    main(args)