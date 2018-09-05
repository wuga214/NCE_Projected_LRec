import numpy as np
import argparse
from experiment.batch_evaluation import lookup
from utils.io import load_numpy, save_dataframe_latex, save_dataframe_csv



def main(args):
    params = {
        #'models': ['NCE-PLRec', 'NCE-SVD', 'AutoRec', 'PLRec', 'WRMF', 'CML', 'POP'],
        'models': ['AutoRec'],
        'rank': 100,
        'topK': [5, 10, 15, 20, 50],
        'metric': ['R-Precision', 'NDCG', 'Precision', 'Recall'],
    }

    R_train = load_numpy(path=args.path, name=args.train)
    R_valid = load_numpy(path=args.path, name=args.valid)
    df = lookup(R_train, R_valid, params, measure="Cosine")
    save_dataframe_latex(df, 'tables/', args.name)
    save_dataframe_csv(df, 'tables/', args.name)

if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="LRec")
    parser.add_argument('-n', dest='name', default="movielens1m")
    parser.add_argument('-d', dest='path', default="datax/")
    parser.add_argument('-t', dest='train', default='Rtrain.npz')
    parser.add_argument('-v', dest='valid', default='Rvalid.npz')
    args = parser.parse_args()

    main(args)