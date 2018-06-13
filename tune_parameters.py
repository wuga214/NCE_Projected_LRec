import numpy as np
import argparse
from models.lrec import embedded_lrec_items
from models.weighted_lrec import weighted_lrec_items
from models.pure_svd import pure_svd, eigen_boosted_pure_svd
from models.als import als
from models.pmi_lrec import pmi_lrec_items
from models.weighted_pmi_lrec import weighted_pmi_lrec_items
from experiment.tuning import hyper_parameter_tuning
from utils.io import load_numpy, save_dataframe_latex, save_dataframe_csv


models = {
    "PLRec": embedded_lrec_items,
    "WPLRec": weighted_lrec_items,
    "PureSVD": pure_svd,
    "PmiPLRec": pmi_lrec_items,
    "PmiWPLRec": weighted_pmi_lrec_items,
    "ALS": als
}


def main(args):
    params = {
        'models': models,
        'alpha': [0, 1, 10, 100],
        'rank': [50, 100],
        'root': [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
        'topK': [5, 10, 15, 20],
        'iter': 7,
        'metric': ['R-Precision', 'NDCG'],
    }

    R_train = load_numpy(path=args.path, name=args.train)
    R_valid = load_numpy(path=args.path, name=args.valid)
    df = hyper_parameter_tuning(R_train, R_valid, params)
    save_dataframe_latex(df, 'tables/', args.name)
    save_dataframe_csv(df, 'tables/', args.name)

if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="LRec")
    parser.add_argument('-n', dest='name', default="movielens1m")
    parser.add_argument('-d', dest='path', default="datax/")
    parser.add_argument('-t', dest='train', default='Rtrain.npz')
    parser.add_argument('-v', dest='valid', default='Rtest.npz')
    args = parser.parse_args()

    main(args)