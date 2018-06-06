import numpy as np
import argparse
from models.lrec import embedded_lrec_items
from models.weighted_lrec import weighted_lrec_items
from models.pure_svd import pure_svd, eigen_boosted_pure_svd
from models.als import als
from experiment.tuning import hyper_parameter_tuning
from plot.plot import curve_weighting
from utils.io import load_numpy, save_datafram_latex, save_dataframe_csv


models = {
    "PLRec": embedded_lrec_items,
    "WPLRec": weighted_lrec_items,
    "PureSVD": pure_svd,
    "EBPureSVD": eigen_boosted_pure_svd,
    "ALS": als
}


def main(args):
    params = {
        'models': {args.model: models[args.model]},
        'alphas': [0.] + np.logspace(-2, 2.0, num=5).tolist(),
        'rank': [50, 100, 200, 400],
        'lambda': [1, 10, 100],
        'topK': 10,
        'iter': 7,
        'metric': ['R-Precision', 'NDCG'],
    }

    R_train = load_numpy(path=args.path, name=args.train)
    R_valid = load_numpy(path=args.path, name=args.valid)
    df = hyper_parameter_tuning(R_train, R_valid, params)
    save_datafram_latex(df, 'tables/', args.model)
    save_dataframe_csv(df, 'tables/', args.model)

if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="LRec")
    parser.add_argument('-m', dest='model', default="PLRec")
    parser.add_argument('-d', dest='path', default="datax/")
    parser.add_argument('-t', dest='train', default='Rtrain.npz')
    parser.add_argument('-v', dest='valid', default='Rvalid.npz')
    args = parser.parse_args()

    main(args)