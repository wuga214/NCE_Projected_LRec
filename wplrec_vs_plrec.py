import numpy as np
import argparse
from models.lrec import embedded_lrec_items
from models.weighted_lrec import weighted_lrec_items
from experiment.weighting import weighting
from plot.plot import curve_weighting
from utils.io import load_numpy, save_pickle, load_pickle

params = {
    'models': {"PLRec": embedded_lrec_items, "WPLRec": weighted_lrec_items},
    'alphas': np.linspace(-0.9, 1, num=50), #[-x for x in [0.] + np.logspace(-2, 0.0, num=5).tolist()],
    'rank': 50,
    'lambda': 0.001,
    'topK': [50],
    'iter': 4,
    'metric': ['R-Precision', 'NDCG'],
}


def main(args):
    R_train = load_numpy(path=args.path, name=args.train)
    R_valid = load_numpy(path=args.path, name=args.valid)
    lrec_result, wlrec_results = weighting(R_train, R_valid, params)

    save_pickle("cache", "weighting", (lrec_result, wlrec_results))
    lrec_result, wlrec_results = load_pickle("cache", "weighting")

    curve_weighting(lrec_result, wlrec_results, params['alphas'], metric='R-Precision', name="R-Precision")
    curve_weighting(lrec_result, wlrec_results, params['alphas'], metric='NDCG', name="NDCG")


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="WPLRec VS PLRec")
    parser.add_argument('-d', dest='path', default="datax/")
    parser.add_argument('-t', dest='train', default='Rtrain.npz')
    parser.add_argument('-v', dest='valid', default='Rvalid.npz')
    args = parser.parse_args()

    main(args)