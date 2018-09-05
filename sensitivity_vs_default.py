import numpy as np
import argparse
from models.pmi_lrec import pmi_lrec_items
from experiment.sensitivity import sensitivity
from plot.plot import curve_sensitivity
from utils.io import load_numpy, save_pickle, load_pickle

params = {
    'models': {"NCE-PLRec": pmi_lrec_items},
    'root': np.linspace(0.95, 1.15, num=50),
    'rank': 100,
    'lambda': 10000,
    'topK': [50],
    'iter': 4,
    'metric': ['R-Precision', 'NDCG'],
}


def main(args):
    R_train = load_numpy(path=args.path, name=args.train)
    R_valid = load_numpy(path=args.path, name=args.valid)
    lrec_result, wlrec_results = sensitivity(R_train, R_valid, params)

    save_pickle("cache", "sensitivity", (lrec_result, wlrec_results))
    lrec_result, wlrec_results = load_pickle("cache", "sensitivity")

    curve_sensitivity(lrec_result, wlrec_results, params['root'], metric='R-Precision', name="sensitivity/R-Precision")
    curve_sensitivity(lrec_result, wlrec_results, params['root'], metric='NDCG', name="sensitivity/NDCG")


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="WPLRec VS PLRec")
    parser.add_argument('-d', dest='path', default="data/")
    parser.add_argument('-t', dest='train', default='Rtrain.npz')
    parser.add_argument('-v', dest='valid', default='Rvalid.npz')
    args = parser.parse_args()

    main(args)