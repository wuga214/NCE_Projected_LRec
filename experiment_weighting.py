import numpy as np
import argparse
from models.lrec import embedded_lrec_items
from models.weighted_lrec import weighted_lrec_items
from experiment.weighting import weighting
from plot.plot import curve_weighting
from utils.io import load_numpy

params = {
    'models': {"PLRec": embedded_lrec_items, "WPLRec": weighted_lrec_items},
    'alphas':  [0.] + np.logspace(-2, 2.0, num=3).tolist(),
    'rank': 200,
    'lambda': 1,
    'topK': 10,
    'iter': 4,
    'metric': ['R-Precision', 'NDCG'],
}


def main(args):
    R_train = load_numpy(path=args.path, name=args.train)
    R_valid = load_numpy(path=args.path, name=args.valid)
    lrec_result, wlrec_results = weighting(R_train, R_valid, params)
    curve_weighting(lrec_result, wlrec_results, params['alphas'], metric='R-Precision', name="weighting")
    curve_weighting(lrec_result, wlrec_results, params['alphas'], metric='NDCG', name="weighting")


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="LRec")

    parser.add_argument('-d', dest='path', default="/media/wuga/Storage/python_project/lrec/data/")
    parser.add_argument('-t', dest='train', default='R_train.npz')
    parser.add_argument('-v', dest='valid', default='R_valid.npz')
    args = parser.parse_args()

    main(args)