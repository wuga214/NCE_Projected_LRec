import numpy as np
import argparse
from models.lrec import embedded_lrec_items
from models.weighted_lrec import weighted_lrec_items
from models.pure_svd import pure_svd, eigen_boosted_pure_svd
from models.als import als
from models.pmi_lrec import pmi_lrec_items
from models.weighted_pmi_lrec import weighted_pmi_lrec_items
from models.cml import cml
from models.cml_normalized import cml_normalized
from models.autorec import autorec
from experiment.tuning import hyper_parameter_tuning
from utils.io import load_numpy, save_dataframe_latex, save_dataframe_csv

#######################
# Set Measure!
#######################

models = {
    # "PLRec": embedded_lrec_items,
    # "WPLRec": weighted_lrec_items,
    # "PureSVD": pure_svd,
    # "PmiPLRec": pmi_lrec_items,
    # "PmiWPLRec": weighted_pmi_lrec_items,
    # "ALS": als,
    "AutoRec": autorec
}

# models = {
#     "CML": cml,
#     "NCML": cml_normalized,
# }


def main(args):
    params = {
        'models': models,
        'alpha': [-0.5, -0.1, 0.1, 10],
        'rank': [100],
        'root': [0.8, 0.9, 1.0, 1.1, 1.2],
        'topK': [5, 10, 15, 20, 50],
        'iter': 200,
        'lam': 10,
        'metric': ['R-Precision', 'NDCG', 'Precision', 'Recall'],
    }

    R_train = load_numpy(path=args.path, name=args.train)
    R_valid = load_numpy(path=args.path, name=args.valid)
    df = hyper_parameter_tuning(R_train, R_valid, params, measure="Euclidean")
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