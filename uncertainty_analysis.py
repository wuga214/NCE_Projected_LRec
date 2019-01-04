import numpy as np
import argparse
from experiment.uncertainty import uncertainty
from plots.rec_plots import show_uncertainty
from utils.io import load_numpy, save_dataframe_csv, find_best_hyperparameters, load_yaml


def main(args):
    table_path = load_yaml('config/global.yml', key='path')['tables']
    df = find_best_hyperparameters(table_path+args.problem, 'NDCG')
    R_train = load_numpy(path=args.path, name=args.train)
    results = uncertainty(R_train, df, rank=200)

    show_uncertainty(results, x='numRated', y='std', hue='model',
                     folder=args.problem, name=args.name, save=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ParameterTuning")
    parser.add_argument('-n', dest='name', default="uncertainty_analysis")
    parser.add_argument('-d', dest='path', default="datax/")
    parser.add_argument('-t', dest='train', default='Rtrain.npz')
    parser.add_argument('-p', dest='problem', default='movielens1m')
    args = parser.parse_args()

    main(args)