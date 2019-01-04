import numpy as np
import argparse
from experiment.tuning import hyper_parameter_tuning
from utils.io import load_numpy, save_dataframe_csv, load_yaml
from utils.modelnames import models


def main(args):
    params = load_yaml(args.grid)
    params['models'] = {params['models']: models[params['models']]}
    R_train = load_numpy(path=args.path, name=args.train)
    R_valid = load_numpy(path=args.path, name=args.valid)
    hyper_parameter_tuning(R_train, R_valid, params, save_path=args.name, measure=params['similarity'], gpu_on=args.gpu)

if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="ParameterTuning")
    parser.add_argument('-n', dest='name', default="autorecs_tuning.csv")
    parser.add_argument('-d', dest='path', default="datax/")
    parser.add_argument('-t', dest='train', default='Rtrain.npz')
    parser.add_argument('-v', dest='valid', default='Rvalid.npz')
    parser.add_argument('-y', dest='grid', default='config/default.yml')
    parser.add_argument('-gpu', dest='gpu', action='store_true')
    args = parser.parse_args()

    main(args)