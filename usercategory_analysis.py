import argparse
from experiment.usercategory import usercategory
from utils.io import load_numpy, find_best_hyperparameters, load_yaml


def main(args):
    table_path = load_yaml('config/global.yml', key='path')['tables']
    df = find_best_hyperparameters(table_path+args.problem, 'NDCG')

    R_train = load_numpy(path=args.path, name=args.train)
    R_valid = load_numpy(path=args.path, name=args.valid)
    R_test = load_numpy(path=args.path, name=args.test)

    R_train = R_train + R_valid

    topK = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    metric = ['R-Precision', 'NDCG', 'Clicks', 'Recall', 'Precision', 'MAP']

    usercategory(R_train, R_test, df, topK, metric, args.problem, args.model_folder, gpu_on=args.gpu)

if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="UserAnalysis")
    parser.add_argument('-d', dest='path', default="datax/")
    parser.add_argument('-t', dest='train', default='Rtrain.npz')
    parser.add_argument('-v', dest='valid', default='Rvalid.npz')
    parser.add_argument('-e', dest='test', default='Rtest.npz')
    parser.add_argument('-p', dest='problem', default='movielens1m')
    parser.add_argument('-s', dest='model_folder', default='latent')  # Model saving folder
    parser.add_argument('-gpu', dest='gpu', action='store_true')
    args = parser.parse_args()

    main(args)