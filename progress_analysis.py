import argparse
from experiment.converge import converge
from utils.io import load_numpy, find_best_hyperparameters, load_yaml
from plots.rec_plots import show_training_progress


def main(args):
    table_path = load_yaml('config/global.yml', key='path')['tables']

    df = find_best_hyperparameters(table_path+args.param, 'NDCG')

    R_train = load_numpy(path=args.path, name=args.train)
    R_valid = load_numpy(path=args.path, name=args.valid)

    results = converge(R_train, R_valid, df, table_path, args.name, epochs=500, gpu_on=args.gpu)

    show_training_progress(results, hue='model', metric='NDCG', name="epoch_vs_ndcg")

if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="progress_analysis")
    parser.add_argument('-n', dest='name', default="convergence_analysis.csv")
    parser.add_argument('-d', dest='path', default="datax/")
    parser.add_argument('-t', dest='train', default='Rtrain.npz')
    parser.add_argument('-v', dest='valid', default='Rtest.npz')
    parser.add_argument('-p', dest='param', default='movielens1m')
    parser.add_argument('-type', dest='type', default='optimizer')
    parser.add_argument('-gpu', dest='gpu', action='store_true')
    args = parser.parse_args()

    main(args)