import argparse
import pandas as pd
from experiment.execute import execute
from utils.io import load_numpy, save_dataframe_csv, find_best_hyperparameters, load_yaml
from utils.modelnames import models
from plots.rec_plots import precision_recall_curve
import timeit

def main(args):
    table_path = load_yaml('config/global.yml', key='path')['tables']

    df = find_best_hyperparameters(table_path+args.problem, 'NDCG')

    R_train = load_numpy(path=args.path, name=args.train)
    R_valid = load_numpy(path=args.path, name=args.valid)
    R_test = load_numpy(path=args.path, name=args.test)

    R_train = R_train + R_valid

    topK = [5, 10, 15, 20, 50]

    frame = []
    for idx, row in df.iterrows():
        start = timeit.default_timer()
        row = row.to_dict()
        row['metric'] = ['R-Precision', 'NDCG', 'Precision', 'Recall', "MAP"]
        row['topK'] = topK
        result = execute(R_train, R_test, row, models[row['model']],
                         measure=row['similarity'], gpu_on=args.gpu, folder=args.model_folder)
        stop = timeit.default_timer()
        print('Time: ', stop - start)
        frame.append(result)

    results = pd.concat(frame)
    save_dataframe_csv(results, table_path, args.name)
    precision_recall_curve(results, topK, save=True, folder='analysis/'+args.problem)

if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="Reproduce")
    parser.add_argument('-n', dest='name', default="final_result.csv")
    parser.add_argument('-d', dest='path', default="datax/")
    parser.add_argument('-t', dest='train', default='Rtrain.npz')
    parser.add_argument('-v', dest='valid', default='Rvalid.npz')
    parser.add_argument('-e', dest='test', default='Rtest.npz')
    parser.add_argument('-p', dest='problem', default='movielens1m')
    parser.add_argument('-s', dest='model_folder', default='latent') # Model saving folder
    parser.add_argument('-gpu', dest='gpu', action='store_true')
    args = parser.parse_args()

    main(args)