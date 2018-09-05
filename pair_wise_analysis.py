import numpy as np
from utils.progress import WorkSplitter, inhour
import argparse
import time
from utils.io import save_mxnet, load_numpy, load_pandas, load_csv
from utils.argument import check_float_positive, check_int_positive, shape
from models.predictor import predict,predict_batch
from evaluation.metrics import evaluate_analysis
import os.path
import pandas as pd
from plot.plot import pandas_scatter_plot, pandas_bar_plot, pandas_group_hist_plot, pandas_group_distribution_plot

import itertools

models = [
    "NCE-PLRec",
    "PLRec",
    # "NCE-SVD",
    # "POP",
    # "WRMF",
    # "CML",
    # "AutoRec"
]


def getGroup(user_counts):
    # patents = [[1, 6], [7, 21], [22, 46], [47, 98], [99, 400], [401, 2000]]
    patents = [[1, 10], [11, 21], [22, 50], [51, 2084]]
    group = []
    for user_count in user_counts:
        for patent in patents:
            if user_count >= patent[0] and user_count <= patent[1]:
                group.append(str(patent))

    return group


def main(args):
    progress = WorkSplitter()

    # Show hyper parameter settings
    progress.section("Parameter Setting")
    print("Data Path: {0}".format(args.path))
    print("Train File Name: {0}".format(args.train))
    R_train = load_numpy(path=args.path, name=args.train)
    R_valid = load_numpy(path=args.path, name=args.valid)

    user_observation_counts = np.array(np.sum(R_train, axis=1)).flatten()
    user_observation_counts = user_observation_counts[np.array(np.sum(R_valid, axis=1)).flatten() != 0]

    pandas_group_distribution_plot(pd.DataFrame({'c': user_observation_counts, 't': getGroup(user_observation_counts)}),
                                   [[1, 10], [11, 21], [22, 50], [51, 2084]],
                                   't',
                                   '# of user ratings',
                                   'Count')

    item_popularity = np.array(np.sum(R_train, axis=0)).flatten()

    if args.validation:
        print("Valid File Name: {0}".format(args.valid))

    results = dict()
    for model in models:

        if model == 'CML':
            similarity = 'Eulidean'
        else:
            similarity = 'Cosine'

        RQ = np.load('latent/U_{0}_{1}.npy'.format(model, args.rank))
        Y = np.load('latent/V_{0}_{1}.npy'.format(model, args.rank))

        if os.path.isfile('latent/B_{0}_{1}.npy'.format(model, args.rank)):
            Bias = np.load('latent/B_{0}_{1}.npy'.format(model, args.rank))
        else:
            Bias = None

        progress.section(model + " Predict")
        prediction = predict(matrix_U=RQ,
                                   matrix_V=Y,
                                   bias=Bias,
                                   topK=args.topk,
                                   matrix_Train=R_train,
                                   measure=similarity,
                                   gpu=True)

        progress.section("Create Metrics")
        start_time = time.time()

        metric_names = ['R-Precision', 'NDCG', 'Recall', 'Precision']
        result = evaluate_analysis(prediction, R_valid, metric_names, [args.topk])
        top_popularity = item_popularity[prediction[:, 0].astype(np.int32)]

        result['pop'] = top_popularity[np.array(np.sum(R_valid, axis=1)).flatten() != 0]
        #import ipdb;ipdb.set_trace()

        results[model] = result

    evaluated_metrics = results['NCE-PLRec'].keys()

    giant_dataframes = []
    for model in models:
        df = pd.DataFrame(results[model])
        df['model'] = model
        df['user_count'] = user_observation_counts
        giant_dataframes.append(df)

    df = pd.concat(giant_dataframes)

    df = df.groupby(['model', 'user_count']).mean().reset_index()

    pandas_group_hist_plot(df, 'pop', 'model', 'Popularity of The First Recommended Item', 'Density')

    df['group'] = getGroup(df['user_count'].values)

    for metric in evaluated_metrics:
        pandas_bar_plot(x='group', y=metric, hue='model', x_name='# of user ratings', y_name=metric, df=df)

    pairs = list(itertools.combinations(models, 2))

    for pair in pairs[:]:
        candid1 = results[pair[0]]
        candid2 = results[pair[1]]

        for metric in evaluated_metrics:
            x = np.array(candid1[metric]).astype(np.float16)
            y = np.array(candid2[metric]).astype(np.float16)

            useless_index = np.intersect1d(np.where(x == 0)[0],np.where(y == 0)[0])

            x = np.delete(x, useless_index)
            y = np.delete(y, useless_index)

            max1 = np.max(x)
            max2 = np.max(y)
            max = np.minimum(max1, max2)

            positive_percentage = ((x - y) > 0).sum()
            negative_percentage = ((x - y) < 0).sum()

            size = np.abs(x-y).astype(np.float16)
            df = pd.DataFrame({'x': x, 'y': y, 'diff': x-y, 'Diverge': size})
            pandas_scatter_plot(df,
                                pair[0],
                                pair[1],
                                metric,
                                positive_percentage,
                                negative_percentage,
                                max, folder='figures/pairwise', save=True)

if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="LRec")

    parser.add_argument('--disable-item-item', dest='item', action='store_false')
    parser.add_argument('--disable-validation', dest='validation', action='store_false')

    parser.add_argument('-i1', dest='iter1', type=check_int_positive, default=1)
    parser.add_argument('-a1', dest='alpha1', type=check_float_positive, default=100.0)
    parser.add_argument('-l1', dest='lamb1', type=check_float_positive, default=100)
    parser.add_argument('-r', dest='rank', type=check_int_positive, default=100)

    parser.add_argument('-d', dest='path', default="datax/")
    parser.add_argument('-t', dest='train', default='Rtrain.npz')
    parser.add_argument('-v', dest='valid', default='Rvalid.npz')
    parser.add_argument('-k', dest='topk', type=check_int_positive, default=50)
    parser.add_argument('--shape', help="CSR Shape", dest="shape", type=shape, nargs=2)
    args = parser.parse_args()

    main(args)