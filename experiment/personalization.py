import os
import numpy as np
import pandas as pd
from collections import defaultdict
from models.predictor import predict
from plots.rec_plots import pandas_ridge_plot


def personalization(Rtrain, Rvalid, df_input, topK, problem, model_folder, gpu_on=True):
    item_popularity = np.array(np.sum(Rtrain, axis=0)).flatten()

    index = None

    medians = []
    giant_dataframes = defaultdict(list)

    for idx, row in df_input.iterrows():
        row = row.to_dict()

        RQ = np.load('{2}/U_{0}_{1}.npy'.format(row['model'], row['rank'], model_folder))
        Y = np.load('{2}/V_{0}_{1}.npy'.format(row['model'], row['rank'], model_folder))

        if os.path.isfile('{2}/B_{0}_{1}.npy'.format(row['model'], row['rank'], model_folder)):
            Bias = np.load('{2}/B_{0}_{1}.npy'.format(row['model'], row['rank'], model_folder))
        else:
            Bias = None

        prediction = predict(matrix_U=RQ,
                             matrix_V=Y,
                             bias=Bias,
                             topK=topK[-1],
                             matrix_Train=Rtrain,
                             measure=row['similarity'],
                             gpu=gpu_on)

        for k in topK:

            result = dict()
            top_popularity = item_popularity[prediction[:,:k].astype(np.int32)]
            result['pop'] = top_popularity[np.array(np.sum(Rvalid, axis=1)).flatten() != 0].flatten()

            df = pd.DataFrame(result)
            df['model'] = row['model']
            giant_dataframes[k].append(df)
            if k == topK[0]:
                medians.append(np.median(result['pop']))

    index = np.argsort(medians).tolist()

    for k in topK:
        giant_dataframes[k] = [giant_dataframes[k][i] for i in index]

        df = pd.concat(giant_dataframes[k])

        df.to_csv('caches/personalization_at_{0}.csv'.format(k))

        pandas_ridge_plot(df, 'model', 'pop', k, folder='analysis/{0}/personalization'.format(problem),
                          name="personalization_at_{0}".format(k))

