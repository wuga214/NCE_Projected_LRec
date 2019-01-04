from utils.modelnames import vaes
from utils.progress import WorkSplitter
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from utils.regularizers import Regularizer


def uncertainty(Rtrain, df_input, rank):
    progress = WorkSplitter()
    m, n = Rtrain.shape

    valid_models = vaes.keys()

    results = []

    for run in range(1):

        for idx, row in df_input.iterrows():
            row = row.to_dict()

            if row['model'] not in valid_models:
                continue

            progress.section(json.dumps(row))

            if 'optimizer' not in row.keys():
                row['optimizer'] = 'RMSProp'

            model = vaes[row['model']](n, rank,
                                       batch_size=100,
                                       lamb=row['lambda'],
                                       optimizer=Regularizer[row['optimizer']])

            model.train_model(Rtrain, corruption=row['corruption'], epoch=row['iter'])
            data_batches = model.get_batches(Rtrain, batch_size=100)
            progress.subsection("Predict")
            for batch in tqdm(data_batches):
                batch_size = batch.shape[0]
                _, stds = model.uncertainty(batch.todense())
                num_rated = np.squeeze(np.asarray(np.sum(batch, axis=1)))
                std = np.mean(stds, axis=1)
                results.append(pd.DataFrame({'model': [row['model']]*batch_size, 'numRated': num_rated, 'std': std}))

    return pd.concat(results)