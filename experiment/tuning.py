import numpy as np
import pandas as pd
from tqdm import tqdm
from evaluation.metrics import evaluate
from utils.progress import WorkSplitter


def hyper_parameter_tuning(train, validation, params):
    progress = WorkSplitter()
    df = pd.DataFrame(columns=['model', 'rank', 'lambda', 'alpha', 'R-Precision', 'NDCG'])

    for algorithm in params['models']:

        for rank in params['rank']:
            for lam in params['lambda']:
                for alpha in tqdm(params['alphas']):
                    progress.section("model: {0}, rank: {1}, lambda: {2}, alpha: {3}".format(algorithm,
                                                                                             rank,
                                                                                             lam,
                                                                                             alpha))
                    RQ, Yt = params['models'][algorithm](train,
                                                         embeded_matrix=np.empty((0)),
                                                         iteration=params['iter'],
                                                         rank=rank,
                                                         lam=lam,
                                                         alpha=alpha)
                    Y = Yt.T

                    result = evaluate(RQ, Y, train, validation, params['topK'], params['metric'])
                    df = df.append({'model': algorithm,
                                    'rank': rank,
                                    'lambda': lam,
                                    'alpha': alpha,
                                    'R-Precision': result['R-Precision'],
                                    'NDCG': result['NDCG']
                                    },
                                   ignore_index=True)

    return df