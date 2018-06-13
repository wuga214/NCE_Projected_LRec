import numpy as np
import pandas as pd
from tqdm import tqdm
from evaluation.metrics import evaluate
from utils.progress import WorkSplitter
import inspect
from models.predictor import predict

def hyper_parameter_tuning(train, validation, params):
    progress = WorkSplitter()
    df = pd.DataFrame(columns=['model', 'rank', 'alpha', 'root', 'topK', 'R-Precision', 'NDCG'])

    for algorithm in params['models']:

        for rank in params['rank']:
            if 'alpha' in inspect.getargspec(params['models'][algorithm])[0]:
                alphas = params['alpha']
            else:
                alphas = [1]

            for alpha in alphas:

                if 'root' in inspect.getargspec(params['models'][algorithm])[0]:
                    roots = params['root']
                else:
                    roots = [1]

                for root in roots:

                    progress.section("model: {0}, rank: {1}, root: {2}, alpha: {3}".format(algorithm,
                                                                                             rank,
                                                                                             root,
                                                                                             alpha))
                    RQ, Yt = params['models'][algorithm](train,
                                                         embeded_matrix=np.empty((0)),
                                                         iteration=params['iter'],
                                                         rank=rank,
                                                         lam=0.01,
                                                         root=root,
                                                         alpha=alpha)
                    Y = Yt.T

                    prediction = predict(matrix_U=RQ, matrix_V=Y,
                                         topK=params['topK'][-1], matrix_Train=train, gpu=True)

                    result = evaluate(prediction, validation, params['metric'], params['topK'])

                    for k in params['topK']:
                        df = df.append({'model': algorithm,
                                        'rank': rank,
                                        'root': root,
                                        'alpha': alpha,
                                        'topK': k,
                                        'R-Precision': result[str(k)]['R-Precision'],
                                        'NDCG': result[str(k)]['NDCG']
                                        },
                                       ignore_index=True)
    return df