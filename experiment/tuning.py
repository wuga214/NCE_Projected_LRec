import numpy as np
import pandas as pd
from tqdm import tqdm
from evaluation.metrics import evaluate
from utils.progress import WorkSplitter
import inspect
from models.predictor import predict


def hyper_parameter_tuning(train, validation, params):
    progress = WorkSplitter()
    df = pd.DataFrame(columns=['model', 'rank', 'alpha', 'root', 'topK'])

    num_user = train.shape[0]

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

                    result_dict = {'model': algorithm, 'rank': rank, 'root': root, 'alpha': alpha}

                    for k in params['topK']:
                        result_dict['R-Precision@{0}'.format(k)] = round(result[str(k)]['R-Precision'], 5)
                        result_dict['RP-CI@{0}'.format(k)] = round(1.96*result[str(k)]['R-Precision_std']/np.sqrt(num_user), 5)
                        result_dict['NDCG@{0}'.format(k)] = round(result[str(k)]['NDCG'], 5)
                        result_dict['NDCG-CI@{0}'.format(k)] = round(1.96*result[str(k)]['NDCG_std']/np.sqrt(num_user), 5)

                    df = df.append(result_dict, ignore_index=True)
    return df