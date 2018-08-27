import numpy as np
import pandas as pd
from tqdm import tqdm
from evaluation.metrics import evaluate
from utils.progress import WorkSplitter
import inspect
from models.predictor import predict


def hyper_parameter_tuning(train, validation, params, measure='Cosine', gpu_on=True):
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
                    RQ, Yt, Bias = params['models'][algorithm](train,
                                                            embeded_matrix=np.empty((0)),
                                                            iteration=params['iter'],
                                                            rank=rank,
                                                            lam=params['lam'],
                                                            root=root,
                                                            alpha=alpha,
                                                            gpu_on=True)
                    Y = Yt.T

                    progress.subsection("Prediction")

                    prediction = predict(matrix_U=RQ, matrix_V=Y, measure=measure, bias=Bias,
                                         topK=params['topK'][-1], matrix_Train=train, gpu=gpu_on)

                    progress.subsection("Evaluation")

                    result = evaluate(prediction, validation, params['metric'], params['topK'])

                    result_dict = {'model': algorithm, 'rank': rank, 'root': root, 'alpha': alpha}

                    for name in result.keys():
                        result_dict[name] = [round(result[name][0], 4), round(result[name][1], 4)]

                    df = df.append(result_dict, ignore_index=True)
    return df
