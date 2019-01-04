import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from evaluation.metrics import evaluate
from utils.progress import WorkSplitter
import inspect
from models.predictor import predict


def execute(train, test, params, model, measure='Cosine', gpu_on=True, analytical=False, folder='latent'):
    progress = WorkSplitter()

    columns = ['model', 'rank', 'alpha', 'lambda', 'iter', 'similarity', 'corruption', 'root', 'topK']

    progress.section("\n".join([":".join((str(k), str(params[k]))) for k in columns]))

    df = pd.DataFrame(columns=columns)

    if os.path.isfile('{2}/U_{0}_{1}.npy'.format(params['model'], params['rank'], folder)):

        RQ = np.load('{2}/U_{0}_{1}.npy'.format(params['model'], params['rank'], folder))
        Y = np.load('{2}/V_{0}_{1}.npy'.format(params['model'], params['rank'], folder))

        if os.path.isfile('{2}/B_{0}_{1}.npy'.format(params['model'], params['rank'], folder)):
            Bias = np.load('{2}/B_{0}_{1}.npy'.format(params['model'], params['rank'], folder))
        else:
            Bias = None
            
    else:

        RQ, Yt, Bias = model(train,
                             embeded_matrix=np.empty((0)),
                             iteration=params['iter'],
                             rank=params['rank'],
                             lam=params['lambda'],
                             alpha=params['alpha'],
                             corruption=params['corruption'],
                             root=params['root'],
                             gpu_on=gpu_on)
        Y = Yt.T

        np.save('{2}/U_{0}_{1}'.format(params['model'], params['rank'], folder), RQ)
        np.save('{2}/V_{0}_{1}'.format(params['model'], params['rank'], folder), Y)
        if Bias is not None:
            np.save('{2}/B_{0}_{1}'.format(params['model'], params['rank'], folder), Bias)

    progress.subsection("Prediction")

    prediction = predict(matrix_U=RQ, matrix_V=Y, measure=measure, bias=Bias,
                         topK=params['topK'][-1], matrix_Train=train, gpu=gpu_on)

    progress.subsection("Evaluation")

    result = evaluate(prediction, test, params['metric'], params['topK'], analytical=analytical)

    if analytical:
        return result
    else:
        result_dict = params

        for name in result.keys():
            result_dict[name] = [round(result[name][0], 4), round(result[name][1], 4)]
        df = df.append(result_dict, ignore_index=True)

        return df
