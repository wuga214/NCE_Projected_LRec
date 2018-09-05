import numpy as np
import pandas as pd
from tqdm import tqdm
from evaluation.metrics import evaluate
from utils.progress import WorkSplitter
import inspect
import os.path
from models.predictor import predict


def lookup(train, validation, params, measure='Cosine', gpu_on=True):
    progress = WorkSplitter()
    df = pd.DataFrame(columns=['model'])

    num_user = train.shape[0]

    for algorithm in params['models']:

        RQ = np.load('latent/U_{0}_{1}.npy'.format(algorithm, params['rank']))
        Y = np.load('latent/V_{0}_{1}.npy'.format(algorithm, params['rank']))
        if os.path.isfile('latent/B_{0}_{1}.npy'.format(algorithm, params['rank'])):
            Bias = np.load('latent/B_{0}_{1}.npy'.format(algorithm, params['rank']))
        else:
            Bias = None


        progress.subsection("Prediction")

        prediction = predict(matrix_U=RQ, matrix_V=Y, measure=measure, bias=Bias,
                             topK=params['topK'][-1], matrix_Train=train, gpu=gpu_on)

        progress.subsection("Evaluation")

        result = evaluate(prediction, validation, params['metric'], params['topK'])

        result_dict = {'model': algorithm}

        for name in result.keys():
            result_dict[name] = [round(result[name][0], 4), round(result[name][1], 4)]

        df = df.append(result_dict, ignore_index=True)
    return df
