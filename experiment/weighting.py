import numpy as np
from tqdm import tqdm
from evaluation.metrics import evaluate
from models.predictor import predict
from utils.progress import WorkSplitter, inhour


def weighting(train, validation, params):
    progress = WorkSplitter()
    progress.section("PLRec")
    RQ, Yt = params['models']['PLRec'](train,
                                       embeded_matrix=np.empty((0)),
                                       iteration=params['iter'],
                                       rank=params['rank'],
                                       lam=params['lambda'])
    Y = Yt.T

    lrec_prediction = predict(matrix_U=RQ, matrix_V=Y, topK=params['topK'][-1], matrix_Train=train, gpu=True)

    lrec_result = evaluate(lrec_prediction, validation, params['metric'], params['topK'])
    print("-")
    print("Rank: {0}".format(params['rank']))
    print("Lambda: {0}".format(params['lambda']))
    print("SVD Iteration: {0}".format(params['iter']))
    print("Evaluation Ranking Topk: {0}".format(params['topK']))
    for key in lrec_result.keys():
        print("{0} :{1}".format(key, lrec_result[key]))

    wlrec_results = dict()
    for alpha in tqdm(params['alphas']):
        progress.section("WPLRec, Alpha: "+str(alpha))
        RQ, Yt = params['models']['WPLRec'](train,
                                            embeded_matrix=np.empty((0)),
                                            iteration=params['iter'],
                                            rank=params['rank'],
                                            lam=params['lambda'],
                                            alpha=alpha)
        Y = Yt.T

        prediction = predict(matrix_U=RQ, matrix_V=Y, topK=params['topK'][-1], matrix_Train=train, gpu=True)

        result = evaluate(prediction, validation, params['metric'], params['topK'])

        wlrec_results[alpha] = result
        print("-")
        print("Alpha: {0}".format(alpha))
        print("Rank: {0}".format(params['rank']))
        print("Lambda: {0}".format(params['lambda']))
        print("SVD Iteration: {0}".format(params['iter']))
        print("Evaluation Ranking Topk: {0}".format(params['topK']))
        for key in result.keys():
            print("{0} :{1}".format(key, result[key]))

    return lrec_result, wlrec_results

