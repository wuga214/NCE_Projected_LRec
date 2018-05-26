import numpy as np
from tqdm import tqdm
from evaluation.metrics import evaluate


def weighting(train, validation, params):
    RQ, Yt = params['models']['PLRec'](train,
                                       embeded_matrix=np.empty((0)),
                                       iteration=params['iter'],
                                       rank=params['rank'],
                                       lam=params['lambda'])
    Y = Yt.T

    lrec_result = evaluate(RQ, Y, train, validation, params['topK'], params['metric'])
    print("-")
    print("Rank: {0}".format(params['rank']))
    print("Lambda: {0}".format(params['lambda']))
    print("SVD Iteration: {0}".format(params['iter']))
    print("Evaluation Ranking Topk: {0}".format(params['topK']))
    for key in lrec_result.keys():
        print("{0} :{1}".format(key, lrec_result[key]))

    wlrec_results = dict()
    for alpha in tqdm(params['alphas']):
        RQ, Yt = params['models']['WPLRec'](train,
                                            embeded_matrix=np.empty((0)),
                                            iteration=params['iter'],
                                            rank=params['rank'],
                                            lam=params['lambda'],
                                            alpha=alpha)
        Y = Yt.T

        result = evaluate(RQ, Y, train, validation, params['topK'], params['metric'])
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

