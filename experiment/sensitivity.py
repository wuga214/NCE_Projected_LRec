import numpy as np
from tqdm import tqdm
from evaluation.metrics import evaluate
from models.predictor import predict
from utils.progress import WorkSplitter, inhour


def sensitivity(train, validation, params):
    progress = WorkSplitter()
    progress.section("PMI-PLRec Default")
    RQ, Yt,_ = params['models']['PmiPLRec'](train,
                                            embeded_matrix=np.empty((0)),
                                            iteration=params['iter'],
                                            rank=params['rank'],
                                            lam=params['lambda'],
                                            root=1.0)
    Y = Yt.T

    default_prediction = predict(matrix_U=RQ, matrix_V=Y, topK=params['topK'][-1], matrix_Train=train, gpu=True)

    default_result = evaluate(default_prediction, validation, params['metric'], params['topK'])
    print("-")
    print("Rank: {0}".format(params['rank']))
    print("Lambda: {0}".format(params['lambda']))
    print("SVD Iteration: {0}".format(params['iter']))
    print("Evaluation Ranking Topk: {0}".format(params['topK']))
    for key in default_result.keys():
        print("{0} :{1}".format(key, default_result[key]))

    sensitivity_results = dict()
    for root in tqdm(params['root']):
        progress.section("PMI-PLRec, Root: "+str(root))
        RQ, Yt,_ = params['models']['PmiPLRec'](train,
                                                embeded_matrix=np.empty((0)),
                                                iteration=params['iter'],
                                                rank=params['rank'],
                                                lam=params['lambda'],
                                                root=root)
        Y = Yt.T

        prediction = predict(matrix_U=RQ, matrix_V=Y, topK=params['topK'][-1], matrix_Train=train, gpu=True)

        result = evaluate(prediction, validation, params['metric'], params['topK'])

        sensitivity_results[root] = result
        print("-")
        print("Root: {0}".format(root))
        print("Rank: {0}".format(params['rank']))
        print("Lambda: {0}".format(params['lambda']))
        print("SVD Iteration: {0}".format(params['iter']))
        print("Evaluation Ranking Topk: {0}".format(params['topK']))
        for key in result.keys():
            print("{0} :{1}".format(key, result[key]))

    return default_result, sensitivity_results

