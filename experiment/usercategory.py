import os
import numpy as np
import pandas as pd
from models.predictor import predict
from evaluation.metrics import evaluate
from plots.rec_plots import pandas_bar_plot

def getGroup(user_counts):

    sorted_user_counts = np.sort(user_counts)
    full_length = len(user_counts)
    first_quater = sorted_user_counts[full_length//4]
    median = sorted_user_counts[full_length // 2]
    third_quater =  sorted_user_counts[full_length // 4 * 3]
    patents = [[0, first_quater], [first_quater+1, median], [median+1, third_quater], [third_quater+1, full_length]]
    group = []
    for user_count in user_counts:
        for patent in patents:
            if user_count >= patent[0] and user_count <= patent[1]:
                group.append(str(patent))

    return group


def usercategory(Rtrain, Rvalid, df_input, topK, metric, problem, model_folder, gpu_on=True):

    user_observation_counts = np.array(np.sum(Rtrain, axis=1)).flatten()
    user_observation_counts = user_observation_counts[np.array(np.sum(Rvalid, axis=1)).flatten() != 0]

    index = None
    evaluated_metrics = None

    medians = []
    giant_dataframes = []

    for idx, row in df_input.iterrows():
        row = row.to_dict()

        RQ = np.load('{2}/U_{0}_{1}.npy'.format(row['model'], row['rank'], model_folder))
        Y = np.load('{2}/V_{0}_{1}.npy'.format(row['model'], row['rank'], model_folder))

        if os.path.isfile('{2}/B_{0}_{1}.npy'.format(row['model'], row['rank'], model_folder)):
            Bias = np.load('{2}/B_{0}_{1}.npy'.format(row['model'], row['rank'], model_folder))
        else:
            Bias = None

        prediction = predict(matrix_U=RQ,
                             matrix_V=Y,
                             bias=Bias,
                             topK=topK[-1],
                             matrix_Train=Rtrain,
                             measure=row['similarity'],
                             gpu=gpu_on)

        result = evaluate(prediction, Rvalid, metric, topK, analytical=True)

        df = pd.DataFrame(result)
        df['model'] = row['model']
        df['user_count'] = user_observation_counts

        giant_dataframes.append(df)

        if evaluated_metrics is None:
            evaluated_metrics = result.keys()

    giant_df = pd.concat(giant_dataframes)
    giant_df['group'] = getGroup(giant_df['user_count'].values)

    giant_df = giant_df.sort_values('group', ascending=True).reset_index(drop=True)

    for metric in evaluated_metrics:
        pandas_bar_plot(x='group', y=metric, hue='model', x_name='User Category', y_name=metric, df=giant_df,
                        folder='analysis/{0}/numofrating'.format(problem), name=metric)




