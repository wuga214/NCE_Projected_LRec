import pandas as pd
from utils.io import load_dataframe_csv
from plots.rec_plots import precision_recall_curve

topK = [5, 10, 15, 20, 50]

df = load_dataframe_csv('tables/', 'movielens20m_result.csv')
precision_recall_curve(df, topK, save=True, folder='analysis/'+'movielens20m', reloaded=True)

df = load_dataframe_csv('tables/', 'netflix_result.csv')
precision_recall_curve(df, topK, save=True, folder='analysis/'+'netflix', reloaded=True)

df = load_dataframe_csv('tables/', 'yahoo_result.csv')
precision_recall_curve(df, topK, save=True, folder='analysis/'+'yahoo', reloaded=True)
