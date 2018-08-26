import numpy as np
from sklearn.manifold import TSNE
from models.lrec import embedded_lrec_items
from models.weighted_lrec import weighted_lrec_items
from models.pure_svd import pure_svd, eigen_boosted_pure_svd
from models.als import als
from models.pmi_lrec import pmi_lrec_items
from models.chainitemitem import chain_item_item
from models.predictor import predict
from plot.plot import scatter_plot
from sklearn.preprocessing import normalize
import plotly.plotly as py
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go

params = {
    'models': {#"PLR": embedded_lrec_items,
               "PmiPLRec": pmi_lrec_items,
               #"ALS": als
               },
    'alphas': 1,
    'rank': 50,
    'lambda': 100,
    'topK': 50,
    'iter': 7,
    'metric': ['R-Precision', 'NDCG'],
}


def latent_analysis(matrix):
    item_popularity = np.array(np.sum(matrix, axis=0)).flatten()

    # Item-Item or User-User
    for model_name in params['models'].keys():
        RQ, Yt, _ = params['models'][model_name](matrix, embeded_matrix=np.empty((0)),
                                                 root=2,
                                                 iteration=params['iter'], rank=params['rank'],
                                                 lam=params['lambda'], alpha=params['alphas'], seed=1)

        scatter_plot(100*Yt.T[:,:2], item_popularity, model_name, save=True)









