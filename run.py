import argparse
from utils.io import load_numpy
import numpy as np
from experiment.latent_analysis import latent_analysis
from experiment.popular_analysis import popular_overlapping
from utils.argument import shape
from utils.argument import check_float_positive, check_int_positive, shape

from models.lrec import embedded_lrec_items
from models.weighted_lrec import weighted_lrec_items
from models.pure_svd import pure_svd, eigen_boosted_pure_svd
from models.als import als
from models.pmi_lrec import pmi_lrec_items
from models.weighted_pmi_lrec import weighted_pmi_lrec_items
from models.chainitemitem import chain_item_item
from models.predictor import predict


models = {
    "PLRec": embedded_lrec_items,
    "WPLRec": weighted_lrec_items,
    "PmiPLRec": pmi_lrec_items,
    "PmiWPLRec": weighted_pmi_lrec_items,
    "PureSVD": pure_svd,
    "EBPureSVD": eigen_boosted_pure_svd,
    "ALS": als,
    "CII": chain_item_item,
}



def main(args):
    R_train = load_numpy(path=args.path, name=args.train)
    #latent_analysis(R_train)

    RQ, Yt = models['PLRec'](R_train, embeded_matrix=np.empty((0)),
                             iteration=args.iter, rank=args.rank,
                             lam=args.lamb, alpha=args.alpha, seed=args.seed, root=args.root)

    Y = Yt.T

    prediction = predict(matrix_U=RQ, matrix_V=Y, topK=10, matrix_Train=R_train, gpu=True)

    popular_overlapping(R_train, prediction)

if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="Function Caller")
    parser.add_argument('-d', dest='path', default="datax/")
    parser.add_argument('-t', dest='train', default='Rtrain.npz')
    parser.add_argument('-v', dest='valid', default='Rvalid.npz')
    parser.add_argument('--shape', help="CSR Shape", dest="shape", type=shape, nargs=2)
    parser.add_argument('-i', dest='iter', type=check_int_positive, default=4)
    parser.add_argument('-a', dest='alpha', type=check_float_positive, default=1.0)
    parser.add_argument('-l', dest='lamb', type=check_float_positive, default=1.0)
    parser.add_argument('-r', dest='rank', type=check_int_positive, default=100)
    parser.add_argument('-f', dest='root', type=check_float_positive, default=1)
    parser.add_argument('-s', dest='seed', type=check_int_positive, default=1)
    args = parser.parse_args()

    main(args)