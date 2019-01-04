from models.pop import pop
from models.cdae import cdae, CDAE
from models.vae import vae_cf, VAE
from models.autorec import autorec, AutoRec
from models.bpr import bpr
from models.wrmf import als
from models.cml import cml
from models.puresvd import puresvd
from models.nceplrec import nceplrec
from models.plrec import plrec
from models.ncesvd import ncesvd

models = {
    "POP": pop,
    "AutoRec": autorec,
    "CDAE": cdae,
    "VAE-CF": vae_cf,
    "BPR": bpr,
    "WRMF": als,
    "CML": cml,
    "PureSVD": puresvd,
    "NCE-PLRec": nceplrec,
    "NCE-SVD": ncesvd,
    "PLRec": plrec,
}

autoencoders = {
    "AutoRec": AutoRec,
    "CDAE": CDAE,
    "VAE-CF": VAE,
}

vaes = {
    "VAE-CF": VAE,
}