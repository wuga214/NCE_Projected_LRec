import argparse
from utils.io import load_numpy
from experiment.latent_analysis import latent_analysis
from utils.argument import shape



def main(args):
    R_train = load_numpy(path=args.path, name=args.train)
    latent_analysis(R_train)


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="Function Caller")
    parser.add_argument('-d', dest='path', default="datax/")
    parser.add_argument('-t', dest='train', default='Rtrain.npz')
    parser.add_argument('-v', dest='valid', default='Rvalid.npz')
    parser.add_argument('--shape', help="CSR Shape", dest="shape", type=shape, nargs=2)
    args = parser.parse_args()

    main(args)