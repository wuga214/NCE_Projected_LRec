import numpy as np
import cupy as cp
import scipy
import scipy.sparse as sparse
from scipy.sparse import vstack, hstack
from cupy.linalg import inv
from sklearn.utils.extmath import randomized_svd
from utils.progress import WorkSplitter, inhour
from tqdm import tqdm
import time
from functools import partial
from multiprocessing import Pool
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.mllib.linalg import Vectors



def per_item(vector_r, matrix_A, matrix_B, matrix_BT, alpha):
    vector_r = vector_r.ravel()
    vector_c = alpha * vector_r
    denominator = inv(matrix_A+(matrix_BT*vector_c).dot(matrix_B))
    # Change to return if pool
    yield (denominator.dot(matrix_BT)).dot(vector_c*vector_r+vector_r)

def weighted_lrec_items(matrix_train, embeded_matrix=np.empty((0)), iteration=4, lam=80, rank=200, alpha=100):
    """
    Function used to achieve generalized projected lrec w/o item-attribute embedding
    :param matrix_train: user-item matrix with shape m*n
    :param embeded_matrix: item-attribute matrix with length n (each row represents one item)
    :param lam: parameter of penalty
    :param k_factor: ratio of the latent dimension/number of items
    :return: prediction in sparse matrix
    """
    progress = WorkSplitter()
    matrix_input = matrix_train
    if embeded_matrix.shape[0] > 0:
        matrix_input = vstack((matrix_input, embeded_matrix.T))

    progress.section("Randomized SVD")
    start_time = time.time()
    P, sigma, Qt = randomized_svd(matrix_input, n_components=rank, n_iter=iteration, random_state=None)
    print "Elapsed: {0}".format(inhour(time.time() - start_time))

    start_time = time.time()
    progress.section("Create Cacheable Matrices")
    RQ = matrix_input * Qt.T
    matrix_A = sparse.diags(sigma*sigma-lam)#Sigma.T.dot(Sigma) - lam*sparse.identity(rank)
    matrix_B = cp.array(P*sigma)
    matrix_BT = cp.array(matrix_B.T)
    print "Elapsed: {0}".format(inhour(time.time() - start_time))


    progress.section("Item-wised Optimization")
    start_time = time.time()

    # For loop
    m, n = matrix_train.shape
    gpu_memory = cp.get_default_memory_pool()
    Y = []
    for i in tqdm(xrange(n)): #change back to n!!!
        vector_r = matrix_train[:, i].toarray()
        vector_r = cp.array(vector_r)
        vector_y = per_item(vector_r, matrix_A, matrix_B, matrix_BT, alpha)
        y_i_gpu = cp.asnumpy(vector_y)
        y_i_cpu = np.copy(y_i_gpu)
        Y = Y+y_i_cpu
    Y = scipy.vstack(Y)


    # Multi Threads Method
    # pool = Pool(processes=30)
    #
    # count = 0
    # batch_size = 100
    #
    # Y = []
    # for i in tqdm(xrange(n/batch_size+1)):
    #
    #     Y_i = pool.map(partial(per_item,
    #                            matrix_A=matrix_A,
    #                            matrix_B=matrix_B,
    #                            matrix_BT=matrix_BT,
    #                            alpha=alpha),
    #                    matrix_train.T[i*batch_size:(i+1)*batch_size].toarray(),
    #                    chunksize=1)
    #     Y = Y+Y_i


    # Spark does not work
    # sc = SparkContext()
    # sc.setSystemProperty('spark.executor.memory', '200g')
    # spark = SparkSession(sc)
    # import ipdb; ipdb.set_trace()
    # spark_df = spark.createDataFrame(map(lambda x: (Vectors.dense(x),), matrix_train.T.toarray()), schema=["Yeah"])



    return RQ, Y