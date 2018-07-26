import tensorflow as tf
import numpy as np
from tqdm import tqdm
from utils.progress import WorkSplitter, inhour
from scipy.sparse import vstack, hstack


### Under construction...


class CollaborativeMetricLearning(object):
    def __init__(self,
                 num_users,
                 num_items,
                 embed_dim,
                 clip_norm=1.0,
                 lamb=0.01,
                 **unused):
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.clip_norm = clip_norm
        self.lamb = lamb
        self.get_graph()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def get_graph(self):
        user_embedding = tf.Variable(tf.random_normal([self.num_users, self.embed_dim],
                                                      stddev=1 / (self.embed_dim ** 0.5), dtype=tf.float32))

        item_embedding = tf.Variable(tf.random_normal([self.num_users, self.embed_dim],
                                                      stddev=1 / (self.embed_dim ** 0.5), dtype=tf.float32))



        pass

    def get_batches(self, rating_matrix, batch_size):
        pass

    def get_RQ(self, rating_matrix):
        pass

    def get_Y(self):
        pass

    def get_Bias(self):
        pass