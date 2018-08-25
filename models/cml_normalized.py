import tensorflow as tf
import numpy as np
from tqdm import tqdm
from utils.progress import WorkSplitter, inhour
from scipy.sparse import vstack, hstack, lil_matrix
from scipy.stats import rankdata
import pyximport; pyximport.install()
from utils.cython.negative_sampler import get_negative_samples


# Under construction...


class NormalizedCollaborativeMetricLearning(object):
    def __init__(self,
                 num_users,
                 num_items,
                 embed_dim,
                 batch_size=10000,
                 margin=1.0,
                 clip_norm=1.0,
                 cov_loss_weight=0.01,
                 **unused):
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.batch_size = batch_size
        self.margin = margin
        self.clip_norm = clip_norm
        self.cov_loss_weight = cov_loss_weight
        self.get_graph()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def get_graph(self):

        # Placehoders

        # M users
        self.pos_sample_idx = tf.placeholder(tf.int32, [None])

        # M positive items
        self.neg_sample_idx = tf.placeholder(tf.int32, [None, None])

        # M X N positive items
        self.user_idx = tf.placeholder(tf.int32, [None])

        self.orders = tf.placeholder(tf.float32, [None])

        self.idf = tf.placeholder(tf.float32, [None])

        # Variable to learn
        self.user_embeddings = tf.Variable(tf.random_normal([self.num_users, self.embed_dim],
                                                            stddev=1 / (self.embed_dim ** 0.5), dtype=tf.float32))

        self.item_embeddings = tf.Variable(tf.random_normal([self.num_items, self.embed_dim],
                                                            stddev=1 / (self.embed_dim ** 0.5), dtype=tf.float32))

        # idf lookup
        popularity_weights = tf.nn.embedding_lookup(self.idf, self.pos_sample_idx, name="users")

        with tf.variable_scope("covariance_loss"):
            embedding = tf.concat((self.item_embeddings, self.user_embeddings), 0)
            n_rows = tf.cast(tf.shape(embedding)[0], tf.float32)
            X = embedding - (tf.reduce_mean(embedding, axis=0))
            cov = tf.matmul(X, X, transpose_a=True) / n_rows
            cov_loss = tf.reduce_sum(tf.matrix_set_diag(cov, tf.zeros(self.embed_dim, tf.float32))
                                     ) * self.cov_loss_weight

        with tf.variable_scope("metric_loss"):
            users = tf.nn.embedding_lookup(self.user_embeddings, self.user_idx, name="users")
            pos_samples = tf.nn.embedding_lookup(self.item_embeddings, self.pos_sample_idx, name="pos_items")
            neg_samples = tf.transpose(tf.nn.embedding_lookup(self.item_embeddings, self.neg_sample_idx),
                                       (0, 2, 1), name="neg_items")
            pos_distances = tf.reduce_sum(tf.squared_difference(users, pos_samples),
                                          axis=1,
                                          name="pos_distances")
            neg_distances = tf.reduce_sum(tf.squared_difference(tf.expand_dims(users, -1), neg_samples),
                                          axis=1,
                                          name="neg_distances")
            shortest_neg_distances = tf.reduce_min(neg_distances, 1, name="shortest_neg_distances")
            hinge_loss = tf.maximum(pos_distances - shortest_neg_distances + self.margin, 0, name="pair_loss")
            impostors = (tf.expand_dims(pos_distances, -1) - neg_distances + self.margin) > 0
            rank = tf.reduce_mean(tf.cast(impostors, dtype=tf.float32), 1, name="rank_weight") * self.num_items
            metric_loss = hinge_loss * self.orders# * tf.log(rank + 1) # * popularity_weights

        self.loss = metric_loss + cov_loss

        with tf.variable_scope("optimizer"):
            self.optimizer = tf.train.AdamOptimizer().minimize(self.loss,
                                                               var_list=[self.user_embeddings,
                                                                         self.item_embeddings])

        with tf.variable_scope("clip"):
            self.clips = [
                tf.assign(self.user_embeddings, tf.clip_by_norm(self.user_embeddings, self.clip_norm, axes=[1])),
                tf.assign(self.item_embeddings, tf.clip_by_norm(self.item_embeddings, self.clip_norm, axes=[1]))
            ]

    def train_model(self, rating_matrix, orders, epoch=100):

        n_negative = 10

        item_popularity = np.array(np.sum(rating_matrix, axis=0)).flatten()

        idf = np.log(item_popularity+1.0)

        user_item_matrix = lil_matrix(rating_matrix)
        user_item_pairs = np.asarray(user_item_matrix.nonzero()).T
        user_to_positive_set = {u: set(row) for u, row in enumerate(user_item_matrix.rows)}

        summary_writer = tf.summary.FileWriter('ncml', graph=self.sess.graph)

        # Training
        for i in range(epoch):

            batches = self.get_batches(user_item_pairs, user_to_positive_set, orders,
                                       user_item_matrix.shape[1], self.batch_size, n_negative)

            for step in tqdm(range(len(batches))):
                feed_dict = {self.user_idx: batches[step][0],
                             self.pos_sample_idx: batches[step][1],
                             self.neg_sample_idx: batches[step][2],
                             self.orders: batches[step][3],
                             self.idf: idf
                             }
                training = self.sess.run([self.optimizer], feed_dict=feed_dict)
                clip = self.sess.run(self.clips)

    @staticmethod
    def get_batches(user_item_pairs, user_to_positive_set, orders, num_item, batch_size, n_negative):
        batches = []

        index_shuf = range(len(user_item_pairs))
        np.random.shuffle(index_shuf)
        user_item_pairs = user_item_pairs[index_shuf]
        orders = orders[index_shuf]
        for i in tqdm(range(int(len(user_item_pairs) / batch_size))):

            ui_pairs = user_item_pairs[i * batch_size: (i + 1) * batch_size, :]
            ui_order = orders[i * batch_size: (i + 1) * batch_size]

            negative_samples = np.random.randint(
                0,
                num_item,
                size=(batch_size, n_negative))

            negative_samples = get_negative_samples(user_to_positive_set, ui_pairs[:, 0], num_item, negative_samples)
            batches.append([ui_pairs[:, 0], ui_pairs[:, 1], negative_samples, ui_order])

        return batches


    def get_RQ(self):
        return self.sess.run(self.user_embeddings)

    def get_Y(self):
        return self.sess.run(self.item_embeddings)


def get_orders(time_stamp_matrix, threshold=10):
    orders = []

    for row in tqdm(time_stamp_matrix):
        if row.nnz < threshold:
            orders.append(rankdata(row.data)+1)
        else:
            orders.append(np.maximum(rankdata(row.data) + 1 - (row.nnz-threshold), 1))

    return np.hstack(orders)


def cml_normalized(matrix_train, time_stamp_matrix=None, embeded_matrix=np.empty((0)),
                   iteration=100, lam=80, rank=200, seed=1, **unused):
    progress = WorkSplitter()
    matrix_input = matrix_train

    from utils.io import load_numpy
    time_stamp_matrix = load_numpy(path='datax/', name='Rtime.npz')
    orders = get_orders(time_stamp_matrix.multiply(matrix_train))

    if embeded_matrix.shape[0] > 0:
        matrix_input = vstack((matrix_input, embeded_matrix.T))

    m, n = matrix_input.shape
    model = NormalizedCollaborativeMetricLearning(num_users=m, num_items=n, embed_dim=rank, cov_loss_weight=lam)

    model.train_model(matrix_input, orders, iteration)

    RQ = model.get_RQ()
    Y = model.get_Y().T
    tf.reset_default_graph()
    return RQ, Y, None