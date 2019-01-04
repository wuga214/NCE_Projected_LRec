import tensorflow as tf
import numpy as np
from tqdm import tqdm
from utils.progress import WorkSplitter
from scipy.sparse import vstack, lil_matrix


class BPR(object):
    def __init__(self,
                 num_users,
                 num_items,
                 embed_dim,
                 lamb,
                 batch_size=100,
                 uniform_sample=False,
                 **unused):
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.lamb = lamb
        self.batch_size = batch_size
        self.uniform = uniform_sample
        self.get_graph()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())


    def get_graph(self):

        # Placehoder
        self.user_idx = tf.placeholder(tf.int32, [None])
        self.item_idx_i = tf.placeholder(tf.int32, [None])
        self.item_idx_j = tf.placeholder(tf.int32, [None])
        self.label = tf.placeholder(tf.float32, [None])

        # Variable to learn
        self.user_embeddings = tf.Variable(tf.random_normal([self.num_users, self.embed_dim],
                                                            stddev=1 / (self.embed_dim ** 0.5), dtype=tf.float32))
        self.item_embeddings = tf.Variable(tf.random_normal([self.num_items, self.embed_dim],
                                                            stddev=1 / (self.embed_dim ** 0.5), dtype=tf.float32))

        with tf.variable_scope("bpr_loss"):
            users = tf.nn.embedding_lookup(self.user_embeddings, self.user_idx, name="users")
            item_i = tf.nn.embedding_lookup(self.item_embeddings, self.item_idx_i, name="item_i")
            item_j = tf.nn.embedding_lookup(self.item_embeddings, self.item_idx_j, name="item_j")

            x_uij = tf.reduce_sum(tf.multiply(users,
                                              item_i,
                                              name='x_ui'),
                                  axis=1) - tf.reduce_sum(tf.multiply(users,
                                                                      item_j,
                                                                      name='x_uj'),
                                                          axis=1)

            if self.uniform:
                bpr_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=x_uij, labels=self.label))
            else:
                bpr_loss = -tf.reduce_mean(tf.log(tf.nn.sigmoid(x_uij)))

        with tf.variable_scope('l2_loss'):
            unique_user_idx, _ = tf.unique(self.user_idx)
            unique_users = tf.nn.embedding_lookup(self.user_embeddings, unique_user_idx)

            unique_item_idx, _ = tf.unique(tf.concat([self.item_idx_i, self.item_idx_j], 0))
            unique_items = tf.nn.embedding_lookup(self.item_embeddings, unique_item_idx)

            l2_loss = tf.reduce_mean(tf.nn.l2_loss(unique_users)) + tf.reduce_mean(tf.nn.l2_loss(unique_items))

        with tf.variable_scope('loss'):
            self.loss = bpr_loss + self.lamb * l2_loss


        with tf.variable_scope('optimizer'):
            self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)

    def get_uniform_batches(self, rating_matrix, batch_size):
        batches = []

        for i in tqdm(range(int(self.num_users / batch_size))):
            user_idx = np.random.choice(self.num_users, batch_size)
            item_idx_i = np.random.choice(self.num_items, batch_size)
            item_idx_j = np.random.choice(self.num_items, batch_size)

            label = np.max(np.asarray(rating_matrix[user_idx, item_idx_i] - rating_matrix[user_idx, item_idx_j]), 0)

            batches.append([user_idx, item_idx_i, item_idx_j, label])

        return batches

    @staticmethod
    def get_batches(user_item_pairs, rating_matrix, num_item, batch_size):
        batches = []

        index_shuf = range(len(user_item_pairs))
        np.random.shuffle(index_shuf)
        user_item_pairs = user_item_pairs[index_shuf]
        for i in tqdm(range(int(len(user_item_pairs) / batch_size))):

            ui_pairs = user_item_pairs[i * batch_size: (i + 1) * batch_size, :]

            negative_samples = np.random.randint(
                0,
                num_item,
                size=batch_size)

            label = np.max(np.asarray(rating_matrix[ui_pairs[:, 0], ui_pairs[:, 1]] - rating_matrix[ui_pairs[:, 0],
                                                                                                    negative_samples]),
                           0)

            batches.append([ui_pairs[:, 0], ui_pairs[:, 1], negative_samples, label])

        return batches


    def train_model(self, rating_matrix, epoch=100):

        if not self.uniform:
            user_item_matrix = lil_matrix(rating_matrix)
            user_item_pairs = np.asarray(user_item_matrix.nonzero()).T

        # Training
        for i in tqdm(range(epoch)):
            if self.uniform:
                batches = self.get_uniform_batches(rating_matrix, self.batch_size)
            else:
                batches = self.get_batches(user_item_pairs, rating_matrix, self.num_items, self.batch_size)
            for step in range(len(batches)):
                feed_dict = {self.user_idx: batches[step][0],
                             self.item_idx_i: batches[step][1],
                             self.item_idx_j: batches[step][2],
                             self.label: batches[step][3]
                             }
                training = self.sess.run([self.optimizer], feed_dict=feed_dict)

    def get_RQ(self):
        return self.sess.run(self.user_embeddings)

    def get_Y(self):
        return self.sess.run(self.item_embeddings)


def bpr(matrix_train, embeded_matrix=np.empty((0)), iteration=100, lam=80, rank=200, seed=1, **unused):
    progress = WorkSplitter()
    matrix_input = matrix_train
    if embeded_matrix.shape[0] > 0:
        matrix_input = vstack((matrix_input, embeded_matrix.T))

    m, n = matrix_input.shape
    model = BPR(m, n, rank, lamb=lam, batch_size=500)
    model.train_model(matrix_input, iteration)

    RQ = model.get_RQ()
    Y = model.get_Y().T
    model.sess.close()
    tf.reset_default_graph()

    return RQ, Y, None