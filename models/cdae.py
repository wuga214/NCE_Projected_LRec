import tensorflow as tf
from tensorflow.python.ops import math_ops
import numpy as np
from tqdm import tqdm
from utils.progress import WorkSplitter, inhour
from scipy.sparse import vstack, hstack
from utils.regularizers import Regularizer


class CDAE(object):
    """
        Collaborative Denoiseing Auto-Encoders for Top-N Recommender Systems
    """

    def __init__(self,
                 user_dim,
                 input_dim,
                 embed_dim,
                 batch_size,
                 lamb=0.01,
                 learning_rate=1e-4,
                 optimizer=tf.train.RMSPropOptimizer,
                 **unused):
        self.input_dim = self.output_dim = input_dim
        self.user_dim = user_dim
        self.embed_dim = embed_dim
        self.batch_size = batch_size
        self.lamb = lamb
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.get_graph()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def get_graph(self):
        self.inputs = tf.placeholder(tf.float32, (None, self.input_dim))
        self.user_id = tf.placeholder(tf.int32, [None])
        self.corruption = tf.placeholder(tf.float32)

        with tf.variable_scope('encode'):

            # Normal Encoder Part
            encode_weights = tf.Variable(tf.truncated_normal([self.input_dim, self.embed_dim], stddev=1 / 500.0),
                                         name="Weights")
            encode_bias = tf.Variable(tf.constant(0., shape=[self.embed_dim]), name="Bias")

            # User Representation
            user_embeddings = tf.Variable(tf.truncated_normal([self.user_dim, self.embed_dim], stddev=1 / 500.0),
                                          name="UserEmbeddings")

            users = tf.stop_gradient(tf.nn.embedding_lookup(user_embeddings, self.user_id, name="users"))

            # Corrupt Inputs
            corrupped_inputs = tf.nn.dropout(self.inputs, 1-self.corruption)

            self.encoded = tf.nn.relu(tf.matmul(corrupped_inputs, encode_weights) + users + encode_bias)


        with tf.variable_scope('decode'):
            self.decode_weights = tf.Variable(tf.truncated_normal([self.embed_dim, self.output_dim], stddev=1 / 500.0),
                                              name="Weights")
            self.decode_bias = tf.Variable(tf.constant(0., shape=[self.output_dim]), name="Bias")
            prediction = tf.matmul(self.encoded, self.decode_weights) + self.decode_bias

        with tf.variable_scope('loss'):
            l2_loss = tf.nn.l2_loss(encode_weights) + tf.nn.l2_loss(self.decode_weights) + tf.nn.l2_loss(users)
            sigmoid_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.inputs, logits=prediction)

            mask = tf.stop_gradient(math_ops.floor(corrupped_inputs))
            weighted_sigmoid_loss = mask*(1-self.corruption)*sigmoid_loss + (1-mask)*self.corruption*sigmoid_loss

            self.loss = tf.reduce_mean(weighted_sigmoid_loss) + self.lamb * tf.reduce_mean(l2_loss)

        with tf.variable_scope('optimizer'):
            self.train = self.optimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def get_batches(self, rating_matrix, batch_size):
        remaining_size = rating_matrix.shape[0]
        batch_index=0
        batches = []
        while(remaining_size>0):
            if remaining_size<batch_size:
                batches.append([rating_matrix[batch_index*batch_size:], range(batch_index*batch_size, self.user_dim)])
            else:
                batches.append([rating_matrix[batch_index*batch_size:(batch_index+1)*batch_size],
                               range(batch_index*batch_size, (batch_index+1)*batch_size)])
            batch_index += 1
            remaining_size -= batch_size
        return batches

    def train_model(self, rating_matrix, corruption=0.5, epoch=100, batches=None, **unused):
        if batches is None:
            batches = self.get_batches(rating_matrix, self.batch_size)

        # Training
        pbar = tqdm(range(epoch))
        for i in pbar:
            for step in range(len(batches)):
                feed_dict = {self.inputs: batches[step][0].todense(),
                             self.user_id: batches[step][1],
                             self.corruption: corruption}
                training = self.sess.run([self.train], feed_dict=feed_dict)

    def get_RQ(self, rating_matrix):
        batches = self.get_batches(rating_matrix, self.batch_size)
        RQ = []
        for step in range(len(batches)):
            feed_dict = {self.inputs: batches[step][0].todense(),
                         self.user_id: batches[step][1],
                         self.corruption: 0.}
            encoded = self.sess.run(self.encoded, feed_dict=feed_dict)
            RQ.append(encoded)

        return np.vstack(RQ)

    def get_Y(self):
        return self.sess.run(self.decode_weights)

    def get_Bias(self):
        return self.sess.run(self.decode_bias)


def cdae(matrix_train, embeded_matrix=np.empty((0)), iteration=100, lam=80, rank=200,
         corruption=0.5, optimizer='RMSProp', seed=1, **unused):
    progress = WorkSplitter()
    matrix_input = matrix_train
    if embeded_matrix.shape[0] > 0:
        matrix_input = vstack((matrix_input, embeded_matrix.T))

    m, n = matrix_input.shape
    model = CDAE(m, n, rank, 100, lamb=lam, optimizer=Regularizer[optimizer])

    model.train_model(matrix_input, corruption, iteration)

    RQ = model.get_RQ(matrix_input)
    Y = model.get_Y()
    Bias = model.get_Bias()
    model.sess.close()
    tf.reset_default_graph()

    return RQ, Y, Bias