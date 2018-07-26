import tensorflow as tf
import numpy as np
from tqdm import tqdm
from utils.progress import WorkSplitter, inhour
from scipy.sparse import vstack, hstack

# Instruction about feeding sparse placeholder

# X = tf.Variable(tf.truncated_normal([500, 25], stddev=1/500.0))
# sp_indices = tf.placeholder(tf.int64)
# sp_shape = tf.placeholder(tf.int64)
# sp_ids_val = tf.placeholder(tf.int64)
# sp_weights_val = tf.placeholder(tf.float32)
# sp_ids = tf.SparseTensor(sp_indices, sp_ids_val, sp_shape)
# sp_weights = tf.SparseTensor(sp_indices, sp_weights_val, sp_shape)
# y = tf.nn.embedding_lookup_sparse(X, sp_ids, sp_weights, "sum")
# tf.initialize_all_variables().run()  # initialize values in X
#
# y_values = tf.run(y, feed_dict={
#   sp_indices: [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1]],  # 3 entries in minibatch entry 0, 2 entries in entry 1.
#   sp_shape: [2, 3],  # batch size: 2, max index: 2 (so index count == 3)
#   sp_ids_val: [53, 87, 101, 34, 98],
#   sp_weights_val: [0.1, 0.2, 0.3, -1.0, 3.5]})


class AutoRec(object):
    def __init__(self,
                 input_dim,
                 embed_dim,
                 batch_size,
                 lamb=0.01,
                 **unused):
        self.input_dim = self.output_dim = input_dim
        self.embed_dim = embed_dim
        self.batch_size = batch_size
        self.lamb = lamb
        self.get_graph()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def get_graph(self):
        self.inputs = tf.placeholder(tf.float32, (None, self.input_dim))

        with tf.variable_scope('encode'):
            encode_weights = tf.Variable(tf.truncated_normal([self.input_dim, self.embed_dim], stddev=1 / 500.0),
                                         name="Weights")
            encode_bias = tf.Variable(tf.constant(0., shape=[self.embed_dim]), name="Bias")

            self.encoded = tf.nn.sigmoid(tf.matmul(self.inputs, encode_weights) + encode_bias)

        with tf.variable_scope('decode'):
            self.decode_weights = tf.Variable(tf.truncated_normal([self.embed_dim, self.output_dim], stddev=1 / 500.0),
                                     name="Weights")
            #decode_bias = tf.Variable(tf.constant(0., shape=[self.output_dim]), name="Bias")
            prediction = tf.matmul(self.encoded, self.decode_weights)# + decode_bias

        with tf.variable_scope('loss'):
            l2_loss = tf.nn.l2_loss(encode_weights) + tf.nn.l2_loss(self.decode_weights)
            sigmoid_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.inputs, logits=prediction)
            self.loss = sigmoid_loss + self.lamb*l2_loss

        with tf.variable_scope('optimizer'):
            self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)

    def get_batches(self, rating_matrix, batch_size):
        remaining_size = rating_matrix.shape[0]
        batch_index=0
        batches = []
        while(remaining_size>0):
            if remaining_size<batch_size:
                batches.append(rating_matrix[batch_index*batch_size:-1])
            else:
                batches.append(rating_matrix[batch_index*batch_size:(batch_index+1)*batch_size])
            batch_index += 1
            remaining_size -= batch_size
        return batches

    def train_model(self, rating_matrix, epoch=100):
        batches = self.get_batches(rating_matrix, self.batch_size)

        summary_writer = tf.summary.FileWriter('auto_rec', graph=self.sess.graph)

        # Training
        pbar = tqdm(range(epoch))
        for epoch in pbar:
            for step in range(len(batches)):
                feed_dict = {self.inputs: batches[step].todense()}
                training = self.sess.run([self.optimizer], feed_dict=feed_dict)

    def get_RQ(self, rating_matrix):
        batches = self.get_batches(rating_matrix, self.batch_size)
        RQ = []
        for step in range(len(batches)):
            feed_dict = {self.inputs: batches[step].todense()}
            encoded = self.sess.run(self.encoded, feed_dict=feed_dict)
            RQ.append(encoded)

        return np.vstack(RQ)

    def get_Y(self):
        return self.sess.run(self.decode_weights)


def autorec(matrix_train, embeded_matrix=np.empty((0)), iteration=100, lam=80, rank=200, seed=1, **unused):
    progress = WorkSplitter()
    matrix_input = matrix_train
    if embeded_matrix.shape[0] > 0:
        matrix_input = vstack((matrix_input, embeded_matrix.T))

    m, n = matrix_input.shape
    model = AutoRec(n, rank, 100, lamb=lam)

    model.train_model(matrix_input, iteration)

    RQ = model.get_RQ(matrix_input)
    Y = model.get_Y()

    return RQ, Y


























