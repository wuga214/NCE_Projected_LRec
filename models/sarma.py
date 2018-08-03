import tensorflow as tf
import numpy as np
import math
from tqdm import tqdm
from providers.sequential_split import BatchForSequence
from utils.io import load_pandas, load_numpy
from scipy.sparse import vstack, hstack, lil_matrix


# Under construction...


class SeqAwaRecMetAtt(object):
    def __init__(self,
                 num_items,
                 embed_dim,
                 max_seq_length,
                 label_length = 2,
                 batch_size=100,
                 margin=1.0,
                 cov_loss_weight=0.01,
                 **unused):
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.max_seq_length = max_seq_length
        self.label_length = label_length
        self.batch_size = batch_size
        self.margin = margin
        self.cov_loss_weight = cov_loss_weight
        self.get_graph()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def get_graph(self):

        self.features = tf.placeholder(tf.int32, [None, self.max_seq_length])
        self.pos_label = tf.placeholder(tf.int32, [None, self.label_length])
        self.neg_label = tf.placeholder(tf.int32, [None, self.label_length])
        self.item_embeddings= tf.placeholder(tf.float32, [self.num_items, self.embed_dim])

        with tf.variable_scope("cnn_attention"):
            feature_embedding = tf.nn.embedding_lookup(self.item_embeddings, self.features)
            order_embeddings = tf.Variable(self.get_order_embeddings())
            attention_base = feature_embedding + order_embeddings
            filter = tf.Variable(tf.random_normal([1, self.embed_dim, 1],
                                                  stddev=1 / (self.embed_dim ** 0.5), dtype=tf.float32))
            cnn_out = tf.nn.conv1d(attention_base, filter, stride=1, padding="VALID")
            attention = tf.nn.softmax(cnn_out, axis=1)
            self.pred_embeddings = tf.reduce_sum(tf.multiply(feature_embedding, attention), axis=1)

        with tf.variable_scope("loss"):
            pos_embedding = tf.nn.embedding_lookup(self.item_embeddings, self.pos_label)
            neg_embedding = tf.nn.embedding_lookup(self.item_embeddings, self.neg_label)

            self.loss = tf.reduce_sum(tf.nn.relu(tf.matmul(neg_embedding, tf.expand_dims(self.pred_embeddings, -1))
                                                 - tf.matmul(pos_embedding, tf.expand_dims(self.pred_embeddings, -1))
                                                 - self.margin),
                                      axis=1)

        with tf.variable_scope("optimizer"):
            self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)

    def train_model(self, bs, item_embeddings, epoch=20):
        summary_writer = tf.summary.FileWriter('sarma', graph=self.sess.graph)

        # Training
        pbar = tqdm(range(epoch))
        for i in pbar:
            for batch in bs.next_batch():
                # No item embedding yet, extract from CML?
                feed_dict = {self.features: batch[0],
                             self.pos_label: batch[1],
                             self.neg_label: batch[2],
                             self.item_embeddings: item_embeddings}
                training = self.sess.run([self.optimizer], feed_dict=feed_dict)

    def get_order_embeddings(self):
        pe = np.zeros((self.max_seq_length, self.embed_dim), dtype=np.float32)
        positions = np.arange(0, self.max_seq_length)
        div_term = np.exp(np.arange(0, self.embed_dim, 2) * -(math.log(10000) / self.embed_dim))
        positions = np.expand_dims(positions, axis=1)
        pe[:, 0::2] = np.sin(positions * div_term)
        pe[:, 1::2] = np.cos(positions * div_term)
        return pe

    def get_user_embeddings(self, bs, item_embeddings):
        self.user_embeddings = np.zeros((bs.padding_index, self.embed_dim))
        prediction_counter = np.ones(bs.padding_index)

        for batch in tqdm(bs.next_batch()):
            # No item embedding yet, extract from CML?
            feed_dict = {self.features: batch[0],
                         self.item_embeddings: item_embeddings}
            pred_embeddings = self.sess.run(self.pred_embeddings, feed_dict=feed_dict)

            for i, pred_embedding in enumerate(pred_embeddings):
                user_index = batch[3][i]
                self.user_embeddings[user_index] = self.user_embeddings[user_index] +\
                                                   (pred_embedding - self.user_embeddings[user_index]) /\
                                                   prediction_counter[user_index]
                prediction_counter[user_index] += 1

        return self.user_embeddings


def main():
    matrix_train = load_numpy(path='datax/', name='Rtrain')
    num_item = matrix_train.shape[1]
    model = SeqAwaRecMetAtt(num_item, 100, 7)
    timestamp_matrix = load_pandas(path='datax/', value_name='timestamp', name='ml-10/ratings.csv')
    item_embeddings = np.load('latent/V_CML_100')

    bs = BatchForSequence(matrix_train, timestamp_matrix, 100, 7, 2, 1000)
    bs.generate_feature_labels()

    model.train_model(bs, item_embeddings)
    model.get_user_embeddings(bs, item_embeddings)

if __name__ == "__main__":
    main()
