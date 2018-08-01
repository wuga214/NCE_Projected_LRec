import tensorflow as tf
import numpy as np
import math
from tqdm import tqdm
from scipy.sparse import vstack, hstack, lil_matrix


# Under construction...


class SeqAwaRecMetAtt(object):
    def __init__(self,
                 num_users,
                 num_items,
                 embed_dim,
                 max_seq_length,
                 batch_size=100,
                 margin=1.0,
                 cov_loss_weight=0.01,
                 **unused):
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.margin = margin
        self.cov_loss_weight = cov_loss_weight
        self.get_graph()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def get_graph(self):

        self.purchase = tf.placeholder(tf.int32)
        self.item_embeddings= tf.placeholder(tf.float32, [self.num_items, self.embed_dim])
        purchase_embedding = tf.nn.embedding_lookup(self.item_embeddings, self.purchase)

        self.purchase_range = tf.placeholder(tf.int32)
        self.order_embeddings = tf.Variable(self.get_order_embeddings())
        purchase_range_embedding = tf.nn.embedding_lookup(self.order_embeddings, self.purchase_range)

        self.attention_base = purchase_embedding + purchase_range_embedding

        filter = tf.Variable(tf.random_normal([1, self.embed_dim, 1],
                                              stddev=1 / (self.embed_dim ** 0.5), dtype=tf.float32))
        self.output = tf.nn.conv1d(self.attention_base, filter, stride=1, padding="VALID")

        self.attention = tf.nn.softmax(self.output, axis=1)

        self.user_embeddings = tf.reduce_sum(tf.multiply(purchase_embedding, self.attention), axis=1)


    def get_order_embeddings(self):
        pe = np.zeros((self.max_seq_length, self.embed_dim), dtype=np.float32)
        positions = np.arange(0, self.max_seq_length)
        div_term = np.exp(np.arange(0, self.embed_dim, 2) * -(math.log(10000) / self.embed_dim))
        positions = np.expand_dims(positions, axis=1)
        pe[:, 0::2] = np.sin(positions * div_term)
        pe[:, 1::2] = np.cos(positions * div_term)
        return pe








    #     pass
    #
    # @staticmethod
    # def get_batches(rating_matrix, batch_size):
    #     remaining_size = rating_matrix.shape[0]
    #     batch_index=0
    #     batches = []
    #     while(remaining_size>0):
    #         if remaining_size<batch_size:
    #             batches.append(rating_matrix[batch_index*batch_size:-1])
    #         else:
    #             batches.append(rating_matrix[batch_index*batch_size:(batch_index+1)*batch_size])
    #         batch_index += 1
    #         remaining_size -= batch_size
    #     return batches
    #
    # def get_RQ(self):
    #     pass
    #
    # def get_Y(self):
    #     pass


def main():
    rating = np.array([[0, 3, 10], [1, 7, 9]])
    embedding = np.random.rand(10, 100)
    model = SeqAwaRecMetAtt(2, 10, 100, 5, 2)

    #rating_range = np.array([range(len(x)) for x in rating])
    rating_range = np.array([[0, 1, 5], [0, 1, 2]])

    feed = {model.purchase: rating, model.item_embeddings: embedding, model.purchase_range: rating_range}
    import ipdb; ipdb.set_trace()

    model.sess.run(model.output, feed_dict=feed)


if __name__ == "__main__":
    main()
