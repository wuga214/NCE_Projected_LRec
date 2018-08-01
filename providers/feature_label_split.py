# Target:
# A split function used to split training data(unsupervised) into feature and label for the
# Sequential prediction(supervised).

# The function should do
# 1. Time based leave one out split
# 2. Dynamic split and yield training batch

# this function only support implicit feedback!

import scipy.sparse as sparse
import numpy as np


class BatchForSequence(object):
    def __init__(self, rating_matrix, batch_size, feature_length, label_length, neg_label_length):
        self.padding_index = rating_matrix.shape[0]
        self.batch_size = batch_size
        self.feature_length = feature_length
        self.label_length = label_length
        self.max_length = feature_length + label_length
        self.neg_label_length = neg_label_length
        self.rating_lists = sparse.lil_matrix(rating_matrix).rows
        self.lengths = np.vectorize(len)(self.rating_lists)


    def generate_feature_labels(self):

        num_subsequences = sum([c - self.max_length + 1 if c >= self.max_length else 1 for c in self.lengths])

        self.features = np.full((num_subsequences, self.feature_length),
                                dtype=np.int64,
                                fill_value=self.padding_index)
        self.pos_labels = np.full((num_subsequences, self.label_length),
                                  dtype=np.int64,
                                  fill_value=self.padding_index)
        self.neg_labels = np.full((num_subsequences, self.neg_label_length),
                                  dtype=np.int64,
                                  fill_value=self.padding_index)


        seq_index = 0

        for user_index in range(len(self.rating_lists)):
            for seq in self._sliding_window(user_index):
                #import ipdb; ipdb.set_trace()
                if len(seq) > self.label_length:
                    features = seq[:-self.label_length]
                    self.features[seq_index][self.feature_length-len(features):] = features

                    self.pos_labels[seq_index][:] = seq[-self.label_length:]
                    seq_index += 1

        self.features = self.features[:seq_index]
        self.pos_labels = self.pos_labels[:seq_index]
        self.neg_labels = self.neg_labels[:seq_index]

        import ipdb; ipdb.set_trace()



    def _sliding_window(self, user_index, step_size=1):
        if self.lengths[user_index] - self.max_length >= 0:
            for i in range(self.lengths[user_index], 0, -step_size):
                if i - self.max_length >= 0:
                    yield self.rating_lists[user_index][i - self.max_length:i]
                else:
                    break
        else:
            yield self.rating_lists[user_index]


def main():
    x = sparse.random(100, 100)
    bs = BatchForSequence(x, 10, 5, 2, 2)
    bs.generate_feature_labels()


if __name__ == "__main__":
    main()








