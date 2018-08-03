import scipy.sparse as sparse
import numpy as np
from tqdm import tqdm


class BatchForSequence(object):
    def __init__(self, rating_matrix, timestamp_matrix, batch_size, feature_length, label_length, neg_label_length):
        self.padding_index = rating_matrix.shape[0]
        self.batch_size = batch_size
        self.feature_length = feature_length
        self.label_length = label_length
        self.max_length = feature_length + label_length
        self.neg_label_length = neg_label_length
        self.rating_lists = sparse.lil_matrix(rating_matrix).rows
        self.lengths = np.vectorize(len)(self.rating_lists)
        self.generate_rating_lists(rating_matrix, timestamp_matrix)

    def generate_rating_lists(self, rating_matrix, timestamp_matrix):
        timestamp_matrix = rating_matrix.multiply(timestamp_matrix)
        for i in range(self.padding_index):
            argsort = np.argsort(timestamp_matrix[i].data)
            self.rating_lists[i] = np.array(self.rating_lists[i])[argsort]

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

        for user_index in tqdm(range(len(self.rating_lists))):
            rating_list = self.rating_lists[user_index]
            neg_pool = np.array(list(set(range(self.padding_index)) - set(rating_list)), dtype=np.int32)

            for seq in self._sliding_window(user_index):
                if len(seq) > self.label_length:
                    features = seq[:-self.label_length]
                    self.features[seq_index][self.feature_length-len(features):] = features
                    self.pos_labels[seq_index][:] = seq[-self.label_length:]

                    self.neg_labels[seq_index][:] = neg_pool[np.random.randint(len(neg_pool),
                                                                               size=self.neg_label_length)]

                    seq_index += 1

        self.features = self.features[:seq_index]
        self.pos_labels = self.pos_labels[:seq_index]
        self.neg_labels = self.neg_labels[:seq_index]

    def next_batch(self):
        remaining_size = self.features.shape[0]
        batch_index = 0
        while remaining_size > 0:
            if remaining_size < self.batch_size:
                output = (self.features[batch_index*self.batch_size:-1],
                          self.pos_labels[batch_index * self.batch_size:-1],
                          self.neg_labels[batch_index * self.batch_size:-1])
                remaining_size = 0
            else:
                output = (self.features[batch_index*self.batch_size:(batch_index+1)*self.batch_size],
                          self.pos_labels[batch_index*self.batch_size:(batch_index+1)*self.batch_size],
                          self.neg_labels[batch_index*self.batch_size:(batch_index+1)*self.batch_size])
                batch_index += 1
                remaining_size -= self.batch_size
            yield output

    def _sliding_window(self, user_index, step_size=1):
        if self.lengths[user_index] - self.max_length >= 0:
            for i in range(self.lengths[user_index], 0, -step_size):
                if i - self.max_length >= 0:
                    yield self.rating_lists[user_index][i - self.max_length:i]
                else:
                    break
        else:
            yield self.rating_lists[user_index]

# TEST CODE
# def main():
#     x = sparse.random(10000, 1000)
#     bs = BatchForSequence(x, x, 10, 5, 2, 2)
#     bs.generate_feature_labels()
#
#     for batch in bs.next_batch():
#         print(batch)
#
#
# if __name__ == "__main__":
#     main()








