from random import randint


def get_negative_samples(positive_sets, users, num_item, negative_samples):

    cdef int i = 0
    cdef int j = 0

    for i in range(len(negative_samples)):
        user = users[i]
        negatives = negative_samples[i]
        for j in range(len(negatives)):
            neg = negatives[j]
            while neg in positive_sets[user]:
                negative_samples[i, j] = neg = randint(0, num_item)

    return negative_samples