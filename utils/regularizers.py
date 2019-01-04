import tensorflow as tf

Regularizer = {
    "Adam": tf.train.AdamOptimizer,
    "RMSProp": tf.train.RMSPropOptimizer,
    "SGD": tf.train.GradientDescentOptimizer,
    #"Momentum": tf.train.MomentumOptimizer,
}