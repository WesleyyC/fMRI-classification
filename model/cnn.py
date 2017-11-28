import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

import ops

train_X = np.load('../data/train_X.npy')
train_binary_Y = np.load('../data/train_binary_Y.npy')
valid_test_X = np.load('../data/valid_test_X.npy')

tf.reset_default_graph()
sess = tf.InteractiveSession()

X_batch = tf.placeholder(shape=(None, 26, 31, 23, 1), dtype=tf.float32, name='X_batch')
Y_batch = tf.placeholder(shape=(None, 19), dtype=tf.float32, name='Y_batch')
training_flag = tf.placeholder(dtype=tf.bool, name='training_flag')

label_size = 19

kernel_size = 3
stride = 1
filter_depth = 3
filter_height = 3
filter_width = 3

pool_size = 2
pool_stride = pool_size

learning_rate = 0.01

X_batch_norm = tf.layers.batch_normalization(X_batch, training=training_flag)
X_batch_conv = ops.conv3d(X_batch_norm, kernel_size, stride, filter_depth, filter_height, filter_width)
X_batch_pool = tf.nn.max_pool3d(X_batch_conv, [1, pool_size, pool_size, pool_size, 1],
                                [1, pool_stride, pool_stride, pool_stride, 1], padding="SAME")

X_batch_reshape = tf.reshape(X_batch_pool, [-1,
                                            X_batch_pool.get_shape().as_list()[1] * X_batch_pool.get_shape().as_list()[
                                                2] * X_batch_pool.get_shape().as_list()[3] *
                                            X_batch_pool.get_shape().as_list()[4]])

logits = tf.layers.dense(X_batch_reshape, label_size)
Y_prediction = tf.nn.sigmoid(logits)

cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=Y_batch, logits=logits)

loss = tf.reduce_mean(cross_entropy)

params = tf.trainable_variables()
gradients = tf.gradients(loss, params)

optimizer = tf.train.GradientDescentOptimizer(learning_rate)
update_step = optimizer.apply_gradients(zip(gradients, params))

epochs = 30
batch_size = 32

sess.run(tf.global_variables_initializer())

for _ in range(epochs):
    train_X, train_binary_Y = shuffle(train_X, train_binary_Y)
    i = 0
    while i < len(train_X):
        i_end = min(i + batch_size, len(train_X))
        sess.run(update_step, feed_dict={X_batch: train_X[i:i_end, :, :, :, np.newaxis],
                                         Y_batch: train_binary_Y[i:i_end],
                                         training_flag: True})
        i = i_end
    report_loss = sess.run(loss, feed_dict={X_batch: train_X[:, :, :, :, np.newaxis], Y_batch: train_binary_Y, training_flag: False})
    print(report_loss)
