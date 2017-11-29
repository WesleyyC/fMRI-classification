import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

import ops

train_X = np.load('../data/train_X.npy')
train_binary_Y = np.load('../data/train_binary_Y.npy')
valid_test_X = np.load('../data/valid_test_X.npy')

tf.reset_default_graph()
sess = tf.InteractiveSession()

label_size = 19
learning_rate = 0.001
device = '/gpu:0'

with tf.device(device):
    X_batch = tf.placeholder(shape=(None, 26, 31, 23, 1), dtype=tf.float32, name='X_batch')
    Y_batch = tf.placeholder(shape=(None, 19), dtype=tf.float32, name='Y_batch')
    training_flag = tf.placeholder(dtype=tf.bool, name='training_flag')
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate)

    with tf.variable_scope('bn1'):
        X_batch_norm = tf.layers.batch_normalization(X_batch, training=training_flag)

    with tf.variable_scope('conv1'):
        channels = 1; filters = 4; size = 3; stride = 1;
        X_batch_conv1 = ops.conv3d(X_batch_norm, channels, filters, stride, size, size, size)

    with tf.variable_scope('conv1_2'):
        channels = 4; filters = 8; size = 3; stride = 1;
        X_batch_conv1_2 = ops.conv3d(X_batch_conv1, channels, filters, stride, size, size, size)

    with tf.variable_scope('pool1'):
        size = 2; stride = 2;
        X_batch_pool1 = tf.nn.max_pool3d(X_batch_conv1_2, ksize=[1, size, size, size, 1],
                                         strides=[1, stride, stride, stride, 1], padding="SAME")

    with tf.variable_scope('bn2'):
        X_batch_norm2 = tf.layers.batch_normalization(X_batch_pool1, training=training_flag)

    with tf.variable_scope('conv2'):
        channels = 8; filters = 16; size = 3; stride = 1;
        X_batch_conv2 = ops.conv3d(X_batch_norm2, channels, filters, stride, size, size, size)

    with tf.variable_scope('conv2_2'):
        channels = 16; filters = 32; size = 3; stride = 1;
        X_batch_conv2_2 = ops.conv3d(X_batch_conv2, channels, filters, stride, size, size, size)

    with tf.variable_scope('pool2'):
        size = 2; stride = 2;
        X_batch_pool2 = tf.nn.max_pool3d(X_batch_conv2_2, ksize=[1, size, size, size, 1],
                                         strides=[1, stride, stride, stride, 1], padding="SAME")

    conv_out_shape = X_batch_pool2.get_shape().as_list()
    conv_out_shape[0] = 1
    dim = reduce(lambda x, y: x * y, conv_out_shape)

    X_batch_reshape = tf.reshape(X_batch_pool2, [-1, dim])

    with tf.variable_scope('linear1'):
        out_dim = 512
        X_batch_linear1 = ops.linear_layer(X_batch_reshape, dim, out_dim)

    with tf.variable_scope('linear2'):
        out_dim = label_size; dim = 512;
        X_batch_linear2 = ops.linear_layer(X_batch_linear1, dim, out_dim)

    logits = X_batch_linear2

    Y_prediction = tf.nn.sigmoid(logits)

    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=Y_batch, logits=logits)

    loss = tf.reduce_mean(cross_entropy)

    grad_var_list = optimizer.compute_gradients(loss)
    apply_gradients_op = optimizer.apply_gradients(grad_var_list, global_step)


epochs = 30
batch_size = 32

sess.run(tf.global_variables_initializer())

sess.run(apply_gradients_op, feed_dict={X_batch: np.expand_dims(train_X[:100], axis=-1), Y_batch: train_binary_Y[:100], training_flag: True})
#
# for _ in range(epochs):
#     train_X, train_binary_Y = shuffle(train_X, train_binary_Y)
#     i = 0
#     while i < len(train_X):
#         i_end = min(i + batch_size, len(train_X))
#         sess.run(apply_gradients_op, feed_dict={X_batch: train_X[i:i_end, :, :, :, np.newaxis],
#                                          Y_batch: train_binary_Y[i:i_end],
#                                          training_flag: True})
#         i = i_end
#     report_loss = sess.run(loss, feed_dict={X_batch: train_X[:, :, :, :, np.newaxis], Y_batch: train_binary_Y, training_flag: False})
#     print(report_loss)
