import tensorflow as tf
import tensorflow.contrib as tfc


def conv3d(input_, channels, filters, stride, filter_d, filter_h, filter_w, padding="SAME"):
    w = tf.get_variable('w', [filter_d, filter_h, filter_w, channels, filters],
                        dtype=tf.float32,
                        initializer=tfc.layers.variance_scaling_initializer(1.0, mode='FAN_AVG',
                                                                            uniform=True))
    b = tf.get_variable('b', [filters], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
    conv = tf.nn.conv3d(input_, w, [1, stride, stride, stride, 1], padding=padding)
    pre_acitivation = tf.nn.bias_add(conv, b)
    conv_out = tf.nn.relu(pre_acitivation)
    return conv_out


def linear_layer(linear_in, dim, hiddens):
    weights = tf.get_variable('weights', [dim, hiddens], tf.float32,
                              initializer=tfc.layers.variance_scaling_initializer(mode='FAN_AVG', uniform=True))
    bias = tf.get_variable('bias', [hiddens], tf.float32,
                           initializer=tf.constant_initializer(0.1))
    pre_activations = tf.add(tf.matmul(linear_in, weights), bias)
    linear_out = tf.nn.relu(pre_activations)
    return linear_out
