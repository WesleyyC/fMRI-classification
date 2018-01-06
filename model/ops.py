import tensorflow as tf


def conv3d(input_, output_channels, stride, filter_depth, filter_height, filter_width, regularizer, layer_no,
           padding="SAME"):
    with tf.variable_scope('conv3d_layer_' + str(layer_no), regularizer=regularizer):
        w = tf.get_variable('w', [filter_depth, filter_height, filter_width, input_.shape[-1], output_channels],
                            dtype=tf.float32,
                            initializer=tf.contrib.layers.variance_scaling_initializer(1.0, mode='FAN_AVG',
                                                                                       uniform=True))
        b = tf.get_variable('b', [output_channels], dtype=tf.float32, initializer=tf.constant_initializer(0.0))

        conv = tf.nn.conv3d(input_, w, [1, stride, stride, stride, 1], padding=padding)

        return tf.nn.bias_add(conv, b)


def conv2d(input_, output_channels, stride, filter_depth, filter_height, layer_no, padding="SAME"):
    with tf.variable_scope('conv2d_layer_' + str(layer_no)):
        w = tf.get_variable('w', [filter_depth, filter_height, input_.shape[-1], output_channels],
                            dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer(stddev=0.02))
        b = tf.get_variable('b', [output_channels], dtype=tf.float32, initializer=tf.constant_initializer(0.0))

        conv = tf.nn.conv2d(input_, w, [1, stride, stride, 1], padding=padding)

        return tf.nn.bias_add(conv, b)


def conv3d_block(X_, training_flag, kernel_size, conv_stride, filter_depth, filter_height, filter_width, regularizer,
                 layer_no, padding="SAME"):
    with tf.variable_scope('conv_block_' + str(layer_no)):
        conv1 = conv3d(X_, kernel_size, conv_stride, filter_depth, filter_height, filter_width, regularizer, 1, padding=padding)
        alpha = 0
        relu1 = tf.maximum(conv1, alpha * conv1)
    return relu1


def dense_block(X_, output_channel, regularizer, layer_no):
    with tf.variable_scope('dense_block_' + str(layer_no), regularizer=regularizer):
        X_dim = X_.get_shape().as_list()
        X_dim[0] = 1
        X_dim = reduce(lambda x, y: x * y, X_dim)
        reshape = tf.reshape(X_, [-1, X_dim])
        return tf.layers.dense(reshape, output_channel)


def gaussian_noise_layer(input_layer, std, training_flag):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
    return tf.cond(training_flag, lambda: input_layer + noise, lambda :input_layer)
