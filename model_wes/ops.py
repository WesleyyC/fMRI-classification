import tensorflow as tf


def conv3d(input_, output_channels, stride, filter_depth, filter_height, filter_width, layer_no, padding="SAME"):
    with tf.variable_scope('conv3d_layer_'+str(layer_no)):
        w = tf.get_variable('w', [filter_depth, filter_height, filter_width, input_.shape[-1], output_channels],
                            dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer(stddev=0.02))
        b = tf.get_variable('b', [output_channels], dtype=tf.float32, initializer=tf.constant_initializer(0.0))

        conv = tf.nn.conv3d(input_, w, [1, stride, stride, stride, 1], padding=padding)

        return tf.nn.bias_add(conv, b)


def conv_pool_block(X_, kernel_size, conv_stride, filter_depth, filter_height, filter_width, pool_size, pool_stride, layer_no):
    with tf.variable_scope('conv_block_' + str(layer_no)):
        conv1 = conv3d(X_, kernel_size, conv_stride, filter_depth, filter_height, filter_width, 1)
        relu = tf.nn.relu(conv1)
        conv2 = conv3d(relu, kernel_size, conv_stride, filter_depth, filter_height, filter_width, 2)
        res = X_+conv2
        pool = tf.nn.max_pool3d(res, [1, pool_size, pool_size, pool_size, 1],
                                [1, pool_stride, pool_stride, pool_stride, 1], padding="VALID")
    return pool


def dense_block(X_, output_channel, layer_no):
    with tf.variable_scope('dense_block_' + str(layer_no)):
        X_dim = X_.get_shape().as_list()
        reshape = tf.reshape(X_, [-1,
                                  X_dim[1] * X_dim[2] *
                                  X_dim[3] * X_dim[4]])
        return tf.layers.dense(reshape, output_channel)
