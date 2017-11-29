import tensorflow as tf


def conv3d(input_, output_channels, stride, filter_depth, filter_height, filter_width, layer_no, padding="SAME"):
    with tf.variable_scope('conv3d_layer_' + str(layer_no)):
        w = tf.get_variable('w', [filter_depth, filter_height, filter_width, input_.shape[-1], output_channels],
                            dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer(stddev=0.02))
        b = tf.get_variable('b', [output_channels], dtype=tf.float32, initializer=tf.constant_initializer(0.0))

        conv = tf.nn.conv3d(input_, w, [1, stride, stride, stride, 1], padding=padding)

        return tf.nn.bias_add(conv, b)


def conv3d_res_block(X_, training_flag, kernel_size, conv_stride, filter_depth, filter_height, filter_width, layer_no):
    with tf.variable_scope('conv_block_' + str(layer_no)):
        conv1 = conv3d(X_, kernel_size, conv_stride, filter_depth, filter_height, filter_width, 1)
        bn1 = tf.layers.batch_normalization(conv1, training=training_flag)
        relu1 = tf.nn.relu(bn1)
        conv2 = conv3d(relu1, kernel_size, conv_stride, filter_depth, filter_height, filter_width, 2)
        bn2 = tf.layers.batch_normalization(conv2, training=training_flag)
        relu2 = tf.nn.relu(bn2)
        res = X_ + relu2
    return res


def conv3d_block(X_, training_flag, kernel_size, conv_stride, filter_depth, filter_height, filter_width, layer_no):
    with tf.variable_scope('conv_block_' + str(layer_no)):
        conv1 = conv3d(X_, kernel_size, conv_stride, filter_depth, filter_height, filter_width, 1)
        relu1 = tf.nn.relu(conv1)
    return relu1


def dense_block(X_, output_channel, layer_no):
    with tf.variable_scope('dense_block_' + str(layer_no)):
        X_dim = X_.get_shape().as_list()
        reshape = tf.reshape(X_, [-1,
                                  X_dim[1] * X_dim[2] *
                                  X_dim[3] * X_dim[4]])
        return tf.layers.dense(reshape, output_channel)
