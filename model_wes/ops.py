import tensorflow as tf


def conv3d(input_, output_channels, stride, filter_depth, filter_height, filter_width, padding="SAME", name='3d_conv'):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [filter_depth, filter_height, filter_width, input_.shape[-1], output_channels],
                            dtype=tf.float32,
                            initializer=tf.contrib.layers.variance_scaling_initializer(1.0, mode='FAN_AVG',
                                                                                       uniform=True))
        b = tf.get_variable('b', [output_channels], dtype=tf.float32, initializer=tf.constant_initializer(0.1))

        conv = tf.nn.conv3d(input_, w, [1, stride, stride, stride, 1], padding=padding)

        pre_acitivation = tf.nn.bias_add(conv, b)

        conv_out = tf.nn.relu(pre_acitivation)

        return conv_out
