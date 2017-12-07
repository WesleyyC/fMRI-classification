import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle

import ops
import utils

# Mode

infer_only = True

# Load data
train_X = np.load('../data/train_X.npy')
train_Y = np.load('../data/train_binary_Y.npy')

# Process Data

train_X = utils.normalized_data(train_X)

train_X = train_X[:, :, :, :, np.newaxis]

# Shuffle and Divide Two Train/Test
train_X, train_Y = shuffle(train_X, train_Y)
sample_number = len(train_X)

test_fraction = 0.1
test_range = int(test_fraction * sample_number)
test_X = train_X[:test_range]
train_X = train_X[test_range:]
test_Y = train_Y[:test_range]
train_Y = train_Y[test_range:]

# Model Parameter

label_size = 19

regularizer_scale = 0.0

starting_learning_rate = 0.001
decay_step = 20
decay_rate = 1

# Build NN Graph

tf.reset_default_graph()
sess = tf.InteractiveSession()

X_batch = tf.placeholder(shape=(None, 26, 31, 23, 1), dtype=tf.float32, name='X_batch')
Y_batch = tf.placeholder(shape=(None, 19), dtype=tf.float32, name='Y_batch')
training_flag = tf.placeholder(dtype=tf.bool, name='training_flag')
keep_prob = tf.placeholder(tf.float32)


regularizer = tf.contrib.layers.l2_regularizer(scale=regularizer_scale)

noise_layer_1 = ops.gaussian_noise_layer(X_batch, 1, training_flag)

kernel_size = 32
stride = 1
filter_depth = 5
filter_height = 5
filter_width = 5
conv_layer_1 = ops.conv3d_block(X_batch, training_flag, kernel_size, stride, filter_depth, filter_height,
                                filter_width, regularizer, 1, padding="SAME")

pool_size = 2
pool_stride = pool_size
pool_layer_1 = tf.nn.max_pool3d(conv_layer_1, [1, pool_size, pool_size, pool_size, 1],
                                [1, pool_stride, pool_stride, pool_stride, 1], padding="SAME")

kernel_size = 64
stride = 1
filter_depth = 5
filter_height = 5
filter_width = 5
conv_layer_2 = ops.conv3d_block(pool_layer_1, training_flag, kernel_size, stride, filter_depth, filter_height,
                                filter_width, regularizer, 2, padding="SAME")

pool_size = 2
pool_stride = pool_size
pool_layer_2 = tf.nn.max_pool3d(conv_layer_2, [1, pool_size, pool_size, pool_size, 1],
                                [1, pool_stride, pool_stride, pool_stride, 1], padding="SAME")

dense_1 = ops.dense_block(pool_layer_2, 1024, regularizer, 1)

dense_1 = tf.nn.relu(dense_1)

dense_1 = tf.nn.dropout(dense_1, keep_prob)

dense_2 = tf.layers.dense(dense_1, 19)

logit = dense_2

# Prediction Loss

Y_prediction = tf.round(tf.nn.sigmoid(logit))

# Training Loss

cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y_batch, logits=logit))

loss = cross_entropy

# Add Regularization

if regularizer_scale > 0:
    reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)
else:
    reg_term = 0

loss = loss + reg_term

# Gradient

params = tf.trainable_variables()
gradients = tf.gradients(loss, params)

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(starting_learning_rate, global_step,
                                           decay_step, decay_rate, staircase=True)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
update_step = optimizer.apply_gradients(zip(gradients, params), global_step=global_step)

# Model Persistence

saver = tf.train.Saver()

# Training

sess.run(tf.global_variables_initializer())

epochs = 100
batch_size = 64
report_step = 1000
saved_mdl_name = 'result.mdl'

best_subset_accuracy = 0

if not infer_only:

    for epoch in range(epochs):

        train_X, train_Y = shuffle(train_X, train_Y)

        i = 0
        i_report = report_step
        while i < len(train_X):
            i_end = min(i + batch_size, len(train_X))
            report_Y_prediction, _ = sess.run([Y_prediction, update_step],
                                              feed_dict={
                                                  X_batch: train_X[i:i_end],
                                                  Y_batch: train_Y[i:i_end],
                                                  keep_prob: 0.5,
                                                  training_flag: True})
            if i > i_report:
                auc = roc_auc_score(train_Y[i:i_end], report_Y_prediction, average='micro')
                subset_accuracy = accuracy_score(train_Y[i:i_end], report_Y_prediction)
                print(str.format("Epoch %d train auc is %f and subset accuracy is %f"
                                 % (epoch, auc, subset_accuracy)))
                i_report += report_step
            i = i_end

        i = 0
        auc_list = []
        subset_accuracy_list = []
        while i < len(test_X):
            i_end = min(i + batch_size, len(test_X))
            report_Y_prediction = sess.run(Y_prediction,
                                           feed_dict={
                                               X_batch: test_X[i:i_end],
                                               keep_prob: 1,
                                               training_flag: False})
            auc_list.append(roc_auc_score(test_Y[i:i_end], report_Y_prediction, average='micro'))
            subset_accuracy_list.append(accuracy_score(test_Y[i:i_end], report_Y_prediction))
            i = i_end
        print(str.format("Epoch %d test  auc is %f and subset accuracy is %f"
                         % (epoch, float(np.mean(auc_list)), float(np.mean(subset_accuracy_list)))))
        if np.mean(subset_accuracy_list) >= best_subset_accuracy:
            saver.save(sess, saved_mdl_name)
            best_subset_accuracy = np.mean(subset_accuracy_list)
            print("Updated best model.")
        print("**************************************************************")

# Generate Validation Data

print("Generating Validation Submission...")

valid_test_X = np.load('../data/valid_test_X.npy')
valid_test_X = utils.normalized_data(valid_test_X)
valid_test_Y = np.zeros([len(valid_test_X), 19])

if infer_only:
    saver.restore(sess, saved_mdl_name)

i = 0
while i < len(valid_test_X):
    i_end = min(i + batch_size, len(valid_test_X))
    valid_Y_prediction = sess.run(Y_prediction,
                                  feed_dict={
                                      X_batch: valid_test_X[i:i_end, :, :, :, np.newaxis],
                                      keep_prob: 1,
                                      training_flag: False})
    valid_test_Y[i:i_end] = valid_Y_prediction
    i = i_end

np.save('result', valid_test_Y)

print("Done.")
