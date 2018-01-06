import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle

import ops
import utils

# Mode

infer_only = False

# Load data
train_X = np.load('../data/train_X.npy')
train_Y = np.load('../data/train_binary_Y.npy')

# Process data

train_X = utils.normalized_data(train_X)

train_X = train_X[:, :, :, :, np.newaxis]

# Shuffle and divide two train/test
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
starting_learning_rate = 0.001
decay_step = 100
decay_rate = 0.96
regularizer_scale = 0.01
dropout_keep = 1
mask_weight = 0
# Build NN Graph

tf.reset_default_graph()
sess = tf.InteractiveSession()

X_batch = tf.placeholder(shape=(None, 26, 31, 23, 1), dtype=tf.float32, name='X_batch')
Y_batch = tf.placeholder(shape=(None, 19), dtype=tf.float32, name='Y_batch')
training_flag = tf.placeholder(dtype=tf.bool, name='training_flag')

regularizer = tf.contrib.layers.l1_regularizer(scale=regularizer_scale)

kernel_size = 64
stride = 1
filter_depth = 2
filter_height = 3
filter_width = 2
conv_layer_1 = ops.conv3d_block(X_batch, training_flag, kernel_size, stride, filter_depth, filter_height,
                                filter_width, None, 1)

kernel_size = 32
stride = 1
filter_depth = 2
filter_height = 3
filter_width = 2
conv_layer_2 = ops.conv3d_block(conv_layer_1, training_flag, kernel_size, stride, filter_depth, filter_height,
                                filter_width, None, 2)

mask = tf.layers.dense(conv_layer_2, 2)
mask = tf.nn.sigmoid(mask)

masked_X_batch = tf.multiply(X_batch, mask)

kernel_size = 64
stride = 1
filter_depth = 2
filter_height = 3
filter_width = 2
conv_layer_3 = ops.conv3d_block(masked_X_batch, training_flag, kernel_size, stride, filter_depth, filter_height,
                                filter_width, None, 3)

kernel_size = 32
stride = 1
filter_depth = 2
filter_height = 3
filter_width = 2
conv_layer_4 = ops.conv3d_block(conv_layer_3, training_flag, kernel_size, stride, filter_depth, filter_height,
                                filter_width, None, 4)

pool_layer_4 = tf.nn.max_pool3d(conv_layer_4, [1, 2, 3, 2, 1],
                                [1, 2, 3, 2, 1], padding="VALID")

kernel_size = 32
stride = 1
filter_depth = 2
filter_height = 3
filter_width = 2
conv_layer_5 = ops.conv3d_block(pool_layer_4, training_flag, kernel_size, stride, filter_depth, filter_height,
                                filter_width, None, 5)

kernel_size = 8
stride = 1
filter_depth = 2
filter_height = 3
filter_width = 2
conv_layer_6 = ops.conv3d_block(conv_layer_5, training_flag, kernel_size, stride, filter_depth, filter_height,
                                filter_width, None, 6)

pool_layer_5 = tf.nn.max_pool3d(conv_layer_6, [1, 2, 3, 2, 1],
                                [1, 2, 3, 2, 1], padding="VALID")

dropout_1 = tf.nn.dropout(pool_layer_5, dropout_keep)

dense_2 = ops.dense_block(dropout_1, label_size, regularizer, 2)

logits = dense_2

# Prediction Loss

Y_prediction = tf.round(tf.nn.sigmoid(logits))

# Training Loss

cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y_batch, logits=logits))

if regularizer_scale > 0:
    reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)
else:
    reg_term = 0

loss = cross_entropy + reg_term

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

epochs = 300
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
                                               training_flag: False})
            auc_list.append(roc_auc_score(test_Y[i:i_end], report_Y_prediction, average='micro'))
            subset_accuracy_list.append(accuracy_score(test_Y[i:i_end], report_Y_prediction))
            i = i_end
        print(str.format("Epoch %d test  auc is %f and subset accuracy is %f"
                         % (epoch, float(np.mean(auc_list)), float(np.mean(subset_accuracy_list)))))
        if np.mean(subset_accuracy_list) > best_subset_accuracy:
            saver.save(sess, saved_mdl_name)
            best_subset_accuracy = np.mean(subset_accuracy_list)
            print("Updated best model."
                  "")
        print("**************************************************************")

# Generate Validation Data

print("Generating Validation Submission...")

valid_test_X = np.load('../data/valid_test_X.npy')
valid_test_X = utils.normalized_data(valid_test_X)
valid_test_Y = np.zeros([len(valid_test_X), 19])

saver.restore(sess, saved_mdl_name)

i = 0
while i < len(valid_test_X):
    i_end = min(i + batch_size, len(valid_test_X))
    valid_Y_prediction = sess.run(Y_prediction,
                                  feed_dict={
                                      X_batch: valid_test_X[i:i_end, :, :, :, np.newaxis],
                                      training_flag: False})
    valid_test_Y[i:i_end] = valid_Y_prediction
    i = i_end

np.save('result', valid_test_Y)

print("Done.")
