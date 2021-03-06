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
starting_learning_rate = 0.005
decay_step = 10
decay_rate = 0.95

# Build NN Graph

tf.reset_default_graph()
sess = tf.InteractiveSession()

X_batch = tf.placeholder(shape=(None, 26, 31, 23), dtype=tf.float32, name='X_batch')
Y_batch = tf.placeholder(shape=(None, 19), dtype=tf.float32, name='Y_batch')
training_flag = tf.placeholder(dtype=tf.bool, name='training_flag')

noise_layer_1 = ops.gaussian_noise_layer(X_batch, 0.5, training_flag)

kernel_size = 524
stride = 2
filter_depth = 4
filter_height = 4

conv_layer_1 = ops.conv2d(noise_layer_1, kernel_size, stride, filter_depth, filter_height, 1, padding="VALID")
conv_layer_3 = ops.conv2d(tf.transpose(noise_layer_1, [0, 1, 3, 2]), kernel_size, stride, filter_depth, filter_height,
                          3, padding="VALID")
conv_layer_5 = ops.conv2d(tf.transpose(noise_layer_1, [0, 2, 3, 1]), kernel_size, stride, filter_depth, filter_height,
                          5, padding="VALID")

kernel_size = 524
stride = 1
filter_depth = 3
filter_height = 3

conv_layer_2 = ops.conv2d(conv_layer_1, kernel_size, stride, filter_depth, filter_height, 2, padding="VALID")
conv_layer_4 = ops.conv2d(conv_layer_3, kernel_size, stride, filter_depth, filter_height, 4, padding="VALID")
conv_layer_6 = ops.conv2d(conv_layer_5, kernel_size, stride, filter_depth, filter_height, 6, padding="VALID")

kernel_size = 524
stride = 1
filter_depth = 3
filter_height = 3

conv_layer_7 = ops.conv2d(conv_layer_1, kernel_size, stride, filter_depth, filter_height, 7, padding="VALID")
conv_layer_8 = ops.conv2d(conv_layer_3, kernel_size, stride, filter_depth, filter_height, 8, padding="VALID")
conv_layer_9 = ops.conv2d(conv_layer_5, kernel_size, stride, filter_depth, filter_height, 9, padding="VALID")

dense_1 = ops.dense_block(conv_layer_7, label_size, None, 1)
dense_2 = ops.dense_block(conv_layer_8, label_size, None, 2)
dense_3 = ops.dense_block(conv_layer_9, label_size, None, 3)

dense = tf.concat([dense_1, dense_2, dense_3], axis=-1)

logits = tf.layers.dense(dense, 19)

# Prediction Loss

Y_prediction = tf.round(tf.nn.sigmoid(logits))

# Training Loss

cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=Y_batch, logits=logits)

loss = tf.reduce_mean(cross_entropy)

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

epochs = 1000
batch_size = 512
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
