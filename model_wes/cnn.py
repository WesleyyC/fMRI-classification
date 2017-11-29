import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle

import ops

# Load data
train_X = np.load('../data/train_X.npy')
train_Y = np.load('../data/train_binary_Y.npy')

# Process data

train_X = train_X[:, :, :, :, np.newaxis]

# Shuffle and divide two train/test
train_X, train_Y = shuffle(train_X, train_Y)
sample_number = len(train_X)

test_fraction = 0.2
test_range = int(test_fraction * sample_number)
test_X = train_X[:test_range]
train_X = train_X[test_range:]
test_Y = train_Y[:test_range]
train_Y = train_Y[test_range:]

# Build NN Graph

label_size = 19

kernel_size = 3
stride = 1
filter_depth = 3
filter_height = 3
filter_width = 3

pool_size = 2
pool_stride = pool_size

learning_rate = 0.01

tf.reset_default_graph()
sess = tf.InteractiveSession()

X_batch = tf.placeholder(shape=(None, 26, 31, 23, 1), dtype=tf.float32, name='X_batch')
Y_batch = tf.placeholder(shape=(None, 19), dtype=tf.float32, name='Y_batch')
training_flag = tf.placeholder(dtype=tf.bool, name='training_flag')

conv_1 = ops.conv3d(X_batch, kernel_size, stride, filter_depth, filter_height, filter_width)
pool_1 = tf.nn.max_pool3d(conv_1, [1, pool_size, pool_size, pool_size, 1],
                          [1, pool_stride, pool_stride, pool_stride, 1], padding="SAME")

dense_1_reshape_dim = pool_1.get_shape().as_list()
dense_1_reshape = tf.reshape(pool_1, [-1,
                                      dense_1_reshape_dim[1] * dense_1_reshape_dim[2] *
                                      dense_1_reshape_dim[3] *dense_1_reshape_dim[4]])

dense_1 = tf.layers.dense(dense_1_reshape, label_size)

logits = dense_1

# Prediction Loss

Y_prediction = tf.round(tf.nn.sigmoid(logits))

# Training Loss

cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=Y_batch, logits=logits)

loss = tf.reduce_mean(cross_entropy)

# Gradient

params = tf.trainable_variables()
gradients = tf.gradients(loss, params)

optimizer = tf.train.GradientDescentOptimizer(learning_rate)
update_step = optimizer.apply_gradients(zip(gradients, params))

# Training

epochs = 100
batch_size = 16
report_step = 1000

sess.run(tf.global_variables_initializer())

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
    print("****************************************************************")

# Generate Validation Data

print("Generating Validation Submission...")

valid_test_X = np.load('../data/valid_test_X.npy')
valid_test_Y = np.zeros([len(valid_test_X), 19])

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
