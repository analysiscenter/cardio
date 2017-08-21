# pylint: skip-file
import os
import sys
import numpy as np
import tensorflow as tf
from time import time
import threading

sys.path.append("../..")
from dataset.opensets import CIFAR10


if __name__ == "__main__":
    BATCH_SIZE = 64
    N_ITERS = 1000

    cifar = CIFAR10()
    N_CLASSES = len(np.unique(cifar._data[0][1]))

    input_images = tf.placeholder("uint8", [None, 32, 32, 3])
    input_labels = tf.placeholder("uint8", [None])
    is_training = tf.placeholder("bool", ())

    encoded_labels = tf.one_hot(input_labels, depth=N_CLASSES)
    input_cast = tf.cast(input_images, 'float')
    conv1 = tf.layers.conv2d(input_cast, filters=32, kernel_size=[5,5], padding="same", activation=tf.nn.relu)
    max1 = tf.layers.max_pooling2d(conv1, pool_size=[2, 2], strides=2)
    bn1 = tf.layers.batch_normalization(max1, training=is_training)
    conv2 = tf.layers.conv2d(bn1, filters=64, kernel_size=[5,5], padding="same", activation=tf.nn.relu)
    bn2 = tf.layers.batch_normalization(conv2, training=is_training)
    max2 = tf.layers.max_pooling2d(bn2, pool_size=[2, 2], strides=2)
    flat = tf.contrib.layers.flatten(max2)
    flat1 = tf.layers.dense(flat, units=512, activation=tf.nn.relu)
    flat2 = tf.layers.dense(flat1, units=256, activation=tf.nn.relu)
    drop = tf.layers.dropout(flat2, rate=0.5)
    model_output = tf.layers.dense(drop, units=N_CLASSES)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=encoded_labels, logits=model_output))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(cost)

    prediction = tf.argmax(model_output, 1)
    correct_prediction = tf.equal(prediction, tf.argmax(encoded_labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())


    print("Start training...")
    t = time()
    i = 0
    for batch in cifar.train.gen_batch(BATCH_SIZE, shuffle=False, n_epochs=2):
        i += 1
        _, loss = sess.run([train_op, cost], feed_dict={input_images: batch.images, input_labels: batch.labels, is_training: True})
        if (i + 1) % 50 == 0:
            print("Iteration", i + 1, "loss =", loss)
    print("Iteration", i + 1, "loss =", loss)
    print("End training", time() - t)

    print()
    print("Start validating...")
    for i in range(3):
        batch = cifar.test.next_batch(BATCH_SIZE * 10, shuffle=False, n_epochs=None)
        acc = sess.run(accuracy, feed_dict={input_images: batch.images, input_labels: batch.labels, is_training: False})
        print("Batch", i, "accuracy =", acc)
    print("End validating")
