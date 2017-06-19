# pylint: skip-file
import os
import sys
import numpy as np
import tensorflow as tf

sys.path.append("../..")
from dataset.opensets import MNIST

if __name__ == "__main__":
    BATCH_SIZE = 64

    input_images = tf.placeholder("uint8", [None, 28, 28, 1])
    input_labels = tf.placeholder("uint8", [None])

    input_vectors = tf.cast(tf.reshape(input_images, [-1, 28 * 28]), 'float')
    layer1 = tf.layers.dense(input_vectors, units=512, activation=tf.nn.relu)
    layer2 = tf.layers.dense(layer1, units=256, activation=tf.nn.relu)
    model_output = tf.layers.dense(layer2, units=10)
    encoded_labels = tf.one_hot(input_labels, depth=10)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=encoded_labels, logits=model_output))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

    prediction = tf.argmax(model_output, 1)
    correct_prediction = tf.equal(prediction, tf.argmax(encoded_labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    mnist = MNIST()

    print()
    print("Start training...")
    i = 0
    for batch in mnist.train.gen_batch(BATCH_SIZE, shuffle=True, n_epochs=1):
        i += 1
        _, loss = sess.run([optimizer, cost], feed_dict={input_images: batch.images, input_labels: batch.labels})
        if i % 100 == 0:
            print("Iteration", i, "loss =", loss)
    print("Iteration", i, "loss =", loss)
    print("End training")

    print()
    print("Start validating...")
    for i in range(3):
        batch = mnist.test.next_batch(BATCH_SIZE, shuffle=False, n_epochs=None)
        acc = sess.run(accuracy, feed_dict={input_images: batch.images, input_labels: batch.labels})
        print("Batch", i, "accuracy =", acc)
    print("End validating")
