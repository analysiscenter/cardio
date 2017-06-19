# pylint: skip-file
import os
import sys
import numpy as np
import tensorflow as tf

sys.path.append("../..")
from dataset import action, model, ImagesBatch
from dataset.opensets import MNIST


class MyMNIST(ImagesBatch):
    @model()
    def simple_nn():
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

        return [[input_images, input_labels], [optimizer, cost, accuracy]]

    @action(model='simple_nn', singleton=True)
    def train_nn(self, model, sess):
        input_images, input_labels = model[0]
        optimizer, cost, accuracy = model[1]
        _, loss = sess.run([optimizer, cost], feed_dict={input_images: self.images, input_labels: self.labels})
        return self

    @action(model='simple_nn')
    def print_accuracy(self, model, sess):
        input_images, input_labels = model[0]
        optimizer, cost, accuracy = model[1]
        acc = sess.run(accuracy, feed_dict={input_images: self.images, input_labels: self.labels})
        print("Accuracy =", acc)
        return self


if __name__ == "__main__":
    BATCH_SIZE = 64

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    mnist = MNIST(batch_class=MyMNIST)

    print()
    print("Start training...")
    mnist.train.p.train_nn(sess).run(BATCH_SIZE, shuffle=True, n_epochs=2, drop_last=True)
    print("End training")

    print()
    print("Start validating...")
    mnist.test.p.print_accuracy(sess).next_batch(BATCH_SIZE * 100, shuffle=True)
    print("End validating")
