"""Contains Dirichlet model."""

import numpy as np
import tensorflow as tf

from ..tf_base_model import TFBaseModel
from ..layers import conv_cell


class DirichletModel(TFBaseModel):
    def __init__(self):
        super().__init__()

        self._input_layer = None
        self._target = None
        self._is_training = None

        self._output_layer = None

        self._loss = None
        self._global_step = None
        self._train_step = None

    def build(self, input_shape, output_shape):  # pylint: disable=protected-access
        input_shape = (None,) + input_shape
        output_shape = (None,) + output_shape
        k = 0.001

        self._graph = tf.Graph()
        with self.graph.as_default():  # pylint: disable=not-context-manager
            self._input_layer = tf.placeholder(tf.float32, shape=input_shape, name="input_layer")
            input_channels_last = tf.transpose(self._input_layer, perm=[0, 2, 1], name="channels_last")

            self._target = tf.placeholder(tf.float32, shape=output_shape, name="target")
            target = (1 - 2 * k) * self._target + k

            self._is_training = tf.placeholder(tf.bool, shape=[], name="is_training")

            n_filters = [10, 10, 10, 15, 15, 15, 20, 20, 20, 30, 30]
            kernel_size = [5, 5, 5, 5, 5, 5, 3, 3, 3, 3, 3]
            cell = input_channels_last
            for i, (n, s) in enumerate(zip(n_filters, kernel_size)):
                cell = conv_cell("cell_" + str(i + 1), cell, self._is_training, n, s)

            flat = tf.contrib.layers.flatten(cell)
            with tf.variable_scope("dense_1"):  # pylint: disable=not-context-manager
                dense = tf.layers.dense(flat, 8, use_bias=False, name="dense")
                bnorm = tf.layers.batch_normalization(dense, training=self._is_training, name="batch_norm")
                act = tf.nn.elu(bnorm, name="activation")

            with tf.variable_scope("dense_2"):  # pylint: disable=not-context-manager
                dense = tf.layers.dense(act, output_shape[1], use_bias=False, name="dense")
                bnorm = tf.layers.batch_normalization(dense, training=self._is_training, name="batch_norm")
                self._output_layer = tf.nn.softplus(bnorm, name="output_layer")

            self._loss = tf.reduce_mean(tf.lbeta(self._output_layer) -
                                        tf.reduce_sum((self._output_layer - 1) * tf.log(target), axis=1),
                                        name="loss")

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self._global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")
                opt = tf.train.AdamOptimizer()
                self._train_step = opt.minimize(self._loss, global_step=self._global_step, name="train_step")
        return self

    def save(self, path):
        with self.graph.as_default():  # pylint: disable=not-context-manager
            saver = tf.train.Saver()
            saver.save(self.session, path, global_step=self._global_step)
        return self

    @staticmethod
    def _concatenate_batch(batch):
        x = np.concatenate(batch.signal)
        y = np.concatenate([np.tile(item.target, (item.signal.shape[0], 1)) for item in batch])
        split_indices = np.cumsum([item.signal.shape[0] for item in batch])[:-1]
        return x, y, split_indices

    @staticmethod
    def _append_result(res, res_list, accept_none=False):
        if not accept_none and res_list is None:
            raise ValueError("List with results must not be None")
        if res_list is not None:
            res_list.append(res)

    def train_on_batch(self, batch, loss_list=None):  # pylint: disable=arguments-differ
        self._create_session()
        x, y, _ = self._concatenate_batch(batch)
        feed_dict = {self._input_layer: x, self._target: y, self._is_training: True}
        _, loss = self.session.run([self._train_step, self._loss], feed_dict=feed_dict)
        self._append_result(loss, loss_list, accept_none=True)
        return batch

    def test_on_batch(self, batch, loss_list=None):  # pylint: disable=arguments-differ
        self._create_session()
        x, y, _ = self._concatenate_batch(batch)
        feed_dict = {self._input_layer: x, self._target: y, self._is_training: False}
        loss = self.session.run(self._loss, feed_dict=feed_dict)
        self._append_result(loss, loss_list)
        return batch

    @staticmethod
    def _get_dirichlet_stats(alpha):
        alpha_sum = np.sum(alpha, axis=1)[:, np.newaxis]
        comp_m1 = alpha / alpha_sum
        comp_m2 = (alpha * (alpha + 1)) / (alpha_sum * (alpha_sum + 1))
        mean = np.mean(comp_m1, axis=0)
        var = np.mean(comp_m2, axis=0) - mean**2
        return mean, var

    def predict_on_batch(self, batch, predictions_list=None, target_list=None):  # pylint: disable=arguments-differ
        self._create_session()
        n_classes = len(batch.label_binarizer.classes_)
        max_var = (n_classes - 1) /  n_classes**2
        x, _, split_indices = self._concatenate_batch(batch)
        feed_dict = {self._input_layer: x, self._is_training: False}
        alpha = self.session.run(self._output_layer, feed_dict=feed_dict)
        alpha = np.split(alpha, split_indices)
        for a, t in zip(alpha, batch.target):
            mean, var = self._get_dirichlet_stats(a)
            uncertainty = var[np.argmax(mean)] / max_var
            predictions_dict = {"class_prob": dict(zip(batch.label_binarizer.classes_, mean)),
                                "uncertainty": uncertainty}
            self._append_result(predictions_dict, predictions_list)
            target_dict = {"class_prob": dict(zip(batch.label_binarizer.classes_, t))}
            self._append_result(target_dict, target_list, accept_none=True)
        return batch
