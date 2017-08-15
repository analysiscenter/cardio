"""Contains beta model."""

import numpy as np
import tensorflow as tf

from ..tf_base_model import TFBaseModel
from ..layers import conv_cell
from ... import dataset as ds
from ...batch import EcgBatch


class BetaModel(TFBaseModel):
    def __init__(self):
        super().__init__()

        # model placeholders
        self._input_layer = None
        self._target = None
        self._is_training = None

        # model output
        self._alpha = None
        self._beta = None

        # model loss
        self._loss = None

        # model training step
        self._train_step = None

    def build(self):  # pylint: disable=protected-access
        k = 0.001
        input_shape = (None, 1, 2048)
        output_shape = (None, 1)

        self._graph = tf.Graph()
        with self.graph.as_default():  # pylint: disable=not-context-manager
            self._input_layer = tf.placeholder(tf.float32, shape=input_shape, name="input_layer")
            input_channels_last = tf.reshape(self._input_layer, [-1, 2048, 1], name="channels_last")

            self._target = tf.placeholder(tf.float32, shape=output_shape, name="target")
            target_flat = tf.reshape((1 - 2 * k) * self._target + k, [-1])

            self._is_training = tf.placeholder(tf.bool, shape=[], name="batch_norm_mode")

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
                dense = tf.layers.dense(act, 2, use_bias=False, name="dense")
                bnorm = tf.layers.batch_normalization(dense, training=self._is_training, name="batch_norm")
                output_layer = tf.nn.softplus(bnorm, name="output_layer")

            self._alpha = output_layer[:, 0]
            self._beta = output_layer[:, 1]
            self._loss = tf.reduce_mean(tf.lbeta(output_layer) -
                                        (self._alpha - 1) * tf.log(target_flat) -
                                        (self._beta - 1) * tf.log1p(-target_flat))

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self._train_step = tf.train.AdamOptimizer().minimize(self._loss)

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

    def train_on_batch(self, batch, loss_list=None):
        self._create_session()
        x, y, _ = self._concatenate_batch(batch)
        feed_dict = {self._input_layer: x, self._target: y, self._is_training: True}
        _, loss = self.session.run([self._train_step, self._loss], feed_dict=feed_dict)
        _append_result(loss, loss_list, accept_none=True)
        return batch

    def test_on_batch(self, batch, loss_list=None):
        self._create_session()
        x, y, _ = self._concatenate_batch(batch)
        feed_dict = {self._input_layer: x, self._target: y, self._is_training: False}
        loss = self.session.run(self._loss, feed_dict=feed_dict)
        _append_result(loss, loss_list)
        return batch

    def predict_on_batch(self, batch, predictions_list=None):
        self._create_session()
        x, _, split_indices = self._concatenate_batch(batch)
        feed_dict = {self._input_layer: x, self._is_training: False}
        alpha, beta = self.session.run([self._alpha, self._beta], feed_dict=feed_dict)
        alpha = np.split(alpha, split_indices)
        beta = np.split(beta, split_indices)
        for a, b in zip(alpha, beta):
            mean = np.mean(a / (a + b))
            var = np.mean(a*b / ((a + b)**2 * (a + b + 1)) + (a / (a + b))**2) - mean**2
            res_dict = dict(zip(batch.label_binarizer.classes_, (1 - mean, mean)))
            res_dict["uncertainty"] = 4 * var
            _append_result(res_dict, predictions_list)
        return batch
