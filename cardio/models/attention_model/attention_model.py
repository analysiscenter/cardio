from itertools import zip_longest

import numpy as np
import tensorflow as tf

from ..layers import resnet1d_block, attention1d_block
from ...dataset.dataset.models.tf import TFModel


class AttentionModel(TFModel):
    def _build(self, config=None):  # pylint: disable=too-many-locals
        """Build Dirichlet model."""
        input_shape = self.config["input_shape"]
        class_names = self.config["class_names"]

        with self:  # pylint: disable=not-context-manager
            self.store_to_attr("class_names", tf.constant(class_names))

            signals = tf.placeholder(tf.float32, shape=(None,) + input_shape, name="signals")
            self.store_to_attr("signals", signals)
            signals_channels_last = tf.transpose(signals, perm=[0, 2, 1], name="signals_channels_last")

            targets = tf.placeholder(tf.float32, shape=(None, len(class_names)), name="targets")
            self.store_to_attr("targets", targets)

            block = signals_channels_last
            print(block.get_shape())

            block_config = [
                (15, 5, 2, True),
                (15, 5, 2, True),
            ]
            for i, (filters, kernel_size, dilation_rate, downsample) in enumerate(block_config):
                block = resnet1d_block("block_" + str(i + 1), block, is_training=self.is_training,
                                       filters=filters, kernel_size=kernel_size, dilation_rate=dilation_rate,
                                       downsample=downsample)
                block = tf.layers.dropout(block, rate=0.25, training=self.is_training)
                print("block_" + str(i + 1), block.get_shape())

            block_config = [
                (20, 5, 2, True, 4),
                (20, 5, 2, True, 4),
                (20, 5, 2, True, 4),
            ]
            for i, (filters, kernel_size, dilation_rate, downsample, downsample_mask) in enumerate(block_config):
                block = attention1d_block("attention_block_" + str(i + 1), block, is_training=self.is_training,
                                          filters=filters, kernel_size=kernel_size, downsample_mask=downsample_mask,
                                          dilation_rate=dilation_rate, downsample=downsample)
                block = tf.layers.dropout(block, rate=0.25, training=self.is_training)
                print("attention_block_" + str(i + 1), block.get_shape())

            block_config = [
                (25, 5, 2, True),
                (25, 5, 2, True),
            ]
            for i, (filters, kernel_size, dilation_rate, downsample) in enumerate(block_config):
                block = resnet1d_block("end_block_" + str(i + 1), block, is_training=self.is_training,
                                       filters=filters, kernel_size=kernel_size, dilation_rate=dilation_rate,
                                       downsample=downsample)
                block = tf.layers.dropout(block, rate=0.25, training=self.is_training)
                print("end_block_" + str(i + 1), block.get_shape())

            with tf.variable_scope("global_max_pooling"):  # pylint: disable=not-context-manager
                block = tf.reduce_max(block, axis=1)
            print(block.get_shape())

            with tf.variable_scope("dense"):  # pylint: disable=not-context-manager
                dense = tf.layers.dense(block, len(class_names), use_bias=False, name="dense")
                bnorm = tf.layers.batch_normalization(dense, training=self.is_training, name="batch_norm", fused=True)
                act = tf.nn.softmax(bnorm, name="activation")

            predictions = tf.identity(act, name="predictions")
            self.store_to_attr("predictions", predictions)
            loss = tf.losses.softmax_cross_entropy(targets, bnorm)

            with self.graph.as_default():
                print(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))

    def train(self, fetches=None, feed_dict=None, use_lock=False, *args, **kwargs):
        _ = args, kwargs
        return super().train(fetches, feed_dict, use_lock)

    def predict(self, fetches=None, feed_dict=None, split_indices=None):  # pylint: disable=arguments-differ
        if isinstance(fetches, (list, tuple)):
            fetches_list = list(fetches)
        else:
            fetches_list = [fetches]
        output = super().predict(fetches_list, feed_dict)
        for i, fetch in enumerate(fetches_list):
            if fetch == "predictions":
                class_names = self.class_names.eval(session=self.session)  # pylint: disable=no-member
                class_names = [c.decode("utf-8") for c in class_names]
                probs = np.split(output[i], split_indices)
                targets = feed_dict.get("targets")
                targets = [] if targets is None else [t[0] for t in np.split(targets, split_indices)]
                res = []
                for a, t in zip_longest(probs, targets):
                    mean = np.mean(a, axis=0)
                    predictions_dict = {"target_pred": dict(zip(class_names, mean))}
                    if t is not None:
                        predictions_dict["target_true"] = dict(zip(class_names, t))
                    res.append(predictions_dict)
                output[i] = res
        if isinstance(fetches, list):
            pass
        elif isinstance(fetches, tuple):
            output = tuple(output)
        else:
            output = output[0]
        return output
