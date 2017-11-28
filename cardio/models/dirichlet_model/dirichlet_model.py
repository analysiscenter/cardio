"""Contains Dirichlet model class."""

from itertools import zip_longest

import numpy as np
import tensorflow as tf

from ..layers import conv1d_block, resnet1d_block
from ...dataset.dataset.models.tf import TFModel


def concatenate_ecg_batch(batch, model, return_targets=True):
    """Concatenate batch signals and (optionally) targets.

    Parameters
    ----------
    batch : EcgBatch
        Batch to concatenate.
    model : BaseModel
        A model to get the resulting arguments.
    return_targets : bool
        Specifies whether to return concatenated targets.

    Returns
    -------
    kwargs : dict
        Named argments for model's train or predict method. Has the following
        structure:
        "feed_dict" : dict
            "signals" : 3-D ndarray
                Concatenated signals.
            "targets" : 2-D ndarray, optional
                Concatenated targets.
        "split_indices" : 1-D ndarray
            Split indices to undo the concatenation.
    """
    _ = model
    x = np.concatenate(batch.signal)
    split_indices = np.cumsum([item.signal.shape[0] for item in batch])[:-1]
    res_dict = {"feed_dict": {"signals": x}, "split_indices": split_indices}
    if return_targets:
        y = np.concatenate([np.tile(item.target, (item.signal.shape[0], 1)) for item in batch])
        res_dict["feed_dict"]["targets"] = y
    return res_dict


class DirichletModelBase(TFModel):
    """Dirichlet model class.

    The model predicts Dirichlet distribution parameters from which class
    probabilities are sampled.

    Notes
    -----
    **Configuration**

    Model config must contain the following keys:
    "input_shape" : tuple
        Input signals's shape without the batch dimension.
    "class_names" : array_like
        Class names.
    "loss" : None
        The model has a predefined loss, so you should leave it ``None``.
    """

    def _build(self, config=None):  # pylint: disable=too-many-locals
        """Build Dirichlet model."""
        input_shape = self.config["input_shape"]
        class_names = self.config["class_names"]

        with self:  # pylint: disable=not-context-manager
            self.store_to_attr("class_names", tf.constant(class_names))

            signals = tf.placeholder(tf.float32, shape=(None,) + input_shape, name="signals")
            self.store_to_attr("signals", signals)
            signals_channels_last = tf.transpose(signals, perm=[0, 2, 1], name="signals_channels_last")

            k = 0.001
            targets = tf.placeholder(tf.float32, shape=(None, len(class_names)), name="targets")
            self.store_to_attr("targets", targets)
            targets_soft = (1 - 2 * k) * targets + k

            block = conv1d_block("conv", signals_channels_last, is_training=self.is_training,
                                 filters=8, kernel_size=5)

            block_config = [
                (8, 3, True),
                (8, 3, False),
                (8, 3, True),
                (8, 3, False),
                (12, 3, True),
                (12, 3, False),
                (12, 3, True),
                (12, 3, False),
                (16, 3, True),
                (16, 3, False),
                (16, 3, False),
                (16, 3, True),
                (16, 3, False),
                (16, 3, False),
                (20, 3, True),
                (20, 3, False),
            ]
            for i, (filters, kernel_size, downsample) in enumerate(block_config):
                block = resnet1d_block("block_" + str(i + 1), block, is_training=self.is_training,
                                       filters=filters, kernel_size=kernel_size, downsample=downsample)

            with tf.variable_scope("global_max_pooling"):  # pylint: disable=not-context-manager
                block = tf.reduce_max(block, axis=1)

            with tf.variable_scope("dense"):  # pylint: disable=not-context-manager
                dense = tf.layers.dense(block, len(class_names), use_bias=False, name="dense")
                bnorm = tf.layers.batch_normalization(dense, training=self.is_training, name="batch_norm", fused=True)
                act = tf.nn.softplus(bnorm, name="activation")

            parameters = tf.identity(act, name="parameters")
            self.store_to_attr("parameters", parameters)
            predictions = tf.identity(act, name="predictions")
            self.store_to_attr("predictions", predictions)
            loss = tf.reduce_mean(tf.lbeta(parameters) -
                                  tf.reduce_sum((parameters - 1) * tf.log(targets_soft), axis=1), name="loss")
            tf.losses.add_loss(loss)


class DirichletModel(DirichletModelBase):
    """Dirichlet model with overloaded train and predict methods.

    ``train`` method is identical to ``DirichletModelBase.train``, but also
    accepts ``args`` and ``kwargs``.
    ``predict`` method splits the resulting tensor for ``parameters`` fetch
    using ``split_indices``. It also splits and aggregates results for
    ``predictions`` fetch to get class probabilities.
    """

    @staticmethod
    def _get_dirichlet_mixture_stats(alpha):
        """Get mean and variance vectors of the mixture of Dirichlet
        distributions with equal weights and given parameters.

        Parameters
        ----------
        alpha : 2-D ndarray
            Dirichlet distribution parameters along axis 1 for each mixture
            component.

        Returns
        -------
        mean : 1-D ndarray
            Mean of the mixture.
        var : 1-D ndarray
            Variance of the mixture.
        """
        alpha_sum = np.sum(alpha, axis=1)[:, np.newaxis]
        comp_m1 = alpha / alpha_sum
        comp_m2 = (alpha * (alpha + 1)) / (alpha_sum * (alpha_sum + 1))
        mean = np.mean(comp_m1, axis=0)
        var = np.mean(comp_m2, axis=0) - mean**2
        return mean, var

    def train(self, fetches=None, feed_dict=None, use_lock=False, *args, **kwargs):
        """Train the model with the data provided.

        The only difference between ``DirichletModel.train`` and
        ``TFModel.train`` is that the former also accepts ``args`` and
        ``kwargs``.

        Parameters
        ----------
        fetches : tf.Operation or tf.Tensor or array-like sequence of them
            Graph element to evaluate in addition to ``train_step``.
        feed_dict : dict
            A dictionary that maps graph elements to values.
        use_lock : bool
            If ``True``, the whole train step is locked, thus allowing for
            multithreading.

        Returns
        -------
        output : same structure as ``fetches``
            Calculated values for each element in ``fetches``.
        """
        _ = args, kwargs
        return super().train(fetches, feed_dict, use_lock)

    def predict(self, fetches=None, feed_dict=None, split_indices=None):  # pylint: disable=arguments-differ
        """Get predictions on the data provided.

        Parameters
        ----------
        fetches : tf.Operation or tf.Tensor or array-like sequence of them
            Graph element to evaluate.
            If ``fetches`` contains ``parameters`` tensor, the corresponding
            output is split using ``split_indices``.
            If ``fetches`` contains ``predictions`` tensor, the corresponding
            output is split using ``split_indices`` and then aggregated to get
            class probabilities.
        feed_dict : dict
            A dictionary that maps graph elements to values.
        split_indices : 1-D ndarray
            Indices used to split ``parameters`` and ``predictions`` tensors.

        Returns
        -------
        output : same structure as ``fetches``
            Calculated values for each element in ``fetches``.
        """
        if isinstance(fetches, (list, tuple)):
            fetches_list = list(fetches)
        else:
            fetches_list = [fetches]
        output = super().predict(fetches_list, feed_dict)
        for i, fetch in enumerate(fetches_list):
            if fetch == "parameters":
                output[i] = np.split(output[i], split_indices)
            elif fetch == "predictions":
                class_names = self.class_names.eval(session=self.session)  # pylint: disable=no-member
                class_names = [c.decode("utf-8") for c in class_names]
                n_classes = len(class_names)
                max_var = (n_classes - 1) / n_classes**2
                alpha = np.split(output[i], split_indices)
                targets = feed_dict.get("targets")
                targets = [] if targets is None else [t[0] for t in np.split(targets, split_indices)]
                res = []
                for a, t in zip_longest(alpha, targets):
                    mean, var = self._get_dirichlet_mixture_stats(a)
                    uncertainty = var[np.argmax(mean)] / max_var
                    predictions_dict = {"target_pred": dict(zip(class_names, mean)),
                                        "uncertainty": uncertainty}
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
