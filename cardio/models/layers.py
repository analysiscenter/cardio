"""Contains helper functions for tensorflow layers creation."""

import tensorflow as tf


def conv_cell(scope, input_layer, is_training, n_filters, kernel_size, pool_size=2, pool_stride=2, act=tf.nn.elu):
    """Create conv1d -> max_pooling1d -> batch_normalization -> activation layer.

    Parameters
    ----------
    scope : str
        Variable scope name.
    input_layer : 3-D Tensor
        Input tensor with channels_last ordering.
    is_training : 0-D Tensor with bool dtype
        True for training phase and False for inference.
    n_filters : int
        The number of filters in the convolution.
    kernel_size : int
        The length of the convolution kernel.
    pool_size : int
        The size of the pooling window.
    pool_stride : int
        The strides of the pooling operation.
    act : callable
        Activation function.

    Returns
    -------
    tensor : Tensor
        Output tensor.
    """
    with tf.variable_scope(scope):  # pylint: disable=not-context-manager
        conv = tf.layers.conv1d(input_layer, n_filters, kernel_size, padding="same",
                                data_format="channels_last", use_bias=False, name="conv")
        pool = tf.layers.max_pooling1d(conv, pool_size, pool_stride,
                                       data_format="channels_last", name="pool")
        bnorm = tf.layers.batch_normalization(pool, training=is_training, name="batch_norm")
        act = act(bnorm, name="activation")
    return act
