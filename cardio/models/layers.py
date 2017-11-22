"""Contains helper functions for tensorflow layers creation."""

import tensorflow as tf


def conv1d_block(scope, input_layer, is_training, filters, kernel_size, pool_size=2, pool_stride=2, act_fn=tf.nn.relu):
    """Create conv1d -> max_pooling1d -> batch_normalization -> activation layer.

    Parameters
    ----------
    scope : str
        Variable scope name.
    input_layer : 3-D Tensor
        Input tensor with channels_last ordering.
    is_training : 0-D Tensor with bool dtype
        True for training phase and False for inference.
    filters : int
        The number of filters in the convolution.
    kernel_size : int
        The length of the convolution kernel.
    pool_size : int
        The size of the pooling window.
    pool_stride : int
        The strides of the pooling operation.
    act_fn : callable
        Activation function.

    Returns
    -------
    tensor : Tensor
        Output tensor.
    """
    with tf.variable_scope(scope):  # pylint: disable=not-context-manager
        conv = tf.layers.conv1d(input_layer, filters, kernel_size, padding="same",
                                data_format="channels_last", use_bias=False, name="conv")
        pool = tf.layers.max_pooling1d(conv, pool_size, pool_stride,
                                       data_format="channels_last", name="pool")
        bnorm = tf.layers.batch_normalization(pool, training=is_training, name="batch_norm", fused=True)
        act = act_fn(bnorm, name="activation")
    return act


def resnet1d_block(scope, input_layer, is_training, filters, kernel_size, downsample=False, act_fn=tf.nn.relu):
    """Create 1-D resnet block with two consequent convolutions and shortcut connection.

    Parameters
    ----------
    scope : str
        Variable scope name.
    input_layer : 3-D Tensor
        Input tensor with channels_last ordering.
    is_training : 0-D Tensor with bool dtype
        True for training phase and False for inference.
    filters : int
        The number of filters in both convolutions.
    kernel_size : int
        The length of the convolution's kernels.
    downsample : bool
        Specifies whether to reduce the spatial dimension by the factor of two.
    act_fn : callable
        Activation function.

    Returns
    -------
    tensor : Tensor
        Output tensor.
    """
    with tf.variable_scope(scope):  # pylint: disable=not-context-manager
        strides = 2 if downsample else 1
        conv = tf.layers.conv1d(input_layer, filters, kernel_size, strides, padding="same",
                                data_format="channels_last", use_bias=False, name="conv_1")
        bnorm = tf.layers.batch_normalization(conv, training=is_training, name="batch_norm_1", fused=True)
        act = act_fn(bnorm, name="act_1")
        conv = tf.layers.conv1d(act, filters, kernel_size, padding="same",
                                data_format="channels_last", use_bias=False, name="conv_2")
        if downsample:
            input_layer = tf.layers.max_pooling1d(input_layer, 2, 2, data_format="channels_last", name="pool")
        if input_layer.get_shape()[-1] != conv.get_shape()[-1]:
            input_layer = tf.layers.conv1d(input_layer, conv.get_shape()[-1], 1, padding="same",
                                           data_format="channels_last", use_bias=False, name="conv_3")
        conv = conv + input_layer
        bnorm = tf.layers.batch_normalization(conv, training=is_training, name="batch_norm_2", fused=True)
        act = act_fn(bnorm, name="act_2")
    return act
