"""Contains helper functions for tensorflow layers creation."""

import tensorflow as tf
from ..dataset.dataset.models.tf.layers import subpixel_conv, resize_bilinear


def conv1d_block(scope, input_layer, is_training, filters, kernel_size, dilation_rate=1, pool_size=2, pool_stride=2,
                 act_fn=tf.nn.relu):
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
        conv = tf.layers.conv1d(input_layer, filters, kernel_size, dilation_rate=dilation_rate, padding="same",
                                data_format="channels_last", use_bias=False, name="conv")
        pool = tf.layers.max_pooling1d(conv, pool_size, pool_stride,
                                       data_format="channels_last", name="pool")
        bnorm = tf.layers.batch_normalization(pool, training=is_training, name="batch_norm", fused=True)
        act = act_fn(bnorm, name="activation")
    return act


def resnet1d_block(scope, input_layer, is_training, filters, kernel_size, dilation_rate=1, downsample=False,
                   act_fn=tf.nn.relu):
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
        conv = tf.layers.conv1d(input_layer, filters, kernel_size, dilation_rate=dilation_rate, padding="same",
                                data_format="channels_last", use_bias=False, name="conv_1")
        if downsample:
            conv = tf.layers.max_pooling1d(conv, 2, 2, data_format="channels_last", padding="same", name="trunk_pool")
            input_layer = tf.layers.max_pooling1d(input_layer, 2, 2, data_format="channels_last", padding="same",
                                                  name="skip_pool")
        bnorm = tf.layers.batch_normalization(conv, training=is_training, name="batch_norm_1", fused=True)
        act = act_fn(bnorm, name="act_1")
        conv = tf.layers.conv1d(act, filters, kernel_size, dilation_rate=dilation_rate, padding="same",
                                data_format="channels_last", use_bias=False, name="conv_2")
        if input_layer.get_shape()[-1] != conv.get_shape()[-1]:
            input_layer = tf.layers.conv1d(input_layer, conv.get_shape()[-1], 1, padding="same",
                                           data_format="channels_last", use_bias=False, name="conv_3")
        conv = conv + input_layer
        bnorm = tf.layers.batch_normalization(conv, training=is_training, name="batch_norm_2", fused=True)
        act = act_fn(bnorm, name="act_2")
    return act


def attention1d_block(scope, input_layer, is_training, filters, kernel_size, dilation_rate=1, downsample=False,
                      downsample_mask=None, act_fn=tf.nn.relu):
    with tf.variable_scope(scope):  # pylint: disable=not-context-manager
        pre_block = resnet1d_block("pre_block", input_layer, is_training=is_training, filters=filters,
                                   kernel_size=kernel_size, act_fn=act_fn)

        trunk = pre_block
        trunk = resnet1d_block("trunk_block_1", trunk, is_training=is_training, dilation_rate=dilation_rate,
                               filters=filters, kernel_size=kernel_size, act_fn=act_fn)
        trunk = resnet1d_block("trunk_block_2", trunk, is_training=is_training, dilation_rate=dilation_rate,
                               filters=filters, kernel_size=kernel_size, act_fn=act_fn)
        trunk = resnet1d_block("trunk_block_3", trunk, is_training=is_training, dilation_rate=dilation_rate,
                               filters=filters, kernel_size=kernel_size, act_fn=act_fn)

        mask = pre_block
        if downsample_mask is not None:
            filters, downsample_mask = downsample_mask, filters
            mask = tf.layers.conv1d(mask, filters, 1, padding="same",
                                    data_format="channels_last", use_bias=False, name="downsample_conv")
        mask = tf.layers.max_pooling1d(mask, 2, 2, data_format="channels_last", padding="same", name="pool_1")
        mask = resnet1d_block("mask_block_1", mask, is_training=is_training, dilation_rate=dilation_rate,
                              filters=filters, kernel_size=kernel_size, act_fn=act_fn)
        mask2 = tf.layers.max_pooling1d(mask, 2, 2, data_format="channels_last", padding="same", name="pool_2")
        mask2 = resnet1d_block("mask_block_2", mask2, is_training=is_training, dilation_rate=dilation_rate,
                               filters=filters, kernel_size=kernel_size, act_fn=act_fn)
        mask2 = resnet1d_block("mask_block_3", mask2, is_training=is_training, dilation_rate=dilation_rate,
                               filters=filters, kernel_size=kernel_size, act_fn=act_fn)
        mask2 = resize_bilinear(mask2, name="resize_1")[:, :mask.get_shape()[1], :]
        mask = mask + mask2
        mask = resnet1d_block("mask_block_4", mask, is_training=is_training, dilation_rate=dilation_rate,
                              filters=filters, kernel_size=kernel_size, act_fn=act_fn)
        mask = resize_bilinear(mask, name="resize_1")[:, :trunk.get_shape()[1], :]
        if downsample_mask is not None:
            filters, downsample_mask = downsample_mask, filters
            mask = tf.layers.conv1d(mask, filters, 1, padding="same",
                                    data_format="channels_last", use_bias=False, name="upsample_conv")
#         mask = tf.layers.batch_normalization(mask, training=is_training, name="batch_norm", fused=True)

#       --------------------------------

#         if downsample_mask is not None:
#             filters, downsample_mask = downsample_mask, filters
#             mask = tf.layers.conv1d(mask, filters, 1, padding="same",
#                                     data_format="channels_last", use_bias=True, name="downsample_conv")
#         mask = tf.layers.conv1d(mask, filters, kernel_size, dilation_rate=2, padding="same",
#                                 data_format="channels_last", use_bias=False, name="conv_1")
#         mask = tf.layers.batch_normalization(mask, training=is_training, name="batch_norm_1", fused=True)
#         mask = act_fn(mask, name="activation_1")
        
#         mask = tf.layers.conv1d(mask, filters, kernel_size, dilation_rate=2, padding="same",
#                                 data_format="channels_last", use_bias=False, name="conv_2")
#         mask = tf.layers.batch_normalization(mask, training=is_training, name="batch_norm_2", fused=True)
#         mask = act_fn(mask, name="activation_2")
        
#         mask = tf.layers.conv1d(mask, filters, kernel_size, dilation_rate=2, padding="same",
#                                 data_format="channels_last", use_bias=False, name="conv_3")
#         if downsample_mask is not None:
#             filters, downsample_mask = downsample_mask, filters
#             mask = tf.layers.conv1d(mask, filters, 1, padding="same",
#                                     data_format="channels_last", use_bias=False, name="upsample_conv")
#         mask = tf.layers.batch_normalization(mask, training=is_training, name="batch_norm_3", fused=True)
        
        mask = tf.nn.sigmoid(mask) + 1

        post_block = trunk * mask
        post_block = resnet1d_block("post_block", post_block, is_training=is_training, filters=filters,
                                    kernel_size=kernel_size, downsample=downsample, act_fn=act_fn)
        return post_block

    
def attention1d_block_old(scope, input_layer, is_training, filters, kernel_size, dilation_rate=1, downsample=False,
                      downsample_mask=None, act_fn=tf.nn.relu):
    with tf.variable_scope(scope):  # pylint: disable=not-context-manager
        pre_block = resnet1d_block("pre_block", input_layer, is_training=is_training, filters=filters,
                                   kernel_size=kernel_size, act_fn=act_fn)

        with tf.variable_scope("trunk"):
            trunk = pre_block
            
            trunk = resnet1d_block("trunk_block_1", trunk, is_training=is_training, dilation_rate=dilation_rate,
                                   filters=filters, kernel_size=kernel_size, act_fn=act_fn)
            trunk = resnet1d_block("trunk_block_2", trunk, is_training=is_training, dilation_rate=dilation_rate,
                                   filters=filters, kernel_size=kernel_size, act_fn=act_fn)

#             trunk = tf.layers.conv1d(trunk, filters, kernel_size, dilation_rate=dilation_rate, padding="same",
#                                      data_format="channels_last", use_bias=False, name="conv_1")
#             trunk = tf.layers.batch_normalization(trunk, training=is_training, name="batch_norm_1", fused=True)
#             trunk = act_fn(trunk, name="activation_1")

#             trunk = tf.layers.conv1d(trunk, filters, kernel_size, dilation_rate=dilation_rate, padding="same",
#                                      data_format="channels_last", use_bias=False, name="conv_2")
#             trunk = tf.layers.batch_normalization(trunk, training=is_training, name="batch_norm_2", fused=True)
#             trunk = act_fn(trunk, name="activation_2")

#             trunk = tf.layers.conv1d(trunk, filters, kernel_size, dilation_rate=dilation_rate, padding="same",
#                                      data_format="channels_last", use_bias=False, name="conv_3")
#             trunk = tf.layers.batch_normalization(trunk, training=is_training, name="batch_norm_3", fused=True)
#             trunk = act_fn(trunk, name="activation_3")

        with tf.variable_scope("mask"):
            mask = pre_block

            filters, downsample_mask = downsample_mask, filters
            mask = tf.layers.conv1d(mask, filters, 1, padding="same",
                                    data_format="channels_last", use_bias=False, name="downsample_conv")
            
            mask = tf.layers.conv1d(mask, filters, kernel_size, dilation_rate=4, padding="same",
                                    data_format="channels_last", use_bias=False, name="conv_1")
            mask = tf.layers.batch_normalization(mask, training=is_training, name="batch_norm_1", fused=True)
            mask = act_fn(mask, name="activation_1")
            
            mask = tf.layers.conv1d(mask, filters, kernel_size, dilation_rate=4, padding="same",
                                    data_format="channels_last", use_bias=False, name="conv_2")
            mask = tf.layers.batch_normalization(mask, training=is_training, name="batch_norm_2", fused=True)
            mask = act_fn(mask, name="activation_2")
            
            mask = tf.layers.conv1d(mask, filters, kernel_size, dilation_rate=4, padding="same",
                                    data_format="channels_last", use_bias=False, name="conv_3")
            mask = tf.layers.batch_normalization(mask, training=is_training, name="batch_norm_3", fused=True)
            mask = act_fn(mask, name="activation_3")
            
            filters, downsample_mask = downsample_mask, filters
            mask = tf.layers.conv1d(mask, filters, 1, padding="same",
                                    data_format="channels_last", use_bias=False, name="upsample_conv")
            
            mask = tf.layers.batch_normalization(mask, training=is_training, name="batch_norm_4", fused=True)
            mask = tf.nn.sigmoid(mask)

        post_block = trunk * mask
        post_block = resnet1d_block("post_block", post_block, is_training=is_training, filters=filters,
                                    kernel_size=kernel_size, downsample=downsample, act_fn=act_fn)
        return post_block
