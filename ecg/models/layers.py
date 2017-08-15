import tensorflow as tf


def conv_cell(scope, input_layer, is_training, n_filters, kernel_size, pool_size=2, pool_stride=2, act=tf.nn.elu):
    with tf.variable_scope(scope):
        conv = tf.layers.conv1d(input_layer, n_filters, kernel_size, padding="same",
                                data_format="channels_last", use_bias=False, name="conv")
        pool = tf.layers.max_pooling1d(conv, pool_size, pool_stride,
                                       data_format="channels_last", name="pool")
        bnorm = tf.layers.batch_normalization(pool, training=is_training, name="batch_norm")
        act = act(bnorm, name="activation")
    return act
