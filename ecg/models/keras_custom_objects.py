""" Contains keras custom objects """

import tensorflow as tf
from keras.engine.topology import Layer
from keras.layers import Input, Conv1D, Conv2D, Lambda, \
                         MaxPooling1D, MaxPooling2D, \
                         TimeDistributed, BatchNormalization, Activation
from keras.layers.core import Dropout
from keras.layers.merge import Concatenate
import keras.backend as K

def conv_block(x, filters, kernel_size, activation, timedist):
    """Apply Conv1D, then BatchNormalization, then Activation.

    Parameters
    ----------
    x : tensor
        Input tensor with shape (nb_series, siglen, nb_channels).
    filters : int
        Number of filters in Conv1D.
    kernel_size : int
        Kernel_size for Conv1D.
    activation : string
        Neuron activation function.
    timedist : bool
        True if input has temporal dimension.

    Returns
    -------
    output : tensor
        Resulting tensor
    """
    if timedist:
        conv = TimeDistributed(Conv1D(filters, kernel_size, padding='same'))(x)
    else:
        conv = Conv1D(filters, kernel_size, padding='same')(x)
    b_norm = BatchNormalization()(conv)
    return Activation(activation)(b_norm)

def conv_block_series(x, filters, kernel_size, activation, timedist,
                      repeat=1, max_pool=True, dropout=0):
    """Series of conv_block repeated and followed by maxpooling and dropout.

    Parameters
    ----------
    x : tensor
        Input tensor with shape (nb_series, siglen, nb_channels).
    filters : int
        Number of filters in Conv1D.
    kernel_size : int
        Kernel_size for Conv1D.
    activation : string
        Neuron activation function.
    timedist : bool
        True if input has temporal dimension.
    repeat : positive int
        Number or times to repeat distributed_conv. Default 1.
    max_pool : bool
        If True, maxpooling is applied. Default True.
    dropout : float in [0, 1]
        Parameter for dropout layer. Default 0.
    
    Returns
    -------
    output : tensor
    """
    conv = conv_block(x, filters, kernel_size, activation, timedist)
    for _ in range(repeat - 1):
        conv = conv_block(conv, filters, kernel_size, activation, timedist)
    if max_pool:
        if timedist:
            conv = TimeDistributed(MaxPooling1D())(conv)
        else:
            conv = MaxPooling1D()(conv)
    return Dropout(dropout)(conv)

def cos_metr(a, b):
    """
    Cosine distance between slices along last axis of tensors a and b. Distance is scaled to [0, 1].

    Parameters
    ----------
    a : tensor
        Tensor of shape (batch_size, emb_length).
    b : tensor
        Tensor of shape (batch_size, emb_length).

    Returns
    -------
    output : tensor
        Reduced tensor
    """
    a = a / K.tf.norm(a, ord=2, axis=-1, keep_dims=True)
    b = b / K.tf.norm(b, ord=2, axis=-1, keep_dims=True)
    return (K.tf.reduce_sum(a * b, axis=1, keep_dims=True) + 1.) / 2

def triplet_distance(x):
    """
    Triplet distance between anchor, positive and negative ecg segments in triplet.

    Parameters
    ----------
    x : tensor
        Tensor of shape (batch_size, component, emb_length).

    Returns
    -------
    output : tensor
        Concatenated tensor of cosine distances between positive
        and negative items.
    """
    a = x[:, 0] #anchor item
    pos = x[:, 1] #positive item
    neg = x[:, 2] #negative item
    d_pos = cos_metr(a, pos)
    d_neg = cos_metr(a, neg)
    return K.tf.concat([d_pos, d_neg], axis=-1)

def total_loss(y_true, y_pred):
    '''
    Loss function for triplets.

    Parameters
    ----------
    y_true : tensor
        Any tensor of shape (batch_size, 1), not used for computation.
    y_pred : tensor
        Tensor of shape (batch_size, 2) with predicted anchor to positive
        and anchor to negative embedding distances.
    '''
    _ = y_true
    return K.mean(-(y_pred[:, 0] - y_pred[:, 1]))


class RFFT(Layer):
    '''
    Keras layer for one-dimensional discrete Fourier Transform for real input.
    Computes rfft transforn on each slice along last dim.

    Input shape
    3D tensor (batch_size, signal_length, nb_channels)

    Output shape
    3D tensor (batch_size, int(signal_length / 2), nb_channels)
    '''
    def __init__(self, *agrs, **kwargs):
        super(RFFT, self).__init__(*agrs, **kwargs)

    def rfft(self, x, fft_fn):
        '''
        Computes one-dimensional discrete Fourier Transform on each slice along last dim.
        Returns amplitude spectrum.

        Parameters
        ----------
        x : tensot
            3D tensor (batch_size, signal_length, nb_channels)
        fft_fn : function
            Function that performs fft
        '''
        resh = K.cast(K.map_fn(K.transpose, x), dtype='complex64')
        spec = K.abs(K.map_fn(fft_fn, resh))
        out = K.cast(K.map_fn(K.transpose, spec), dtype='float32')
        shape = tf.shape(out)
        new_shape = [shape[0], shape[1] // 2, shape[2]]
        out_real = tf.slice(out, [0, 0, 0], new_shape)
        return out_real

    def call(self, x):
        return Lambda(self.rfft, arguments={'fft_fn': K.tf.fft})(x)

    def compute_output_shape(self, input_shape):
        '''
        Get output shape
        '''
        if input_shape[1] is None:
            return input_shape
        else:
            return (input_shape[0], input_shape[1] // 2, input_shape[2])


class Crop(Layer):
    '''
    Keras layer returns cropped signal.

    Parameters
    ----------
    begin : int
        Begin of the cropped segment
    size : int
        Size of the cropped segment

    Input shape
    3D tensor (batch_size, signal_length, nb_channels)

    Output shape
    3D tensor (batch_size, size, nb_channels)
    '''
    def __init__(self, begin, size, *agrs, **kwargs):
        self.begin = begin
        self.size = size
        super(Crop, self).__init__(*agrs, **kwargs)

    def call(self, x):
        return x[:, self.begin: self.begin + self.size, :]

    def compute_output_shape(self, input_shape):
        '''
        Get output shape
        '''
        return (input_shape[0], self.size, input_shape[2])


class Inception2D(Layer):#pylint: disable=too-many-instance-attributes
    '''
    Keras layer implements inception block.

    Parameters
    ----------
    base_dim : int
        nb_filters for the first convolution layers.
    nb_filters : int
        nb_filters for the second convolution layers.
    kernel_size_1 : int
        Kernel_size for the second convolution layer.
    kernel_size_2 : int
        Kernel_size for the second convolution layer.
    activation : string
        Activation function for each convolution, default is 'linear'.

    Input shape
    4D tensor (batch_size, width, height, nb_channels)

    Output shape
    4D tensor (batch_size, width, height, 3 * nb_filters + base_dim)
    '''
    def __init__(self, base_dim, nb_filters,#pylint: disable=too-many-arguments
                 kernel_size_1, kernel_size_2,
                 activation=None, *agrs, **kwargs):
        self.base_dim = base_dim
        self.nb_filters = nb_filters
        self.kernel_size_1 = kernel_size_1
        self.kernel_size_2 = kernel_size_2
        self.activation = activation if activation is not None else 'linear'
        self.layers = {}
        super(Inception2D, self).__init__(*agrs, **kwargs)

    def build(self, input_shape):
        x = Input(input_shape[1:])
        self.layers.update({'conv_1': Conv2D(self.base_dim, (1, 1), activation=self.activation, padding='same')})
        _ = self.layers['conv_1'](x)

        self.layers.update({'conv_2': Conv2D(self.base_dim, (1, 1), activation=self.activation, padding='same')})
        out = self.layers['conv_2'](x)
        self.layers.update({'conv_2a': Conv2D(self.nb_filters, (self.kernel_size_1, self.kernel_size_1),
                                              activation=self.activation, padding='same')})
        out = self.layers['conv_2a'](out)

        self.layers.update({'conv_3': Conv2D(self.base_dim, (1, 1), activation=self.activation, padding='same')})
        out = self.layers['conv_3'](x)
        self.layers.update({'conv_3a': Conv2D(self.nb_filters, (self.kernel_size_2, self.kernel_size_2),
                                              activation=self.activation, padding='same')})
        out = self.layers['conv_3a'](out)

        self.layers.update({'conv_4': Conv2D(self.nb_filters, (1, 1), activation=self.activation, padding='same')})
        _ = self.layers['conv_4'](x)

        for layer in self.layers.values():
            self.trainable_weights.extend(layer.trainable_weights)

        return super(Inception2D, self).build(input_shape)

    def call(self, x):
        conv_1 = self.layers['conv_1'](x)

        conv_2 = self.layers['conv_2'](x)
        conv_2a = self.layers['conv_2a'](conv_2)

        conv_3 = self.layers['conv_3'](x)
        conv_3a = self.layers['conv_3a'](conv_3)

        pool = MaxPooling2D(strides=(1, 1), padding='same')(x)
        conv_4 = self.layers['conv_4'](pool)

        return Concatenate(axis=-1)([conv_1, conv_2a, conv_3a, conv_4])

    def compute_output_shape(self, input_shape):
        '''
        Get output shape
        '''
        return (*input_shape[:-1], self.base_dim + 3 * self.nb_filters)
