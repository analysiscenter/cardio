""" Contains keras custom objects """

import tensorflow as tf
from keras.engine.topology import Layer
from keras.layers import Input, Conv2D, Lambda, MaxPooling2D
from keras.layers.merge import Concatenate
import keras.backend as K

class RFFT(Layer):
    '''
    Keras layer for one-dimensional discrete Fourier Transform for real input.
    Computes rfft transforn on each slice along last dim.

    Input shape
    3D tensor (batch_size, signal_length, nb_channels)

    Output shape
    3D tensor (batch_size, int(signal_length / 2), nb_channels)
    '''

    def rfft(self, x, fft_fn):
        '''
        Computes one-dimensional discrete Fourier Transform on each slice along last dim.
        Returns amplitude spectrum.

        Parameters
        ----------
        x : tensor
            3D tensor (batch_size, signal_length, nb_channels)
        fft_fn : function
            Function that performs fft

        Returns
        -------
        out : 3D tensor
            Computed fft(x). 3D tensor (batch_size, signal_length // 2, nb_channels)
        '''
        resh = K.cast(K.map_fn(K.transpose, x), dtype='complex64')
        spec = K.abs(K.map_fn(fft_fn, resh))
        out = K.cast(K.map_fn(K.transpose, spec), dtype='float32')
        shape = tf.shape(out)
        new_shape = [shape[0], shape[1] // 2, shape[2]]
        out_real = tf.slice(out, [0, 0, 0], new_shape)
        return out_real

    def call(self, x):
        '''
        Implements Keras call method.
        '''
        return Lambda(self.rfft, arguments={'fft_fn': K.tf.fft})(x)

    def compute_output_shape(self, input_shape):
        ''''
        Implements Keras compute_output_shape method.
        '''
        if input_shape[1] is None:
            return input_shape
        return (input_shape[0], input_shape[1] // 2, input_shape[2])


class Crop(Layer):
    '''
    Keras layer returns cropped signal.

    Input shape
    3D tensor (batch_size, signal_length, nb_channels)

    Output shape
    3D tensor (batch_size, size, nb_channels)

    Parameters
    ----------
    begin : int
        Begin of the cropped segment
    size : int
        Size of the cropped segment

    Attributes
    ----------
    begin : int
        Begin of the cropped segment
    size : int
        Size of the cropped segment
    '''
    def __init__(self, begin, size, *agrs, **kwargs):
        self.begin = begin
        self.size = size
        super(Crop, self).__init__(*agrs, **kwargs)

    def call(self, x):
        '''
        Implements Keras call method.
        '''
        return x[:, self.begin: self.begin + self.size, :]

    def compute_output_shape(self, input_shape):
        '''
        Implements Keras compute_output_shape method.
        '''
        return (input_shape[0], self.size, input_shape[2])


class Inception2D(Layer):#pylint: disable=too-many-instance-attributes
    '''
    Keras layer implements inception block.

    Input shape
    4D tensor (batch_size, width, height, nb_channels)

    Output shape
    4D tensor (batch_size, width, height, 3 * nb_filters + base_dim)

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
        Activation function for each convolution. Default is 'linear'.

    Attributes
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
        Activation function for each convolution. Default is 'linear'.
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
        '''
        Implements Keras build method
        '''
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
        '''
        Implements Keras call method
        '''
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
        Implements Keras compute_output_shape method
        '''
        return (*input_shape[:-1], self.base_dim + 3 * self.nb_filters)
