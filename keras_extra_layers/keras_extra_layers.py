""" contain keras custom layers """

from keras.engine.topology import Layer
from keras.layers import Input, Conv2D, \
                         MaxPooling2D, Lambda, \
                         Reshape 
        GlobalMaxPooling2D
from keras.layers.merge import Concatenate
import keras.backend as K

class RFFT(Layer):
    '''
    Keras layer for one-dimensional discrete Fourier Transform for real input.
    Computes rfft transforn on each slice along last dim.

    Arguments
    None

    Input shape
    3D tensor (batch_size, signal_length, nb_channels)

    Output shape
    3D tensor (batch_size, int(signal_length / 2), nb_channels)
    '''
    def __init__(self, *agrs, **kwargs):
        super(RFFT, self).__init__(*agrs, **kwargs)

    def fft(self, x, fft_fn):
        '''
        Computes one-dimensional discrete Fourier Transform on each slice along last dim.
        Returns amplitude spectrum.

        Arguments
        x: 3D tensor (batch_size, signal_length, nb_channels)
        fft_fn: function that performs fft

        Retrun
        out: 3D tensor (batch_size, signal_length, nb_channels) of type tf.float32
        '''
        resh = K.cast(K.map_fn(K.transpose, x), dtype='complex64')
        spec = K.abs(K.map_fn(fft_fn, resh))
        out = K.cast(K.map_fn(K.transpose, spec), dtype='float32')
        return out

    def call(self, x):
        res = Lambda(self.fft, arguments={'fft_fn': K.tf.fft})(x)
        half = int(res.get_shape().as_list()[1] / 2)
        return res[:, :half, :]

    def compute_output_shape(self, input_shape):
        '''
        Get output shape
        '''
        return (input_shape[0], int(input_shape[1] / 2), input_shape[2])


class Crop(Layer):
    '''
    Keras layer returns cropped signal.

    Arguments
    begin: begin of the cropped segment
    size: size of the cropped segment

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


class To2D(Layer):
    '''
    Keras layer add dim to 1D signal and returns 2D image.

    Arguments
    None

    Input shape
    3D tensor (batch_size, signal_length, nb_channels)

    Output shape
    4D tensor (batch_size, size, nb_channels, 1)
    '''
    def __init__(self, *agrs, **kwargs):
        super(To2D, self).__init__(*agrs, **kwargs)

    def call(self, x):
        shape_1d = x.get_shape().as_list()[1:]
        shape_1d.append(1)
        to2d = Reshape(shape_1d)(x)
        return to2d

    def compute_output_shape(self, input_shape):
        '''
        Get output shape
        '''
        return (*input_shape, 1)


class Inception2D(Layer):#pylint: disable=too-many-instance-attributes
    '''
    Keras layer implements inception block.

    Arguments
    base_dim: nb_filters for the first convolution layers.
    nb_filters: nb_filters for the second convolution layers.
    kernel_size_1: kernel_size for the second convolution layer.
    kernel_size_2: kernel_size for the second convolution layer.
    activation: activation function for each convolution, default is 'linear'.

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
        super(Inception2D, self).__init__(*agrs, **kwargs)

    def build(self, input_shape):
        x = Input(input_shape[1:])
        self.conv_1 = Conv2D(self.base_dim, (1, 1),#pylint: disable=attribute-defined-outside-init
                             activation=self.activation, padding='same')
        _ = self.conv_1(x)
        self.trainable_weights.extend(self.conv_1.trainable_weights)

        self.conv_2 = Conv2D(self.base_dim, (1, 1),#pylint: disable=attribute-defined-outside-init
                             activation=self.activation, padding='same')
        out = self.conv_2(x)
        self.trainable_weights.extend(self.conv_2.trainable_weights)
        self.conv_2a = Conv2D(self.nb_filters,#pylint: disable=attribute-defined-outside-init
                              (self.kernel_size_1, self.kernel_size_1),
                              activation=self.activation, padding='same')
        out = self.conv_2a(out)
        self.trainable_weights.extend(self.conv_2a.trainable_weights)

        self.conv_3 = Conv2D(self.base_dim, (1, 1),#pylint: disable=attribute-defined-outside-init
                             activation=self.activation, padding='same')
        out = self.conv_3(x)
        self.trainable_weights.extend(self.conv_3.trainable_weights)
        self.conv_3a = Conv2D(self.nb_filters,#pylint: disable=attribute-defined-outside-init
                              (self.kernel_size_2, self.kernel_size_2),
                              activation=self.activation, padding='same')
        out = self.conv_3a(out)
        self.trainable_weights.extend(self.conv_3a.trainable_weights)

        self.conv_4 = Conv2D(self.nb_filters, (1, 1),#pylint: disable=attribute-defined-outside-init
                             activation=self.activation, padding='same')
        _ = self.conv_4(x)
        self.trainable_weights.extend(self.conv_4.trainable_weights)

        return super(Inception2D, self).build(input_shape)

    def call(self, x):
        conv_1 = self.conv_1(x)

        conv_2 = self.conv_2(x)
        conv_2a = self.conv_2a(conv_2)

        conv_3 = self.conv_3(x)
        conv_3a = self.conv_3a(conv_3)

        pool = MaxPooling2D(strides=(1, 1), padding='same')(x)
        conv_4 = self.conv_4(pool)

        return Concatenate(axis=-1)([conv_1, conv_2a, conv_3a, conv_4])

    def compute_output_shape(self, input_shape):
        '''
        Get output shape
        '''
        return (*input_shape[:-1], self.base_dim + 3 * self.nb_filters)
