""" Contains fft_model architecture """

import tensorflow as tf
import keras.backend as K

from keras.layers import Input, Conv1D, Lambda, \
                         MaxPooling1D, MaxPooling2D, \
                         Dense, GlobalMaxPooling2D
from keras.layers.core import Dropout

from ...batchflow.models.keras import KerasModel #pylint: disable=no-name-in-module, import-error
from ..keras_custom_objects import RFFT, Crop, Inception2D

class FFTModel(KerasModel):#pylint: disable=too-many-locals
    '''
    FFT inception model. Includes initial convolution layers, then FFT transform, then
    a series of inception blocks.
    '''
    def _build(self, **kwargs):#pylint: disable=too-many-locals
        '''
        Build model
        '''
        with tf.variable_scope('fft_model'):#pylint: disable=not-context-manager
            x = Input(kwargs['input_shape'])

            conv_1 = Conv1D(4, 4, activation='relu')(x)
            mp_1 = MaxPooling1D()(conv_1)
            conv_2 = Conv1D(8, 4, activation='relu')(mp_1)
            mp_2 = MaxPooling1D()(conv_2)
            conv_3 = Conv1D(16, 4, activation='relu')(mp_2)
            mp_3 = MaxPooling1D()(conv_3)
            conv_4 = Conv1D(32, 4, activation='relu')(mp_3)

            fft_1 = RFFT()(conv_4)
            crop_1 = Crop(begin=0, size=128)(fft_1)
            to2d = Lambda(K.expand_dims)(crop_1)

            incept_1 = Inception2D(4, 4, 3, 3, activation='relu')(to2d)
            mp2d_1 = MaxPooling2D(pool_size=(4, 2))(incept_1)

            incept_2 = Inception2D(4, 8, 3, 3, activation='relu')(mp2d_1)
            mp2d_2 = MaxPooling2D(pool_size=(4, 2))(incept_2)

            incept_3 = Inception2D(4, 12, 3, 3, activation='relu')(mp2d_2)

            pool = GlobalMaxPooling2D()(incept_3)

            fc_1 = Dense(8, kernel_initializer='uniform', activation='relu')(pool)
            drop = Dropout(0.2)(fc_1)

            fc_2 = Dense(2, kernel_initializer='uniform',
                         activation='softmax')(drop)

            return x, fc_2
