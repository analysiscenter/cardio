"""Model and model tools for ECG"""

from keras.layers import Input, Conv1D, Lambda, \
                         MaxPooling1D, MaxPooling2D, \
                         Dense, GlobalMaxPooling2D
from keras.layers.core import Dropout
from keras.models import Model, load_model
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf

from ..ecg_base_model import EcgBaseModel
from ..keras_custom_objects import RFFT, Crop, Inception2D

class FFTModel(EcgBaseModel):#pylint: disable=too-many-locals
    '''
    FFT inception model. Includes initial convolution layers, then FFT transform, then
    a series of inception blocks.
    '''
    def __init__(self):
        super().__init__()

    def build(self):#pylint: disable=too-many-locals
        '''
        Build and compile model
        '''
        with tf.variable_scope('fft_model'):#pylint: disable=not-context-manager
            x = Input((None, 1))

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

            opt = Adam()
            self.model = Model(inputs=x, outputs=fc_2)
            self.model.compile(optimizer=opt, loss="binary_crossentropy")

            return self

    def load(self, fname):#pylint: disable=arguments-differ
        '''
        Load keras model
        '''
        custom_objects = {'RFFT': RFFT, 'Crop': Crop, 'Inception2D': Inception2D}
        self.model = load_model(fname, custom_objects=custom_objects)
        return self
