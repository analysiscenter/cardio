"""Model and model tools for ECG"""

from keras import regularizers
from keras.layers import Input, GlobalMaxPooling1D, Dense, \
                         BatchNormalization, Activation
from keras.layers.core import Dropout
from keras.models import Model, load_model
import tensorflow as tf

from ..keras_base_model import KerasBaseModel
from ..keras_custom_objects import conv_block, conv_block_series

class ConvModel(KerasBaseModel):#pylint: disable=too-many-locals
    '''
    Convolution model.
    '''#pylint: disable=duplicate-code
    def __init__(self):
        super().__init__()
        self._input_shape = None

    def build(self, input_shape):
        '''
        Build and compile conv model
        '''
        self._input_shape = input_shape
        with tf.variable_scope('conv_model'):#pylint: disable=not-context-manager
            x = Input(self._input_shape)

            conv_1 = conv_block_series(x, 20, 4, 'elu', False, 1, True, 0)
            conv_2 = conv_block_series(conv_1, 24, 4, 'elu', False, 1, True, 0.2)
            conv_3 = conv_block_series(conv_2, 24, 4, 'elu', False, 1, True, 0.2)
            conv_4 = conv_block_series(conv_3, 24, 4, 'elu', False, 1, True, 0.2)
            conv_5 = conv_block_series(conv_4, 28, 4, 'elu', False, 1, True, 0.2)
            conv_6 = conv_block_series(conv_5, 32, 4, 'elu', False, 1, False, 0)

            res = GlobalMaxPooling1D()(conv_6)
            drop = Dropout(0.2)(res)

            fc_1 = Activation('elu')(BatchNormalization()(Dense(8, kernel_regularizer=regularizers.l2(0.01))(drop)))
            out = Activation('softmax')(BatchNormalization()(Dense(2, kernel_regularizer=regularizers.l2(0.01))(fc_1)))

            self.model = Model(inputs=x, outputs=out)
            self.model.compile(loss="binary_crossentropy", optimizer="adam")

            return self

    def load(self, fname):#pylint: disable=arguments-differ
        '''
        Load keras model
        '''
        custom_objects = {'conv_block': conv_block,
                          'conv_block_series': conv_block_series}
        self.model = load_model(fname, custom_objects=custom_objects)
        return self
