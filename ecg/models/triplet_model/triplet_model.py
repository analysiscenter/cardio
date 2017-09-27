"""Model and model tools for ECG"""

from keras import regularizers
from keras.layers import Input, Conv1D, Lambda, \
                         MaxPooling1D, Dense, \
                         TimeDistributed, BatchNormalization, \
                         Activation, Flatten
from keras.layers.core import Dropout
from keras.models import Model, load_model
import keras.backend as K
import tensorflow as tf

from ..ecg_base_model import EcgBaseModel
from ..keras_custom_objects import distributed_conv, conv_block, \
                                  cos_metr, triplet_distance, total_loss


class TripletModel(EcgBaseModel):#pylint: disable=too-many-locals
    '''
    Model for triplet learn. Triplet consists of anchor, positive and negative ecg segments.
    Distance between embedding of ecg segments is cosine distance. Model learns to make distanse between
    anchor and positive embeddings smaller as compared to negative embedding.
    '''
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        '''
        Build and compile model
        '''
        self._input_shape = input_shape
        with tf.variable_scope('triplet_model'):#pylint: disable=not-context-manager
            x = Input(self._input_shape)

            conv_block_1 = conv_block(x, 4, 4, 'elu', True, True, 0)
            conv_block_2 = conv_block(conv_block_1, 8, 4, 'elu', True, True, 0)
            conv_block_3 = conv_block(conv_block_2, 8, 4, 'elu', True, True, 0)
            conv_block_4 = conv_block(conv_block_3, 16, 4, 'elu', True, True, 0.2)
            conv_block_5 = conv_block(conv_block_4, 16, 4, 'elu', True, True, 0.2)
            conv_block_6 = conv_block(conv_block_5, 24, 4, 'elu', True, True, 0.2)
            conv_block_7 = conv_block(conv_block_6, 24, 4, 'elu', True, True, 0.2)
            conv_block_8 = conv_block(conv_block_7, 32, 4, 'elu', False, True, 0.2)
            conv_block_9 = conv_block(conv_block_8, 48, 4, 'elu', False, False, 0.2)

            flat = TimeDistributed(Flatten())(conv_block_9)

            fc_1 = TimeDistributed(Dense(16, kernel_regularizer=regularizers.l2(0.02)))(flat)
            fc_1 = Activation('elu')(BatchNormalization()(fc_1))

            dist = Lambda(triplet_distance)(fc_1)

            self.model = Model(inputs=x, outputs=dist)
            self.model.compile(loss=total_loss, optimizer="adam")

            return self
    
    def load(self, fname):#pylint: disable=arguments-differ
        '''
        Load keras model
        '''
        custom_objects = {'cos_metr': cos_metr,
                          'triplet_distance': triplet_distance,
                          'total_loss': total_loss,
                          'conv_block': conv_block,
                          'distributed_conv': distributed_conv
                         }
        self.model = load_model(fname, custom_objects = custom_objects)
        return self
