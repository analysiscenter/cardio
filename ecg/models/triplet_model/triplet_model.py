"""Model and model tools for ECG"""

from keras import regularizers
from keras.layers import Input, Conv1D, Lambda, \
                         MaxPooling1D,  Dense, \
                         TimeDistributed, BatchNormalization, \
                         Activation, Flatten
from keras.layers.core import Dropout
from keras.models import Model
import keras.backend as K
import tensorflow as tf

from ..ecg_base_model import EcgBaseModel

class TripletModel(EcgBaseModel):#pylint: disable=too-many-locals
    '''
    Model for triplet learn. Triplet consists of anchor, positive and negative ecg segments.
    Distance between embedding of ecg segments is cosine distance. Model learns to make distanse between
    anchor and positive embeddings smaller as compared to negative embedding.
    '''
    def __init__(self, input_shape):
        super().__init__()
        self._input_shape = input_shape

    def build(self):
        '''
        Build and compile model
        '''
        with tf.variable_scope('triplet_model'):#pylint: disable=not-context-manager
            x = Input(self._input_shape)

            def distributed_conv(x, filters, kernel_size, activation):
                '''
                Apply successively timedistributed 1D convolution, then batchnorm, then activation.
                Arguments
                x: input of shape (nb_series, siglen, nb_channels).
                filters: number of filters in Conv1D.
                kernel_size: kernel_size for Conv1D.
                activation: neuron activation function.
                '''
                conv = TimeDistributed((Conv1D(filters, kernel_size, padding='same')))(x)
                b_norm = BatchNormalization()(conv)
                return Activation(activation)(b_norm)

            def conv_block(x, filters, kernel_size, activation, repeat, max_pool, dropout):
                '''
                Block of several distributed_conv followed by maxpooling and dropout.
                Arguments
                x: input of shape (nb_series, siglen, nb_channels).
                filters: number of filters in Conv1D.
                kernel_size: kernel_size for Conv1D.
                activation: neuron activation function.
                repeat: if true, distributed_conv is applied twice, else once.
                max_pool: if true, maxpooling is applied.
                dropout: parameter for dropout layer.
                '''
                conv = distributed_conv(x, filters, kernel_size, activation)
                if repeat:
                    conv = distributed_conv(conv, filters, kernel_size, activation)
                if max_pool:
                    conv = TimeDistributed(MaxPooling1D())(conv)
                return Dropout(dropout)(conv)

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

            def cos_metr(a, b):
                '''
                Cosine distance between slices along last axis of tensors a and b. Distance is scaled to [0, 1].
                Arguments
                a: tensor of shape (batch_size, emb_length).
                b: tensor of shape (batch_size, emb_length).
                '''
                a = a / K.tf.norm(a, ord=2, axis=-1, keep_dims=True)
                b = b / K.tf.norm(b, ord=2, axis=-1, keep_dims=True)
                return (K.tf.reduce_sum(a * b, axis=1, keep_dims=True) + 1.) / 2

            def triplet_distance(x):
                '''
                Triplet distance between anchor, positive and negative ecg segments in triplet.
                Arguments
                x: tensor of shape (batch_size, component, emb_length).
                '''
                a = x[:, 0] #anchor item
                pos = x[:, 1] #positive item
                neg = x[:, 2] #negative item
                d_pos = cos_metr(a, pos)
                d_neg = cos_metr(a, neg)
                return K.tf.concat([d_pos, d_neg], axis=-1)

            dist = Lambda(triplet_distance)(fc_1)

            def total_loss(y_true, y_pred):
                '''
                Loss function for triplets.
                Arguments
                y_true: any tensor of shape (batch_size, 1), not used.
                y_pred: tensor of shape (batch_size, 2) with predicted anchor to positive
                        and anchor to negative embedding distances.
                '''
                _ = y_true
                return K.mean(-(y_pred[:, 0] - y_pred[:, 1]))

            self.model = Model(inputs=x, outputs=dist)
            self.model.compile(loss=total_loss, optimizer="adam")

            return self
