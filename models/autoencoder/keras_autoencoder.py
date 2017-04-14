import numpy as np

import tensorflow as tf
from keras import backend as K
from keras import metrics
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Reshape, Lambda

from .base_autoencoder import BaseAutoencoder


class KerasBaseAutoencoder(BaseAutoencoder):
    def __init__(self):
        self.autoencoder = None
        self.encoder = None
        self.decoder = None

    def __getattr__(self, name):
        return getattr(self.autoencoder, name)

    def fit(self, x, *args, **kwargs):
        if "validation_data" in kwargs:
            kwargs["validation_data"] = (kwargs["validation_data"], kwargs["validation_data"])
        return self.autoencoder.fit(x, x, *args, **kwargs)

    def predict(self, x, *args, **kwargs):
        return self.autoencoder.predict(x, *args, **kwargs)

    def encode(self, x, *args, **kwargs):
        return self.encoder.predict(x, *args, **kwargs)

    def decode(self, x, *args, **kwargs):
        return self.decoder.predict(x, *args, **kwargs)


class KerasAutoencoder(KerasBaseAutoencoder):
    def __init__(self, input_shape, dims, hidden_activation, output_activation):
        ndim = len(input_shape)

        encoder_input = Input(shape=input_shape)
        if ndim >= 2:
            encoded = Flatten()(encoder_input)
        else:
            encoder_input
        for dim in dims:
            encoded = Dense(dim, activation=hidden_activation)(encoded)

        decoder_input = Input(shape=(dims[-1],))
        decoder = decoder_input
        decoded = encoded
        decoder_dims = dims[-2::-1] + [np.prod(input_shape)]
        decoder_activations = [hidden_activation] * (len(decoder_dims) - 1) + [output_activation]
        for dim, activation in zip(decoder_dims, decoder_activations):
            layer = Dense(dim, activation=activation)
            decoded = layer(decoded)
            decoder = layer(decoder)
        if ndim >= 2:
            decoded = Reshape(input_shape)(decoded)
            decoder = Reshape(input_shape)(decoder)

        self.autoencoder = Model(input=encoder_input, output=decoded)
        self.encoder = Model(input=encoder_input, output=encoded)
        self.decoder = Model(input=decoder_input, output=decoder)


class KerasVariationalAutoencoder(KerasBaseAutoencoder):
    def __init__(self, input_shape, dims, hidden_activation):
        ndim = len(input_shape)
        encoder_input = Input(shape=input_shape)
        batch_size = tf.shape(encoder_input)[0]
        if ndim >= 2:
            encoded = Flatten()(encoder_input)
        else:
            encoded = encoder_input
        for dim in dims:
            encoded = Dense(dim, activation=hidden_activation)(encoded)
        z_mean = Dense(dims[-1])(encoded)
        z_log_std = Dense(dims[-1])(encoded)

        def sample(args):
            z_mean, z_log_std = args
            z = K.random_normal(shape=(batch_size, dims[-1]), mean=0, std=1)
            return z_mean + z * K.exp(z_log_std)
        z = Lambda(sample)([z_mean, z_log_std])

        decoder_input = Input(shape=(dims[-1],))
        decoder = decoder_input
        decoded = z
        decoder_dims = dims[-2::-1] + [np.prod(input_shape)]
        decoder_activations = [hidden_activation] * (len(decoder_dims) - 1) + ["sigmoid"]
        for dim, activation in zip(decoder_dims, decoder_activations):
            layer = Dense(dim, activation=activation)
            decoded = layer(decoded)
            decoder = layer(decoder)
        if ndim >= 2:
            decoded = Reshape(input_shape)(decoded)
            decoder = Reshape(input_shape)(decoder)

        def loss(true, pred):
            reconstruction_loss = np.prod(input_shape) * metrics.binary_crossentropy(true, pred)
            kl_loss = -K.sum(1 + 2 * z_log_std - K.square(z_mean) - K.exp(2 * z_log_std), axis=-1) / 2
            return K.mean(reconstruction_loss + kl_loss)

        self.loss = loss
        self.autoencoder = Model(input=encoder_input, output=decoded)
        self.encoder = Model(input=encoder_input, output=z_mean)
        self.decoder = Model(input=decoder_input, output=decoder)

    def compile(self, *args, **kwargs):
        self.autoencoder.compile(loss=self.loss, *args, **kwargs)
