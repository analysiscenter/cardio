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

    @staticmethod
    def _build_encoder(encoded, dims, activation):
        for dim in dims:
            encoded = Dense(dim, activation=activation)(encoded)
        return encoded

    @staticmethod
    def _build_decoder(decoded, decoder, dims, hidden_activation, output_activation):
        activations = [hidden_activation] * (len(dims) - 1) + [output_activation]
        for dim, activation in zip(dims, activations):
            layer = Dense(dim, activation=activation)
            decoded = layer(decoded)
            decoder = layer(decoder)
        return decoded, decoder

    @staticmethod
    def _flatten(tensor):
        if len(tensor.shape) > 2:
            return Flatten()(tensor)
        return tensor

    @staticmethod
    def _reshape(tensor, shape):
        if len(shape) > 2:
            return Reshape(shape)(tensor)
        return tensor

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
        super().__init__()

        # Encoder part
        encoder_input = Input(shape=input_shape)
        encoded = self._flatten(encoder_input)
        encoded = self._build_encoder(encoded, dims, hidden_activation)

        # Decoder part
        decoder_dims = dims[-2::-1] + [np.prod(input_shape)]
        decoder_input = Input(shape=(dims[-1],))
        decoded, decoder = self._build_decoder(encoded, decoder_input, decoder_dims,
                                               hidden_activation, output_activation)
        decoded = self._reshape(decoded, input_shape)
        decoder = self._reshape(decoder, input_shape)

        self.autoencoder = Model(input=encoder_input, output=decoded)
        self.encoder = Model(input=encoder_input, output=encoded)
        self.decoder = Model(input=decoder_input, output=decoder)


class KerasVariationalAutoencoder(KerasBaseAutoencoder):
    def __init__(self, input_shape, dims, activation):
        super().__init__()

        # Encoder part
        encoder_input = Input(shape=input_shape)
        batch_size = tf.shape(encoder_input)[0]
        encoded = self._flatten(encoder_input)
        encoded = self._build_encoder(encoded, dims, activation)
        z_mean = Dense(dims[-1])(encoded)
        z_log_std = Dense(dims[-1])(encoded)

        # Sampling part
        def sample_normal(args):
            mean, log_std = args
            return K.random_normal(shape=(batch_size, dims[-1]), mean=mean, std=K.exp(log_std))
        z = Lambda(sample_normal)([z_mean, z_log_std])

        # Decoder part
        decoder_dims = dims[-2::-1] + [np.prod(input_shape)]
        decoder_input = Input(shape=(dims[-1],))
        decoded, decoder = self._build_decoder(z, decoder_input, decoder_dims,
                                               activation, "sigmoid")
        decoded = self._reshape(decoded, input_shape)
        decoder = self._reshape(decoder, input_shape)

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
