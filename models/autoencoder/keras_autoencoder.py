"""Contains keras-based autoencoder classes."""


import numpy as np

import tensorflow as tf
from keras import backend as K
from keras import metrics
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Reshape, Lambda

from .base_autoencoder import BaseAutoencoder


class KerasBaseAutoencoder(BaseAutoencoder):
    """Base keras autoencoder class.

    Attributes
    ----------
    autoencoder : Model
        A neural network, that reconstructs its own inputs.
    encoder : Model
        Encoding part, that maps input samples into hidden representations.
    decoder : Model
        Decoding part, that maps hidden representations into reconstructions.
    """

    def __init__(self):
        """Initialize autoencoder, encoder and decoder models."""
        self.autoencoder = None
        self.encoder = None
        self.decoder = None

    def __getattr__(self, name):
        """Check if an unknown attribute is an autoencoder attribute."""
        return getattr(self.autoencoder, name)

    @staticmethod
    def _build_encoder(encoder_input, dims, activation):
        """Build an encoder part.

        Parameters
        ----------
        encoder_input : Tensor
            Input data to encode.
        dims : list
            Hidden layers' dimensions.
        activation : str or tensorflow element-wise function
            Hidden layers' activation function.

        Returns
        -------
        encoded : Tensor
            Hidden representations for input data.
        """

        encoded = encoder_input
        for dim in dims:
            encoded = Dense(dim, activation=activation)(encoded)
        return encoded

    @staticmethod
    def _build_decoder(encoded, decoder_input, dims, hidden_activation, output_activation):
        """Build a decoder part.

        Parameters
        ----------
        encoded : Tensor
            Encoded data to decode.
        decoder_input : Tensor
            Extra input for decoder model.
        dims : list
            Hidden layers' dimensions.
        hidden_activation : str or tensorflow element-wise function
            Hidden layers' activation function.
        output_activation : str or tensorflow element-wise function
            Output layer activation function.

        Returns
        -------
        decoded : Tensor
            Encoded data reconstruction.
        decoded : Tensor
            Extra output for decoder model.
        """

        decoded = encoded
        decoder = decoder_input
        activations = [hidden_activation] * (len(dims) - 1) + [output_activation]
        for dim, activation in zip(dims, activations):
            layer = Dense(dim, activation=activation)
            decoded = layer(decoded)
            decoder = layer(decoder)
        return decoded, decoder

    @staticmethod
    def _flatten(tensor):
        """Flatten the input tensor. Does not affect the batch size."""
        if len(tensor.shape) > 2:
            return Flatten()(tensor)
        return tensor

    @staticmethod
    def _reshape(tensor, shape):
        """Reshape the input tensor. Does not affect the batch size."""
        if len(shape) >= 2:
            return Reshape(shape)(tensor)
        return tensor

    def fit(self, x, *args, **kwargs):
        """Fit autoencoder model to given data.

        Parameters
        ----------
        x : NumPy array
            Input data to reconstruct.
        *args, **kwargs :
            Any keras predict arguments. Validation_data argument expects
            NumPy array, not an (input, output) tuple.
        """

        if "validation_data" in kwargs:
            kwargs["validation_data"] = (kwargs["validation_data"], kwargs["validation_data"])
        return self.autoencoder.fit(x, x, *args, **kwargs)

    def predict(self, x, *args, **kwargs):
        """Generate reconstructions for input samples.

        Parameters
        ----------
        x : NumPy array
            Input data to reconstruct.
        *args, **kwargs :
            Any keras predict arguments.
        """

        return self.autoencoder.predict(x, *args, **kwargs)

    def encode(self, x, *args, **kwargs):
        """Generate hidden representations for input samples.

        Parameters
        ----------
        x : NumPy array
            Input data to encode.
        *args, **kwargs :
            Any keras predict arguments.
        """

        return self.encoder.predict(x, *args, **kwargs)

    def decode(self, x, *args, **kwargs):
        """Generate reconstructions for hidden representations.

        Parameters
        ----------
        x : NumPy array
            Input data to decode.
        *args, **kwargs :
            Any keras predict arguments.
        """

        return self.decoder.predict(x, *args, **kwargs)


class KerasAutoencoder(KerasBaseAutoencoder):
    """Keras autoencoder class."""

    def __init__(self, input_shape, dims, hidden_activation, output_activation):
        """Initialize autoencoder, encoder and decoder models.

        Parameters
        ----------
        input_shape : tuple
            Input tensor shape without the batch dimension.
        dims : list
            Encoder hidden layers' dimensions.
        hidden_activation : str or tensorflow element-wise function
            Hidden layers' activation function.
        output_activation : str or tensorflow element-wise function
            Output layer activation function.
        """

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
    """Keras variational autoencoder class with Bernoulli decoder distribution."""

    def __init__(self, input_shape, dims, activation):
        """Initialize autoencoder, encoder and decoder models.

        Parameters
        ----------
        input_shape : tuple
            Input tensor shape without the batch dimension.
        dims : list
            Encoder hidden layers' dimensions.
        activation : str or tensorflow element-wise function
            Hidden layers' activation function.
        """

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
            """Return samples from normal distribution.

            Parameters
            ----------
            args : (mean, log_std)
                Normal distribution mean and log standard deviation.

            Returns
            -------
            z : Tensor
                A tensor of shape (batch_size, decoder_input_size) filled with
                random normal values.
            """

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

        # Loss specification
        def loss(true, pred):
            """Calculate batch loss.

            Consists of two parts:
            1. Reconstruction loss - batch log likelihood, specified by
               decoder distribution,
            2. Kullback–Leibler divergence between the approximate posterior
               and the prior.

            Parameters
            ----------
            true : Tensor
                True target.
            pred : Tensor
                Model predictions.

            Returns
            -------
            loss : float
                Batch loss.
            """

            reconstruction_loss = np.prod(input_shape) * metrics.binary_crossentropy(true, pred)
            kl_loss = -K.sum(1 + 2 * z_log_std - K.square(z_mean) - K.exp(2 * z_log_std), axis=-1) / 2
            return K.mean(reconstruction_loss + kl_loss)
        self.loss = loss

        self.autoencoder = Model(input=encoder_input, output=decoded)
        self.encoder = Model(input=encoder_input, output=z_mean)
        self.decoder = Model(input=decoder_input, output=decoder)

    def compile(self, *args, **kwargs):
        """Compile autoencoder model.

        Parameters
        ----------
        *args, **kwargs :
            Any keras compile arguments, except loss.
        """

        self.autoencoder.compile(loss=self.loss, *args, **kwargs)
