"""Contains keras-based autoencoder classes."""

import numpy as np

from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Reshape, Lambda

from .base_autoencoder import BaseAutoencoder


class KerasBaseAutoencoder(BaseAutoencoder):
    """Base keras autoencoder class.

    All calls to unknown methods and attributes are redirected to the
    autoencoder model.

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
    def _get_block(dim, activation):
        """Return Layer instance. Only Dense is supported yet."""
        return Dense(dim, activation=activation)

    @classmethod
    def _build_encoder(cls, encoder_input, dims, activation):
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
            encoded = cls._get_block(dim, activation=activation)(encoded)
        return encoded

    @classmethod
    def _build_decoder(cls, encoded, decoder_input, dims, hidden_activation, output_activation):
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
            layer = cls._get_block(dim, activation=activation)
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

    def fit(self, x, batch_size=32, epochs=10, verbose=1, callbacks=None,  # pylint: disable=too-many-arguments
            validation_split=0.0, validation_data=None, *args, **kwargs):
        """Fit autoencoder model to given data.

        Parameters
        ----------
        x : NumPy array
            Input data to reconstruct.
        *args, **kwargs :
            Any keras predict arguments. Validation_data argument expects
            NumPy array, not an (input, output) tuple.
        """
        if validation_data is not None:
            validation_data = (validation_data, validation_data)
        return self.autoencoder.fit(x, x, batch_size, epochs, verbose, callbacks,
                                    validation_split, validation_data, *args, **kwargs)

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
        self._build_autoencoder(input_shape, dims, hidden_activation, output_activation)

    def _build_autoencoder(self, input_shape, dims, hidden_activation, output_activation):
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

        self.autoencoder = Model(inputs=encoder_input, outputs=decoded)
        self.encoder = Model(inputs=encoder_input, outputs=encoded)
        self.decoder = Model(inputs=decoder_input, outputs=decoder)


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
        self._build_autoencoder(input_shape, dims, activation)

    def _build_autoencoder(self, input_shape, dims, activation):
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
        # Encoder part
        encoder_input = Input(shape=input_shape)
        encoded = self._flatten(encoder_input)
        encoded = self._build_encoder(encoded, dims, activation)
        z_mean = Dense(dims[-1])(encoded)
        z_std = Dense(dims[-1], activation="softplus")(encoded)
        self.kl_loss = -K.sum(1 + 2 * K.log(z_std) - K.square(z_mean) - K.square(z_std), axis=-1) / 2

        # Sampling from approximate posterior using the reparametrization trick
        z = Lambda(lambda x: self._sample_normal(x[0], x[1]))([z_mean, z_std])

        # Decoder part
        decoder_dims = dims[-2::-1] + [np.prod(input_shape)]
        decoder_input = Input(shape=(dims[-1],))
        decoded, decoder = self._build_decoder(z, decoder_input, decoder_dims,
                                               activation, "sigmoid")
        decoded = self._reshape(decoded, input_shape)
        decoder = self._reshape(decoder, input_shape)

        self.autoencoder = Model(inputs=encoder_input, outputs=decoded)
        self.encoder = Model(inputs=encoder_input, outputs=[z_mean, z_std])
        self.decoder = Model(inputs=decoder_input, outputs=decoder)

    @staticmethod
    def _sample_normal(mean=0, std=1, shape=None):
        """Return independent samples from normal distribution.

        Parameters
        ----------
        mean : float or Tensor of floats
            Mean of the distribution. Defaults to 0.
        std : float or Tensor of floats
            Standard deviation of the distribution. Defaults to 1.
        shape : tuple
            Output tensor shape. If shape is None, it defaults to mean and std
            broadcast shape.

        Returns
        -------
        z : Tensor
            A tensor of given shape filled with random normal values.
        """
        if shape is None:
            shape = np.broadcast(np.atleast_1d(mean), np.atleast_1d(std)).shape
        eps = K.random_normal(shape=shape, mean=0, stddev=1)
        return mean + std * eps

    def _loss(self, true, pred):
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
        reconstruction_loss = K.sum(K.binary_crossentropy(pred, true), axis=list(range(1, K.ndim(true))))
        return K.mean(reconstruction_loss + self.kl_loss)

    def compile(self, *args, **kwargs):
        """Compile autoencoder model.

        Parameters
        ----------
        *args, **kwargs :
            Any keras compile arguments except loss.
        """
        self.autoencoder.compile(loss=self._loss, *args, **kwargs)
