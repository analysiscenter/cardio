"""Contains BaseAutoencoder interface."""


class BaseAutoencoder:
    """Base autoencoder class."""

    def fit(self, x, *args, **kwargs):
        """Fit autoencoder model to given data."""
        raise NotImplementedError("fit method must be defined in a subclass")

    def predict(self, x, *args, **kwargs):
        """Generate reconstructions for input samples."""
        raise NotImplementedError("predict method must be defined in a subclass")

    def encode(self, x, *args, **kwargs):
        """Generate hidden representations for input samples."""
        raise NotImplementedError("encode method must be defined in a subclass")

    def decode(self, x, *args, **kwargs):
        """Generate reconstructions for hidden representations."""
        raise NotImplementedError("decode method must be defined in a subclass")
