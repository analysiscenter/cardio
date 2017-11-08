"""Contains BaseAutoencoder interface."""


class BaseAutoencoder:
    """Base autoencoder class."""

    def fit(self, x, *args, **kwargs):
        """Fit autoencoder model to given data."""
        raise NotImplementedError("Fit method must be defined in a subclass")

    def predict(self, x, *args, **kwargs):
        """Generate reconstructions for input samples."""
        raise NotImplementedError("Predict method must be defined in a subclass")

    def encode(self, x, *args, **kwargs):
        """Generate hidden representations for input samples."""
        raise NotImplementedError("Encode method must be defined in a subclass")

    def decode(self, x, *args, **kwargs):
        """Generate reconstructions for hidden representations."""
        raise NotImplementedError("Decode method must be defined in a subclass")
