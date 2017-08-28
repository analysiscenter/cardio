"""Contains base model class."""

class BaseModel:
    """Base model class."""

    def load(self, *args, **kwargs):
        """Load model."""
        raise NotImplementedError("load method must be defined in a subclass")

    def save(self, *args, **kwargs):
        """Save model."""
        raise NotImplementedError("save method must be defined in a subclass")

    def train_on_batch(self, batch, *args, **kwargs):
        """Run a single gradient update on a single batch."""
        raise NotImplementedError("train_on_batch method must be defined in a subclass")

    def test_on_batch(self, batch, *args, **kwargs):
        """Get model loss for a single batch."""
        raise NotImplementedError("test_on_batch method must be defined in a subclass")

    def predict_on_batch(self, batch, *args, **kwargs):
        """Get model predictions for a single batch."""
        raise NotImplementedError("predict_on_batch method must be defined in a subclass")
