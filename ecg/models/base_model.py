"""Contains base model class."""

class BaseModel:
    """Base model class."""

    def load(self, *args, **kwargs):
        """Load model."""
        raise NotImplementedError

    def save(self, *args, **kwargs):
        """Save model."""
        raise NotImplementedError

    def train_on_batch(self, batch, *args, **kwargs):
        """Run a single gradient update on a single batch."""
        raise NotImplementedError

    def test_on_batch(self, batch, *args, **kwargs):
        """Test the model on a single batch."""
        raise NotImplementedError

    def predict_on_batch(self, batch, *args, **kwargs):
        """Get model predictions for a single batch."""
        raise NotImplementedError
