class BaseAutoencoder:
    def fit(self, x, *args, **kwargs):
        raise NotImplementedError("fit method must be defined in a subclass")

    def predict(self, x, *args, **kwargs):
        raise NotImplementedError("predict method must be defined in a subclass")

    def encode(self, x, *args, **kwargs):
        raise NotImplementedError("encode method must be defined in a subclass")

    def decode(self, x, *args, **kwargs):
        raise NotImplementedError("decode method must be defined in a subclass")
