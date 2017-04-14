from .keras_autoencoder import KerasAutoencoder, KerasVariationalAutoencoder


class AutoencoderFactory:
    def __new__(cls, *args, backend="keras", **kwargs):
        if not isinstance(backend, str):
            raise TypeError("backend name must be a string")
        backend_cls = cls.backend_dict.get(backend)
        if backend_cls is None:
            raise KeyError("{} backend is not supported yet".format(backend))
        return backend_cls(*args, **kwargs)


class Autoencoder(AutoencoderFactory):
    backend_dict = {
        "keras": KerasAutoencoder,
    }


class VariationalAutoencoder(AutoencoderFactory):
    backend_dict = {
        "keras": KerasVariationalAutoencoder,
    }
