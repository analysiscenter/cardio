"""Contains factory classes for autoencoder creation."""

from .keras_autoencoder import KerasAutoencoder, KerasVariationalAutoencoder


class AutoencoderFactory:  # pylint: disable=too-few-public-methods
    """Base autoencoder factory.

    This class can not be instantiated. Instantiation of its subclass leads
    to a backend-specified autoencoder instance creation, depending on the
    passed backend parameter.
    """

    _backend_dict = None

    def __new__(cls, *args, backend="keras", **kwargs):
        if cls is AutoencoderFactory:
            raise TypeError("AutoencoderFactory class can not be instantiated")
        if cls._backend_dict is None:
            raise ValueError("Backend dictionary must be defined")
        if not isinstance(backend, str):
            raise TypeError("Backend name must be a string")
        backend_cls = cls._backend_dict.get(backend)
        if backend_cls is None:
            raise KeyError("{} backend is not supported yet".format(backend))
        return backend_cls(*args, **kwargs)


class Autoencoder(AutoencoderFactory):  # pylint: disable=too-few-public-methods
    """Autoencoder factory.

    Instantiation of Autoencoder class leads to a backend-specified
    autoencoder instance creation, depending on the passed backend parameter.
    """

    _backend_dict = {
        "keras": KerasAutoencoder,
    }


class VariationalAutoencoder(AutoencoderFactory):  # pylint: disable=too-few-public-methods
    """Variational autoencoder factory.

    Instantiation of VariationalAutoencoder class leads to a backend-specified
    variational autoencoder instance creation, depending on the passed backend
    parameter.
    """

    _backend_dict = {
        "keras": KerasVariationalAutoencoder,
    }
