""" Contains the base batch class """
from binascii import hexlify
import numpy as np
from .decorators import action


class BaseBatch:
    """ Base class for batches
    Required to solve circular module dependencies
    """
    def __init__(self, index):
        self.index = index
        self._data = None

    @staticmethod
    def make_filename():
        """ Generate unique filename for the batch """
        random_data = np.random.uniform(0, 1, size=10) * 123456789
        # probability of collision is around 2e-10.
        filename = hexlify(random_data.data)[:8]
        return filename.decode("utf-8")

    @action
    def load(self, src, fmt=None):
        """ Load data from a file or another data source """
        raise NotImplementedError()

    @action
    def dump(self, dst, fmt=None):
        """ Save batch data to disk """
        raise NotImplementedError()

    @action
    def save(self, *args, **kwargs):
        """ Save batch data to a file (an alias for dump method)"""
        return self.dump(*args, **kwargs)
