""" Base class """
import numpy as np


class Baseset:
    """ Base class """
    def __init__(self, *args, **kwargs):
        self._index = self.build_index(*args, **kwargs)

        self.train = None
        self.test = None
        self.validation = None

        self._start_index = 0
        self._order = None
        self._n_epochs = 0
        self._batch_generator = None


    @staticmethod
    def build_index(index, *args, **kwargs):
        """ Create the index. Child classes should generate index from the arguments given """
        _ = args, kwargs
        return index

    @property
    def index(self):
        """ Return the index """
        return self._index

    @property
    def indices(self):
        """ Return an array-like with the indices """
        if isinstance(self.index, Baseset):
            return self.index.indices
        else:
            return self.index

    def __len__(self):
        if self.indices is None:
            return 0
        else:
            return len(self.indices)

    @property
    def is_splitted(self):
        """ True if dataset was splitted into train / test / validation sub-datasets """
        return self.train is not None

    def calc_cv_split(self, shares=0.8):
        """ Calculate split into train, test and validation subsets

        Return: a tuple which contains number of items in train, test and validation subsets

        Usage:
           # split into train / test in 80/20 ratio
           bs.calc_cv_split()
           # split into train / test / validation in 60/30/10 ratio
           bs.calc_cv_split([0.6, 0.3])
           # split into train / test / validation in 50/30/20 ratio
           bs.calc_cv_split([0.5, 0.3, 0.2])
        """
        _shares = np.array(shares).ravel() # pylint: disable=no-member

        if _shares.shape[0] > 3:
            raise ValueError("Shares must have no more than 3 elements")
        if _shares.sum() > 1:
            raise ValueError("Shares must sum to 1")

        if _shares.shape[0] == 3:
            if not np.allclose(1. - _shares.sum(), 0.):
                raise ValueError("Shares must sum to 1")
            train_share, test_share, valid_share = _shares
        elif _shares.shape[0] == 2:
            train_share, test_share, valid_share = _shares[0], _shares[1], 1 - _shares.sum()
        else:
            train_share, test_share, valid_share = _shares[0], 1 - _shares[0], 0.

        n_items = len(self)
        train_share, test_share, valid_share = \
            np.round(np.array([train_share, test_share, valid_share]) * n_items).astype('int')
        train_share = n_items - test_share - valid_share

        return train_share, test_share, valid_share


    def create_subset(self, index):
        """ Create a new subset based on the given index subset """
        raise NotImplementedError("create_subset should be defined in child classes")


    def cv_split(self, shares=0.8, shuffle=False):
        """ Split the dataset into train, test and validation sub-datasets
        Subsets are available as .train, .test and .validation respectively

        Usage:
           # split into train / test in 80/20 ratio
           ds.cv_split()
           # split into train / test / validation in 60/30/10 ratio
           ds.cv_split([0.6, 0.3])
           # split into train / test / validation in 50/30/20 ratio
           ds.cv_split([0.5, 0.3, 0.2])
        """
        self.index.cv_split(shares, shuffle)

        self.train = self.create_subset(self.index.train)
        if self.index.test is not None:
            self.test = self.create_subset(self.index.test)
        if self.index.validation is not None:
            self.validation = self.create_subset(self.index.validation)

    def reset_iter(self):
        """ Clear all iteration metadata in order to start iterating from scratch """
        self._start_index = 0
        self._order = None
        self._n_epochs = 0
        if hasattr(self.index, 'reset_iter'):
            self.index.reset_iter()

    def gen_batch(self, batch_size, shuffle=False, n_epochs=1, drop_last=False, *args, **kwargs):
        """ Generate batches """
        for ix_batch in self.index.gen_batch(batch_size, shuffle, n_epochs, drop_last):
            batch = self.create_batch(ix_batch, *args, **kwargs)
            yield batch

    def next_batch(self, batch_size, shuffle=False, n_epochs=1, drop_last=False, *args, **kwargs):
        """ Return a batch """
        batch_index = self.index.next_batch(batch_size, shuffle, n_epochs, drop_last, *args, **kwargs)
        batch = self.create_batch(batch_index, *args, **kwargs)
        return batch

    def create_batch(self, batch_indices, pos=True):
        """ Create batch with indices given """
        raise NotImplementedError("create_batch should be implemented in child classes")
