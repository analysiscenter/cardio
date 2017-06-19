""" DatasetIndex """

import os
import glob
from collections import Iterable
import numpy as np

from .base import Baseset


class DatasetIndex(Baseset):
    """ Stores an index for a dataset
    The index should be 1-d array-like, e.g. numpy array, pandas Series, etc.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pos = self.build_pos()
        self._random_state = None

    @classmethod
    def from_index(cls, *args, **kwargs):
        """Create index from another index """
        return cls(*args, **kwargs)

    @staticmethod
    def build_index(index):
        """ Check index type and structure """
        if callable(index):
            _index = index()
        else:
            _index = index

        if isinstance(_index, DatasetIndex):
            _index = _index.indices
        else:
            # index should allow for advance indexing (i.e. subsetting)
            _index = np.asarray(_index)

        if len(_index) == 0:
            raise ValueError("Index cannot be empty")

        if len(_index.shape) > 1:
            raise TypeError("Index should be 1-dimensional")

        return _index

    def build_pos(self):
        """ Create a dictionary with positions in the index """
        if self.indices is None:
            return dict()
        else:
            return dict(zip(self.indices, np.arange(len(self))))

    def get_pos(self, index):
        """ Return position of an item in the index """
        if isinstance(index, slice):
            start = self._pos[index.start] if index.start is not None else None
            stop = self._pos[index.stop] if index.stop is not None else None
            return slice(start, stop, index.step)
        elif isinstance(index, Iterable):
            return np.asarray([self._pos[ix] for ix in index])
        else:
            return self._pos[index]

    def subset_by_pos(self, pos):
        """ Return subset of index by given positions in the index """
        return self.index[pos]

    def create_subset(self, index):
        """ Return a new index object based on the subset of indices given """
        return type(self)(index)

    def cv_split(self, shares=0.8, shuffle=False):
        """ Split index into train, test and validation subsets
        Shuffles index if necessary.
        Subsets are available as .train, .test and .validation respectively

        Usage:
           # split into train / test in 80/20 ratio
           di.cv_split()
           # split into train / test / validation in 60/30/10 ratio
           di.cv_split([0.6, 0.3])
           # split into train / test / validation in 50/30/20 ratio
           di.cv_split([0.5, 0.3, 0.2])
        """
        _, test_share, valid_share = self.calc_cv_split(shares)

        # TODO: make a view not copy if not shuffled
        if shuffle:
            order = self._shuffle(shuffle)
        else:
            order = np.arange(len(self))

        if valid_share > 0:
            validation_pos = order[:valid_share]
            self.validation = self.create_subset(self.subset_by_pos(validation_pos))
        if test_share > 0:
            test_pos = order[valid_share : valid_share + test_share]
            self.test = self.create_subset(self.subset_by_pos(test_pos))
        train_pos = order[valid_share + test_share:]
        self.train = self.create_subset(self.subset_by_pos(train_pos))


    def _shuffle(self, shuffle, order=None):
        if order is None:
            if self._order is None:
                order = np.arange(len(self))
            else:
                order = self._order

        if isinstance(shuffle, bool):
            if shuffle:
                order = np.random.permutation(order)
        elif isinstance(shuffle, int):
            if self._random_state is None or self._random_state.seed != shuffle:
                self._random_state = np.random.RandomState(shuffle)
            order = self._random_state.permutation(order)
        elif isinstance(shuffle, np.random.RandomState):
            if self._random_state != shuffle:
                self._random_state = shuffle
            order = self._random_state.permutation(order)
        elif callable(shuffle):
            order = shuffle(self.indices)
        else:
            raise ValueError("shuffle could be bool, int, numpy.random.RandomState or callable")
        return order


    def next_batch(self, batch_size, shuffle=False, n_epochs=1, drop_last=False):
        """ Return next batch """

        if self._order is None:
            self._order = self._shuffle(shuffle)
        num_items = len(self._order)

        rest_items = None
        if self._start_index + batch_size >= num_items:
            rest_items = np.copy(self._order[self._start_index:])
            rest_of_batch = self._start_index + batch_size - num_items
            self._start_index = 0
            self._n_epochs += 1
            self._order = self._shuffle(shuffle, self._order)
        else:
            rest_of_batch = batch_size

        new_items = self._order[self._start_index : self._start_index + rest_of_batch]
        # TODO: concat not only numpy arrays
        if rest_items is None:
            batch_items = new_items
        else:
            batch_items = np.concatenate((rest_items, new_items))

        if n_epochs is not None and self._n_epochs >= n_epochs and rest_items is not None:
            if drop_last and len(rest_items) < batch_size:
                raise StopIteration("Dataset is over. No more batches left.")
            else:
                return self.create_batch(rest_items, pos=True)
        else:
            self._start_index += rest_of_batch
            return self.create_batch(batch_items, pos=True)


    def gen_batch(self, batch_size, shuffle=False, n_epochs=1, drop_last=False):
        """ Generate batches """
        self._start_index = 0
        self._order = None
        self._n_epochs = 0
        while True:
            if n_epochs is not None and self._n_epochs >= n_epochs:
                raise StopIteration()
            else:
                batch = self.next_batch(batch_size, shuffle, n_epochs, drop_last)
                yield batch


    def create_batch(self, batch_indices, pos=True, as_array=False, *args, **kwargs):   # pylint: disable=arguments-differ
        """ Create a batch from given indices
        if pos is False then batch_indices contains the value of indices
        which should be included in the batch (so expected batch is just the very same batch_indices)
        otherwise batch_indices contains positions in the index
        """
        _ = args, kwargs
        if isinstance(batch_indices, DatasetIndex):
            _batch_indices = batch_indices.indices
        else:
            _batch_indices = batch_indices
        if pos:
            batch = self.subset_by_pos(_batch_indices)
        else:
            batch = _batch_indices
        if not as_array:
            batch = self.create_subset(batch)
        return batch


class FilesIndex(DatasetIndex):
    """ Index with the list of files or directories with the given path pattern

        Usage:
        Create sorted index of files in a directory:
        fi = FilesIndex('/path/to/data/files/*', sort=True)
        Create unsorted index of directories through all subdirectories:
        fi = FilesIndex('/path/to/data/archive*/patient*', dirs=True)
    """
    def __init__(self, *args, **kwargs):
        self._paths = None
        super().__init__(*args, **kwargs)

    def build_index(self, index=None, path=None, *args, **kwargs):     # pylint: disable=arguments-differ
        """ Build index from a path string or an index given """
        if path is None:
            return self.build_from_index(index, *args, **kwargs)
        else:
            return self.build_from_path(path, *args, **kwargs)

    def build_from_index(self, index, paths):
        """ Build index from another index for indices given """
        if isinstance(paths, dict):
            self._paths = dict((file, paths[file]) for file in index)
        else:
            self._paths = dict((file, paths[pos]) for pos, file in np.ndenumerate(index))
        return index

    def build_from_path(self, path, dirs=False, no_ext=False, sort=False):
        """ Build index from a path/glob or a sequence of paths/globs """
        if isinstance(path, str):
            paths = [path]
        else:
            paths = path

        _all_index = None
        _all_paths = dict()
        for one_path in paths:
            _index, _paths = self.build_from_one_path(one_path, dirs, no_ext)
            if _all_index is None:
                _all_index = _index
            else:
                _all_index = np.concatenate((_all_index, _index))
            _all_paths.update(_paths)

        if sort:
            _all_index.sort()
        self._paths = _all_paths

        return _all_index

    def build_from_one_path(self, path, dirs=False, no_ext=False):
        """ Build index from a path/glob """
        check_fn = os.path.isdir if dirs else os.path.isfile
        pathlist = glob.iglob(path)
        _full_index = np.asarray([self.build_key(fname, no_ext) for fname in pathlist if check_fn(fname)])
        if _full_index.shape[0] > 0:
            _index = _full_index[:, 0]
            _paths = _full_index[:, 1]
        else:
            _index, _paths = [], []
        _paths = dict(zip(_index, _paths))
        return _index, _paths

    @staticmethod
    def build_key(fullpathname, no_ext=False):
        """ Create index item from full path name """
        if no_ext:
            key_name = '.'.join(os.path.basename(fullpathname).split('.')[:-1])
        else:
            key_name = os.path.basename(fullpathname)
        return key_name, fullpathname

    def get_fullpath(self, key):
        """ Return the full path name for an item in the index """
        return self._paths[key]

    def create_subset(self, index):
        """ Return a new FilesIndex based on the subset of indices given """
        return type(self).from_index(index=index, paths=self._paths)
