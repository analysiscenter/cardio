""" Contains basic Batch classes """

import os

try:
    import dill
    import blosc
except ImportError:
    pass
import numpy as np
try:
    import pandas as pd
except ImportError:
    pass
try:
    import feather
except ImportError:
    pass
try:
    import dask.dataframe as dd
except ImportError:
    pass

from .dsindex import DatasetIndex, FilesIndex
from .decorators import action, inbatch_parallel, ModelDecorator, any_action_failed
from .dataset import Dataset
from .batch_base import BaseBatch
from .components import MetaComponentsTuple


class Batch(BaseBatch):
    """ The core Batch class """
    _item_class = None

    def __init__(self, index, preloaded=None, *args, **kwargs):
        if  self.components is not None and not isinstance(self.components, tuple):
            raise TypeError("components should be a tuple of strings with components names")
        super().__init__(index, *args, **kwargs)
        self._preloaded = preloaded

    @classmethod
    def from_data(cls, index, data):
        """ Create batch from a given dataset """
        # this is roughly equivalent to self.data = data
        if index is None:
            index = np.arange(len(data))
        return cls(index, preloaded=data)

    @classmethod
    def from_batch(cls, batch):
        """ Create batch from another batch """
        return cls(batch.index, preloaded=batch._data)  # pylint: disable=protected-access

    @classmethod
    def merge(cls, batches, batch_size):
        """ Merge several batches to form a new batch of a given size """
        raise NotImplementedError("merge method should be implemented in children batch classes")

    def as_dataset(self, dataset=None):
        """ Makes a new dataset from batch data
        Args:
            dataset: could be a dataset or a Dataset class
        Output:
            an instance of a class specified by `dataset` arg, preloaded with this batch data
        """
        if dataset is None:
            dataset_class = Dataset
        elif isinstance(dataset, Dataset):
            dataset_class = dataset.__class__
        elif isinstance(dataset, type):
            dataset_class = dataset
        else:
            raise TypeError("dataset should be some Dataset class or an instance of some Dataset class or None")
        return dataset_class(self.index, preloaded=self.data)

    @property
    def indices(self):
        """ Return an array-like with the indices """
        if isinstance(self.index, DatasetIndex):
            return self.index.indices
        else:
            return self.index

    def __len__(self):
        return len(self.index)

    @property
    def data(self):
        """ Return batch data """
        if self._data is None and self._preloaded is not None:
            # load data the first time it's requested
            self.load(self._preloaded)
        res = self._data if self.components is None else self._data_named
        return res if res is not None else self._empty_data

    @property
    def components(self):
        """ Return data components names """
        return None

    def make_item_class(self):
        """ Create a class to handle data components """
        # pylint: disable=protected-access
        if self.components is None:
            type(self)._item_class = None
        elif type(self)._item_class is None:
            comp_class = MetaComponentsTuple(type(self).__name__ + 'Components', components=self.components)
            type(self)._item_class = comp_class

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('_data_named')
        return state

    @property
    def _empty_data(self):
        return None if self.components is None else self._item_class()   # pylint: disable=not-callable

    def get_pos(self, data, component, index):
        """ Return a position in data for a given index

        Parameters:
            data: if None, get_pos should return a position in self.data
            components: could be one of [None, int or string]
                None: data has no components (e.g. just an array or pandas.DataFrame)
                int: a position of a data component, when components names are not defined
                str: a name of a data component
            index: an index
        Returns:
            int - a position in a batch data where an item with a given index is stored
        It is used to read / write data in a given component:
            batch_data = data.component[pos]
            data.component[pos] = new_data

        Examples:
            if self.data holds a numpy array, then get_pos(None, None, index) should
            just return self.index.get_pos(index)
            if self.data.images contains BATCH_SIZE images as a numpy array,
                then get_pos(None, 'images', index) should return self.index.get_pos(index)
            if self.data.labels is a dict {index: label}, then get_pos(None, 'labels', index) should return index.

            if data is not None, then you need to know in advance how to get a position for a given index.
            For instance, data is a large numpy array, a batch is a subset of this array and
            batch.index holds row numbers from a large arrays.
            Thus, get_pos(data, None, index) should just return index.

            A more complicated example of data:
            - batch represent small crops of large images
            - self.data.source holds a few large images (e.g just 5 items)
            - self.data.coords holds coordinates for crops (e.g. it contains 100 items)
            - self.data.image_no holds an array of image numbers for each crop (so it also contains 100 items)
            then get_pos(None, 'source', index) should return self.data.image_no[self.index.get_pos(index)].
            Whilst, get_pos(data, 'source', index) should return data.image_no[index].
        """
        _ = component
        if data is None:
            pos = self.index.get_pos(index)
        else:
            pos = index
        return pos

    def __getattr__(self, name):
        if self.components is not None and name in self.components:   # pylint: disable=unsupported-membership-test
            return getattr(self.data, name)
        else:
            raise AttributeError("%s not found in class %s" % (name, self.__class__.__name__))

    def __setattr__(self, name, value):
        if self.components is not None:
            if name == "_data":
                super().__setattr__(name, value)
                if self._item_class is None:
                    self.make_item_class()
                self._data_named = self._item_class(data=self._data)   # pylint: disable=not-callable
            elif name in self.components:    # pylint: disable=unsupported-membership-test
                setattr(self._data_named, name, value)
            else:
                super().__setattr__(name, value)
        else:
            super().__setattr__(name, value)

    def put_into_data(self, items, data):
        """ Loads data into _data property """
        if self.components is None:
            _src = data
        else:
            _src = data if isinstance(data, tuple) else tuple([data])
        self._data = self.get_items(items, _src)

    def get_items(self, index, data=None):
        """ Return one or several data items from a data source """
        if data is None:
            _data = self.data
        else:
            _data = data

        if self._item_class is not None and isinstance(_data, self._item_class):
            pos = [self.get_pos(None, comp, index) for comp in self.components]   # pylint: disable=not-an-iterable
            res = self._item_class(data=_data, pos=pos)    # pylint: disable=not-callable
        elif isinstance(_data, tuple):
            comps = self.components if self.components is not None else range(len(_data))
            res = tuple(data_item[self.get_pos(data, comp, index)] if data_item is not None else None
                        for comp, data_item in zip(comps, _data))
        elif isinstance(_data, dict):
            res = dict(zip(_data.keys(), (_data[comp][self.get_pos(data, comp, index)] for comp in _data)))
        else:
            ix = self.get_pos(data, None, index)
            res = _data[ix]
        return res

    def get(self, item=None, component=None):
        """ Return an item from the batch or the component """
        if item is None:
            if component is None:
                raise ValueError("item and component cannot be both None")
            return getattr(self, component)
        else:
            if component is None:
                res = self[item]
            else:
                res = self[item]
                res = getattr(res, component)
            return res

    def __getitem__(self, item):
        return self.get_items(item)

    def __iter__(self):
        for item in self.indices:
            yield self[item]

    @property
    def items(self):
        """ Init function for batch items parallelism """
        return [[self[ix]] for ix in self.indices]

    def run_once(self, *args, **kwargs):
        """ Init function for no parallelism
        Useful for async action-methods (will wait till the method finishes)
        """
        _ = self.data, args, kwargs
        return [[]]

    def infer_dtype(self, data=None):
        """ Detect dtype of batch data """
        if data is None:
            data = self.data
        return np.asarray(data).dtype.name

    def get_dtypes(self):
        """ Return dtype for batch data """
        if isinstance(self.data, tuple):
            return tuple(self.infer_dtype(item) for item in self.data)
        else:
            return self.infer_dtype(self.data)

    def get_model_by_name(self, model_name):
        """ Return a model specification given its name """
        return ModelDecorator.get_model_by_name(self, model_name)

    def get_all_model_names(self):
        """ Return all model names in the batch class """
        return ModelDecorator.get_all_model_names(self)

    def get_errors(self, all_res):
        """ Return a list of errors from a parallel action """
        all_errors = [error for error in all_res if isinstance(error, Exception)]
        return all_errors if len(all_errors) > 0 else None

    @action
    def do_nothing(self, *args, **kwargs):
        """ An empty action (might be convenient in complicated pipelines) """
        _ = args, kwargs
        return self

    @action
    @inbatch_parallel(init='indices')
    def apply_transform(self, ix, dst, src, func, *args, **kwargs):
        """ Apply a function to each item in the batch

        Args:
            dst: the destination to put the result in, can be:
                 - a string - a component name, e.g. 'images' or 'masks'
                 - an array-like - a numpy-array, list, etc
            src: the source to get data from, can be:
                 - a string - a component name, e.g. 'images' or 'masks'
                 - an array-like - a numpy-array, a list, etc
            func: a callable - a function to apply to each item from the source

        apply_transform does the following:
            for item in batch:
                self.dst[item] = func(self.src[item], *args, **kwargs)
        """
        if not isinstance(dst, str) and not isinstance(src, str):
            raise TypeError("At least one of dst and src should be attribute names, not arrays")

        if src is None:
            _args = args
        else:
            if isinstance(src, str):
                src_attr = self.get(ix, src)
            else:
                pos = self.get_pos(None, dst, ix)
                src_attr = src[pos]
            _args = tuple([src_attr, *args])

        if isinstance(dst, str):
            dst_attr = self.get(component=dst)
            pos = self.get_pos(None, dst, ix)
        else:
            dst_attr = dst
            pos = self.get_pos(None, src, ix)
        dst_attr[pos] = func(*_args, **kwargs)

    @action
    def apply_transform_all(self, dst, src, func, *args, **kwargs):
        """ Apply a function the whole batch at once

        Args:
            dst: the destination to put the result in, can be:
                 - a string - a component name, e.g. 'images' or 'masks'
                 - an array-like - a numpy-array, list, etc
            src: the source to get data from, can be:
                 - a string - a component name, e.g. 'images' or 'masks'
                 - an array-like - a numpy-array, a list, etc
            func: a callable - a function to apply to each item in the source component

        apply_transform_all does the following:
            self.dst = func(self.src, *args, **kwargs)
        """

        if not isinstance(dst, str) and not isinstance(src, str):
            raise TypeError("At least of of dst and src should be attribute names, not arrays")

        if src is None:
            _args = args
        else:
            if isinstance(src, str):
                src_attr = self.get(component=src)
            else:
                src_attr = src
            _args = tuple([src_attr, *args])

        tr_res = func(*_args, **kwargs)
        if isinstance(dst, str):
            setattr(self, dst, tr_res)
        else:
            dst[:] = tr_res
        return self

    def _get_file_name(self, ix, src, ext):
        if src is None:
            if isinstance(self.index, FilesIndex):
                src = self.index.get_fullpath(ix)
                if self.index.dirs:
                    file_name = os.path.join(src, 'data.' + ext)
                else:
                    file_name = src + '.' + ext
            else:
                raise ValueError("File locations must be specified to dump/load data")
        else:
            file_name = os.path.join(os.path.abspath(src), str(ix) + '.' + ext)
        return file_name

    def _assemble_load(self, all_res, *args, **kwargs):
        raise NotImplementedError("_assemble_load should be implemented in the child batch class")

    @inbatch_parallel('indices', post='_assemble_load', target='f')
    def _load_blosc(self, ix, src=None, components=None):
        """ Load data from a blosc packed file """
        file_name = self._get_file_name(ix, src, 'blosc')
        with open(file_name, 'rb') as f:
            data = dill.loads(blosc.decompress(f.read()))
            if self.components is None:
                components = (data.keys()[0],)
            else:
                components = tuple(components or self.components)
            item = tuple(data[i] for i in components)
        return item

    @inbatch_parallel('indices', target='f')
    def _dump_blosc(self, ix, dst, components=None):
        """ Save blosc packed data to file """
        file_name = self._get_file_name(ix, dst, 'blosc')
        with open(file_name, 'w+b') as f:
            if self.components is None:
                components = (None,)
                item = (self[ix],)
            else:
                components = tuple(components or self.components)
                item = self[ix].as_tuple(components)
            data = dict(zip(components, item))
            f.write(blosc.compress(dill.dumps(data)))

    def _load_table(self, src, fmt, components=None, *args, **kwargs):
        """ Load data from table formats: csv, hdf5, feather """
        for i, comp in enumerate(tuple(components)):
            if fmt == 'csv':
                dfr = pd.read_csv(src, *args, **kwargs)
            elif fmt == 'feather':
                dfr = feather.read_dataframe(src, *args, **kwargs)  # pylint: disable=redefined-variable-type
            elif fmt == 'hdf5':
                dfr = pd.read_hdf(src, *args, **kwargs)             # pylint: disable=redefined-variable-type

            # Put into this batch only part of it (defined by index)
            if isinstance(dfr, pd.DataFrame):
                _data = dfr.loc[self.indices]
            elif isinstance(dfr, dd.DataFrame):
                # dask.DataFrame.loc supports advanced indexing only with lists
                _data = dfr.loc[list(self.indices)].compute()
            else:
                raise TypeError("Unknown DataFrame. DataFrameBatch supports only pandas and dask.")
            setattr(self, comp, _data.iloc[:, i].values)

    @action
    def load(self, src=None, fmt=None, components=None, *args, **kwargs):  #pylint: disable=arguments-differ
        """ Load data from another array or a file """
        if fmt is None:
            self.put_into_data(self.indices, src)
        elif fmt == 'blosc':
            self._load_blosc(src, components=components, **kwargs)
        elif fmt in ['csv', 'hdf5', 'feather']:
            self._load_table(src, fmt, components, *args, **kwargs)
        else:
            raise ValueError("Unknown format " + fmt)
        return self

    @action
    def dump(self, dst=None, fmt=None, components=None):    #pylint: disable=arguments-differ
        """ Load data from another array or a file """
        if fmt is None:
            dst[self.indices] = self.data
        elif fmt == 'blosc':
            self._dump_blosc(dst, components=components)
        else:
            raise ValueError("Unknown format " + fmt)
        return self


class ArrayBatch(Batch):
    """ Base Batch class for array-like datasets
    Batch data is a numpy array.
    If components are defined, then each component data is a numpy array
    """
    @classmethod
    def merge(cls, batches, batch_size=None):
        """ Merge several batches to form a new batch of a given size """
        def make_index(data):
            """ Creates a new index for a merged batch """
            return DatasetIndex(np.arange(data.shape[0])) if data is not None and data.shape[0] > 0 else None

        if batch_size is None:
            break_point = len(batches)
            last_batch_len = len(batches[-1])
        else:
            break_point = -1
            last_batch_len = 0
            cur_size = 0
            for i, b in enumerate(batches):
                cur_batch_len = len(b)
                if cur_size + cur_batch_len >= batch_size:
                    break_point = i
                    last_batch_len = batch_size - cur_size
                    break
                else:
                    cur_size += cur_batch_len
                    last_batch_len = cur_batch_len

        components = batches[0].components or (None,)
        new_data = list(None for _ in components)
        rest_data = list(None for _ in components)
        for i, comp in enumerate(components):
            if batch_size is None:
                new_comp = [b.get(component=comp) for b in batches[:break_point]]
            else:
                new_comp = [b.get(component=comp) for b in batches[:break_point-1]] + \
                           [batches[break_point].get(component=comp)[:last_batch_len]]
            new_data[i] = np.concatenate(new_comp)

            if batch_size is not None:
                rest_comp = [batches[break_point].get(component=comp)[last_batch_len:]] + \
                            [b.get(component=comp) for b in batches[break_point:]]
                rest_data[i] = np.concatenate(rest_comp)
        new_index = make_index(new_data[0])
        rest_index = make_index(rest_data[0])

        new_batch = cls(new_index, preloaded=tuple(new_data)) if new_index is not None else None
        rest_batch = cls(rest_index, preloaded=tuple(rest_data)) if rest_index is not None else None

        return new_batch, rest_batch


    def _assemble_load(self, all_res, *args, **kwargs):
        _ = args
        if any_action_failed(all_res):
            raise RuntimeError("Cannot assemble the batch", all_res)

        if self.components is None:
            self._data = np.stack([res[0] for res in all_res])
        else:
            components = tuple(kwargs.get('components', None) or self.components)
            for i, comp in enumerate(components):
                _data = np.stack([res[i] for res in all_res])
                setattr(self, comp, _data)
        return self


class DataFrameBatch(Batch):
    """ Base Batch class for datasets stored in pandas DataFrames """
    @classmethod
    def merge(cls, batches, batch_size=None):
        return None, None

    def _assemble_load(self, all_res, *args, **kwargs):
        """ Build the batch data after loading data from files """
        _ = all_res, args, kwargs
        return self

    @action
    def dump(self, dst, fmt='feather', *args, **kwargs):
        """ Save batch data to disk
            dst should point to a directory where all batches will be stored
            as separate files named 'batch_id.format', e.g. '6a0b1c35.csv', '32458678.csv', etc.
        """
        filename = self.make_filename()
        fullname = os.path.join(dst, filename + '.' + fmt)

        if fmt == 'feather':
            feather.write_dataframe(self.data, fullname, *args, **kwargs)
        elif fmt == 'hdf5':
            self.data.to_hdf(fullname, *args, **kwargs)   # pylint:disable=no-member
        elif fmt == 'csv':
            self.data.to_csv(fullname, *args, **kwargs)   # pylint:disable=no-member
        else:
            raise ValueError('Unknown format %s' % fmt)
        return self
