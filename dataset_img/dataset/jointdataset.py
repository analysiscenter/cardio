""" FullDataset """

import numpy as np
from .base import Baseset
from .dsindex import DatasetIndex
from .dataset import Dataset
from .pipeline import Pipeline


class JointDataset(Baseset):
    """ Dataset comprising several Datasets """
    def __init__(self, datasets, align='order', *args, **kwargs):
        if not isinstance(datasets, (list, tuple)) or len(datasets) == 0:
            raise TypeError("Expected a non-empty list-like with instances of Dataset or Pipeline.")
        else:
            index_len = None
            for dataset in datasets:
                if not isinstance(dataset, (Dataset, Pipeline)):
                    raise TypeError("Dataset or Pipeline is expected, but instead %s was given." % type(dataset))
                ds_ilen = len(dataset.index)
                if index_len is None:
                    index_len = ds_ilen
                elif index_len != ds_ilen:
                    raise TypeError("All datasets should have indices of the same length.")

        if isinstance(align, bool):
            _align = align
        elif align in ['same', 'order']:
            _align = align == 'order'
        else:
            raise ValueError("align should be one of 'order', 'same', True or False")

        self.align = _align
        self.datasets = datasets
        super().__init__(datasets, self.align, *args, **kwargs)


    @staticmethod
    def build_index(datasets, align):   # pylint: disable=arguments-differ
        """ Create a common index for all included datasets """
        if align:
            return DatasetIndex(np.arange(len(datasets[0])))
        else:
            return datasets[0].index


    def create_subset(self, index):
        """ Create new JointDataset from a subset of indices """
        ds_set = list()
        ds_index = self.index.create_batch(index, pos=self.align)
        for dataset in self.datasets:
            ds_set.append(type(dataset).from_dataset(dataset, ds_index))
        return JointDataset(ds_set, align='same')


    def create_batch(self, batch_indices, pos=True, *args, **kwargs):
        """ Create a list of batches from all source datasets """
        ds_batches = list()
        for dataset in self.datasets:
            ds_batches.append(dataset.create_batch(batch_indices, pos=self.align, *args, **kwargs))
        return ds_batches


class FullDataset(JointDataset):
    """ Dataset which include data and target sub-Datasets """
    def __init__(self, data, target):
        super().__init__((data, target))

    @property
    def data(self):
        """ Data dataset """
        return self.datasets[0]

    @property
    def target(self):
        """ Target dataset """
        return self.datasets[1]
