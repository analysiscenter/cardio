"""Contains ECG Dataset class."""

from .. import dataset as ds
from .ecg_batch import EcgBatch


class EcgDataset(ds.Dataset):
    def __init__(self, index=None, batch_class=EcgBatch, preloaded=None, index_class=ds.FilesIndex, **kwargs):
        if index is None:
            index = index_class(**kwargs)
        super().__init__(index, batch_class, preloaded)
