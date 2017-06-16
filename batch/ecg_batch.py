""" contain Batch class for processing ECGs """
import os
import sys
import numpy as np
import wfdb
sys.path.append('..')
from dataset import Batch, action, inbatch_parallel


class EcgBatch(Batch):
    """
    Batch of ECG data
    """

    def __init__(self, index):
        super().__init__(index)
        self.signal = np.ndarray(self.indices.shape, dtype=object)
        self.annotation = {}
        self.meta = {}

    @property
    def components(self):
        return "signal", "annotation", "meta"

    @action
    def load(self, src=None, fmt="wfdb"):
        """
        Loads data from different sources
        src is not used yet, so files locations are defined by the index
        """
        if fmt == "wfdb":
            self._load_wfdb(src=src)  # pylint: disable=no-value-for-parameter
        elif fmt == "npz":
            self._load_npz(src=src)  # pylint: disable=no-value-for-parameter
        else:
            raise TypeError("Incorrect type of source")

        return self

    @action
    @inbatch_parallel(init='indices', target='threads')
    def _load_wfdb(self, index, src=None):
        pos = self.index.get_pos(index)
        if src:
            path = src[index]
        else:
            path = self.index.get_fullpath(index)

        record = wfdb.rdsamp(os.path.splitext(path)[0])
        sig = record.__dict__.pop('p_signals')
        fields = record.__dict__
        self.signal[pos] = sig.T
        self.meta[pos] = fields

        try:
            annot = wfdb.rdann(path, "atr")
            self.annotation[pos] = annot
        except FileNotFoundError:
            self.annotation[pos] = None

    @action
    @inbatch_parallel(init='indices', target='threads')
    def _load_npz(self, index, src=None):
        pos = self.index.get_pos(index)
        if src:
            path = src[index]
        else:
            path = self.index.get_fullpath(index)

        data_npz = np.load(path)
        self.signal[pos] = data_npz["signal"]
        self.annotation[pos] = data_npz["annotation"]
        self.meta[pos] = data_npz["meta"].item()

    @action
    def dump(self, dst, fmt="npz"):
        """
        Save each ecg in its own file named as '<index>.<fmt>'
        """
        if fmt == "npz":
            self._dump_npz(dst=dst)  # pylint: disable=no-value-for-parameter
        else:
            raise NotImplementedError("The format is not supported yet")

        return self

    @action
    @inbatch_parallel(init='indices', target='threads')
    def _dump_npz(self, index, dst):
        signal, ann, meta = self[index]
        np.savez(
            os.path.join(dst, index + ".npz"),
            signal=signal,
            annotation=ann,
            meta=meta)
