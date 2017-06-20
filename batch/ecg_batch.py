""" contain Batch class for processing ECGs """
import os
import sys
import numpy as np
import wfdb
sys.path.append('..')
from dataset import Batch, action, inbatch_parallel


class EcgBatch(Batch):

    """

    Сlass for storing batch of ECG (electrocardiogram) 
    signals.
    Derived from base class Batch
    
    Main attributes:
        1. index: numpy array of signal IDs. Usually string names of files
        2. _data: tuple that contains three data structures with
           relevant ECG information:
           signal - numpy array of signals; initialized as np.array of
           None's same size as index.
           annotation - dict with annotation of the signals; initialized as 
           empty dict.
           meta - dict with metadata of the signals (sampling rate, etc.);
           initialized as empty dict.

    Main methods:
        1. __init__(self, index, preloaded=None):
            Basic initialization of patient
            in accordance with Batch.__init__
            given base class Batch. Also initializes
            _data attribute.
        2. load(self, src, fmt='wfdb'):
            Load signals from files, either 'wfdb' 
            for .mat files, or 'npz' for npz files.
            returns self
        3. dump(self, dst, fmt='nz')
            Create a dump of the batch
            in the folder defined by dst
            in format defined by fmt.
            returns self


    """

    def __init__(self, index, preloaded=None):

        super().__init__(index, preloaded)
        self.signal = np.ndarray(self.indices.shape, dtype=object)
        self.annotation = {}
        self.meta = {}

    @property
    def components(self):
        return "signal", "annotation", "meta"

    @action
    def load(self, src=None, fmt="wfdb"):
        """
        Load signals, annotations and metadata from files into EcgBatch.

        Args:
            src - dict with indice-path pairs, not needed if index is created
            using path;
            fmt - format of files with data, either 'wfdb' for .mat/.atr/.hea
            files, or 'npz' for .npz files.

        Example:
            index = FilesIndex(path="/some/path/*.dcm", no_ext=True)
            batch = EcgBatch(index)
            batch.load(fmt='wfdb')

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
        Save each record with annotations and metadata
        in separate files as 'dst/<index>.<fmt>'

        Args:
            dst - string with path to save data to
            fmt - format of files, only 'npz' is supported now

        Example:
            batch = EcgBatch(ind)
            batch.load(...)
            batch.dump(dst='./dump/')
        
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

