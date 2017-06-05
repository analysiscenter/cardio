""" contain Batch class for processing ECGs """

import os
import sys
import numpy as np
import pandas as pd
import wfdb

sys.path.append('..')
from dataset import Batch, action, inbatch_parallel


class EcgBatch(Batch):
    """
    Batch of ECG data
    """

    def __init__(self, index):
        super().__init__(index)

        self._data = None
        self._annotation = self.create_annotation_df()
        self._meta = dict()
        self.history = []

    @staticmethod
    def create_annotation_df(data=None):
        """ Create a pandas dataframe with ECG annotations """
        return pd.DataFrame(data=data, columns=["ecg", "index", "value"])

    @action
    def load(self, src=None, fmt="wfdb"):
        """
        Loads data from different sources
        src is not used yet, so files locations are defined by the index
        """
        if fmt == "wfdb":
            list_of_arrs, list_of_annotations, meta = self._load_wfdb(src)
        elif fmt == "npz":
            list_of_arrs, list_of_annotations, meta = self._load_npz()
        else:
            raise TypeError("Incorrect type of source")

        # ATTENTION!
        # Construction below is used to overcome numpy bug:
        # adding empty array to list of arrays, then generating array
        # of arrays and removing the last item (empty array)
        list_of_arrs.append(np.array([]))
        self._data = np.array(list_of_arrs)[:-1]
        self._annotation = pd.concat(list_of_annotations)
        self._meta = meta

        # add info in self.history
        info = dict()
        info['method'] = 'load'
        info['params'] = {}
        self.history.append(info)

        return self

    def _wfdb_load_runner_src(self, src, list_of_arrs, list_of_annotations,
                              meta):
        for pos, ecg in np.ndenumerate(self.indices):
            path = src[ecg]
            signal, fields = wfdb.rdsamp(os.path.splitext(path)[0])
            signal = signal.T
            try:
                annot = wfdb.rdann(path, "atr")
            except FileNotFoundError:
                annot = self.create_annotation_df()  # pylint: disable=redefined-variable-type
            list_of_arrs.append(signal)
            list_of_annotations.append(annot)
            fields.update({"__pos": pos[0]})
            meta.update({ecg: fields})
        return list_of_arrs, list_of_annotations, meta

    def _wfdb_load_runner(self, list_of_arrs, list_of_annotations, meta):
        for pos, ecg in np.ndenumerate(self.indices):
            path = self.index.get_fullpath(ecg)
            signal, fields = wfdb.rdsamp(os.path.splitext(path)[0])
            signal = signal.T
            try:
                annot = wfdb.rdann(path, "atr")
            except FileNotFoundError:
                annot = self.create_annotation_df()  # pylint: disable=redefined-variable-type
            list_of_arrs.append(signal)
            list_of_annotations.append(annot)
            fields.update({"__pos": pos[0]})
            meta.update({ecg: fields})

        return list_of_arrs, list_of_annotations, meta

    def _load_wfdb(self, src=None):
        list_of_arrs = []
        list_of_annotations = []
        meta = {}
        if src:
            list_of_arrs, list_of_annotations, meta = self._wfdb_load_runner_src(
                src, list_of_arrs, list_of_annotations, meta)
        else:
            list_of_arrs, list_of_annotations, meta = self._wfdb_load_runner(
                list_of_arrs, list_of_annotations, meta)

        return list_of_arrs, list_of_annotations, meta

    def _load_npz(self):
        list_of_arrs = []
        list_of_annotations = []
        meta = {}
        for pos, ecg in np.ndenumerate(self.indices):
            path = self.index.get_fullpath(ecg)
            data = np.load(path)
            list_of_arrs.append(data["signal"])
            list_of_annotations.append(
                self.create_annotation_df(data["annotation"]))
            fields = data["meta"].item()
            fields.update({"__pos": pos[0]})
            meta.update({ecg: fields})
        return list_of_arrs, list_of_annotations, meta

    @action
    def dump(self, dst, fmt="npz"):
        """
        Save each ecg in its own file named as '<index>.<fmt>'
        """
        if fmt == "npz":
            for ecg in self.indices:
                signal, ann, meta = self[ecg]
                del meta["__pos"]
                np.savez(
                    os.path.join(dst, ecg + "." + fmt),
                    signal=signal,
                    annotation=ann,
                    meta=meta)
        else:
            raise NotImplementedError("The format is not supported yet")

    def __getitem__(self, index):
        if index in self.indices:
            pos = self._meta[index]['__pos']
            return (self._data[pos], self._annotation if self._annotation.empty
                    else self._annotation.loc[pos], self._meta[index])
        else:
            raise IndexError("There is no such index in the batch", index)

    def default_init(self, *args, **kwargs):  #pylint: disable=unused-argument
        """
        Default init for parallelism
        """
        init_indices = self.indices.tolist()
        return init_indices

    def default_post(self, list_of_results, *args, **kwargs):  #pylint: disable=unused-argument
        """
        Default post for parallelism: collect results, make a numpy array
        and change self._data attribute to it.
        """
        # Check if all elements of the resulting list are numpy arrays.
        # If not - throw an exception, otherwise rewrite self._data attribute.
        if all(isinstance(x, np.ndarray) for x in list_of_results):
            self._data = np.array(list_of_results)
        else:
            raise ValueError("List of results contains non-numpy.ndarray elements.")
        return self

    @action
    @inbatch_parallel(init='default_init', post='default_post', target='threads')
    def generate_subseqs(self, indice, segm_length, step):  #pylint: disable=unused-argument,no-self-use
        """
        Function to generate a number of subsequnces of segm_length,
        with step. Number of subseqs is defined by length of the
        initial signal and segm_lenght.
        """
        sig = self[indice][0]
        n_splits = np.int((sig.shape[1] - segm_length) / step) + 1
        splits = np.array([
            np.array(sig[:, i * step:(i * step + segm_length)]) for i in range(n_splits)
        ])
        return splits
