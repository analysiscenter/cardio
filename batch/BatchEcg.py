""" contain Batch class for storing ECGs """

from copy import deepcopy
import sys
import numpy as np
import pandas as pd
import wfdb

sys.path.append('..')
from dataset import Batch



class BatchEcg(Batch):
    """
    TODO
    """

    def __init__(self, index):
        super().__init__(index)

        self._data = None
        self._annotation = pd.DataFrame(columns=["ecg", "index", "value"])
        self._meta = dict()
        self._ecg_index_path = dict()
        self._ecg_index_number = dict()
        self.history = []

    def load(self, all_ecg_paths, btype="wfdb"):
        """
        Loads data from different sources
        """
        self._ecg_index_path = {ecg: all_ecg_paths[ecg] for ecg in self.index}
        self._ecg_index_number = {
            self.index[i]: i
            for i in range(len(self.index))
        }

        if btype == "wfdb":
            list_of_arrs, list_of_annotations = self._load_wfdb()
        elif btype == "npz":
            list_of_arrs, list_of_annotations = self._load_npz()
        else:
            raise TypeError("Incorrect type of batch source")

        max_length = np.max([a.shape[1] for a in list_of_arrs])
        self._data = np.asarray([
            np.pad(sig, ((0, 0), (0, max_length - sig.shape[1])), "constant")
            for sig in list_of_arrs
        ])
        self._annotation = pd.concat(list_of_annotations)

        # add info in self.history
        info = dict()
        info['method'] = 'load'
        info['params'] = {}
        self.history.append(info)

        return self

    def _load_wfdb(self):
        list_of_arrs = []
        list_of_annotations = []
        for ecg in self.index:
            path = self._ecg_index_path[ecg]
            sig, fields = wfdb.rdsamp(path)
            sig = sig.reshape((sig.shape[1], -1))
            try:
                annot = wfdb.rdann(path, "atr")
            except FileNotFoundError:
                annot = pd.DataFrame(columns=["ecg", "index", "value"])
            fields.update({"init_length": sig.shape[1]})
            list_of_arrs.append(sig)
            list_of_annotations.append(annot)
            self._meta.update({self._ecg_index_number[ecg]: fields})

        return list_of_arrs, list_of_annotations

    def _load_npz(self):
        list_of_arrs = []
        list_of_annotations = []
        for ecg in self.index:
            path = self._ecg_index_path[ecg]
            data = np.load(path+".npz")
            list_of_arrs.append(data["arr_0"])
            list_of_annotations.append(pd.DataFrame(data["arr_1"],
                                                    columns=["ecg",
                                                             "index",
                                                             "value"]))
            fields = deepcopy(data["arr_2"].item())
            fields.update({"init_length": data["arr_0"].shape[1]})
            self._meta.update({self._ecg_index_number[ecg]: fields})
        return list_of_arrs, list_of_annotations

    def _load_arrays(self):
        raise NotImplementedError()

    def dump(self, path, fmt):
        """
        Dumps each ecg in its own file with filename identical to the index
        """
        if fmt == "npz":
            for ecg in self.index:
                np.savez(path+ecg+"."+fmt,
                         self[ecg][0][:, :self[ecg][2]["init_length"]],
                         self[ecg][1],
                         self[ecg][2])
        else:
            raise NotImplementedError("The format is not supported yet")

    def __getitem__(self, index):
        """
        Indexation of ecgs by []
        """
        if isinstance(index, int):
            if (np.min(list(self._ecg_index_number.values())) <= index
                    <= np.max(list(self._ecg_index_number.values()))):
                return (self._data[index, :, :],
                        self._annotation if self._annotation.empty else
                        self._annotation.loc[self._annotation.ecg ==
                                             self._ecg_index_number[index], :],
                        self._meta[index])
            else:
                raise IndexError("Index of ecg in the batch is out of range")
        else:
            if index in self.index:
                return (self._data[self._ecg_index_number[index], :, :],
                        self._annotation if self._annotation.empty else
                        self._annotation.loc[self._annotation.ecg == index, :],
                        self._meta[self._ecg_index_number[index]])
            else:
                raise IndexError(
                    "There is no ecg with this index in the batch")

    @property
    def ecg_names_paths(self):
        """
        Return list of tuples containing ecg name and its data directory
        """
        return list(self._ecg_index_path.items())

    @property
    def ecg_indices(self):
        """
        Return ordered list of ecg names.
        """
        return list(self._ecg_index_number.keys())

    @property
    def ecg_paths(self):
        """
        Return ordered list of ecg data directories.
        """
        return list(self._ecg_index_path.values())

    @property
    def ecg_names_indexes(self):
        """
        Return list of tuples containing ecg index
            and its number in batch.
        """
        return list(self._ecg_index_number.items())


if __name__ == "__main__":
    pass
