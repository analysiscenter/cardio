""" Contains CIFAR datasets """

import os
import tempfile
import urllib
import pickle
import tarfile
import numpy as np

from .. import DatasetIndex
from . import ImagesOpenset


class BaseCIFAR(ImagesOpenset):
    """ The base class for the CIFAR dataset """
    SOURCE_URL = None
    LABELS_KEY = None
    TRAIN_NAME_ID = None
    TEST_NAME_ID = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, train_test=True, **kwargs)
        self.cv_split()

    def download(self):
        """ Load data from a web site and extract into numpy arrays """

        def _extract(archive_file, member):
            return pickle.load(archive_file.extractfile(member), encoding='bytes')

        def _gather_extracted(all_res):
            images = np.concatenate([res[b'data'] for res in all_res]).reshape(-1, 3, 32, 32).transpose((0, 2, 3, 1))
            labels = np.concatenate([res[self.LABELS_KEY] for res in all_res])
            return images, labels

        tmpdir = tempfile.gettempdir()
        filename = os.path.basename(self.SOURCE_URL)
        localname = os.path.join(tmpdir, filename)
        if not os.path.isfile(localname):
            print("Downloading", filename, "...")
            urllib.request.urlretrieve(self.SOURCE_URL, localname)
            print("Downloaded", filename)

        print("Extracting...")
        with tarfile.open(localname, "r:gz") as archive_file:
            files_in_archive = archive_file.getmembers()

            data_files = [one_file for one_file in files_in_archive if self.TRAIN_NAME_ID in one_file.name]
            all_res = [_extract(archive_file, one_file) for one_file in data_files]
            train_data = _gather_extracted(all_res)

            test_files = [one_file for one_file in files_in_archive if self.TEST_NAME_ID in one_file.name]
            all_res = [_extract(archive_file, one_file) for one_file in test_files]
            test_data = _gather_extracted(all_res)
        print("Extracted")

        self._train_index = DatasetIndex(np.arange(len(train_data[0])))
        self._test_index = DatasetIndex(np.arange(len(test_data[0])))

        return train_data, test_data


class CIFAR10(BaseCIFAR):
    """ CIFAR10 dataset """
    SOURCE_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    LABELS_KEY = b"labels"
    TRAIN_NAME_ID = "data_batch"
    TEST_NAME_ID = "test_batch"


class CIFAR100(BaseCIFAR):
    """ CIFAR100 dataset """
    SOURCE_URL = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    LABELS_KEY = b"fine_labels"
    TRAIN_NAME_ID = "train"
    TEST_NAME_ID = "test"
