""" Contains MNIST dataset """

import os
import tempfile
import urllib
import gzip
import numpy as np

from . import ImagesOpenset
from .. import DatasetIndex, parallel, any_action_failed


class MNIST(ImagesOpenset):
    """ MNIST dataset """
    TRAIN_IMAGES_URL = "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
    TRAIN_LABELS_URL = "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
    TEST_IMAGES_URL = "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
    TEST_LABELS_URL = "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
    ALL_URLS = [TRAIN_IMAGES_URL, TRAIN_LABELS_URL, TEST_IMAGES_URL, TEST_LABELS_URL]

    def __init__(self):
        super().__init__(train_test=True)
        self.cv_split()

    @property
    def _get_from_urls(self):
        """ List of URLs and type of content (0 - images, 1 - labels) """
        return [[self.ALL_URLS[i], i % 2] for i in range(len(self.ALL_URLS))]

    def _gather_data(self, all_res):
        if any_action_failed(all_res):
            raise IOError('Could not download files:', all_res)
        else:
            train_data = all_res[0], all_res[1]
            test_data = all_res[2], all_res[3]
            self._train_index = DatasetIndex(np.arange(len(train_data[0])))
            self._test_index = DatasetIndex(np.arange(len(test_data[0])))
        return train_data, test_data

    @parallel(init='_get_from_urls', post='_gather_data')
    def download(self, url, content):    # pylint:disable=arguments-differ
        """ Load data from the web site """
        tmpdir = tempfile.gettempdir()
        filename = os.path.basename(url)
        localname = os.path.join(tmpdir, filename)
        if not os.path.isfile(localname):
            urllib.request.urlretrieve(url, localname)
            print("Downloaded", filename)

        with open(localname, 'rb') as f:
            data = self._extract_images(f) if content == 0 else self._extract_labels(f)
        return data

    #
    #  _read32, extract_images, extract_labels are taken from tensorflow
    #
    @staticmethod
    def _read32(bytestream):
        dtype = np.dtype(np.uint32).newbyteorder('>')
        return np.frombuffer(bytestream.read(4), dtype=dtype)[0]

    def _extract_images(self, f):
        """Extract the images into a 4D uint8 numpy array [index, y, x, depth].
        Args:
          f: A file object that can be passed into a gzip reader.
        Returns:
          data: A 4D uint8 numpy array [index, y, x, depth].
        Raises:
          ValueError: If the bytestream does not start with 2051.
        """
        print('Extracting', f.name)
        with gzip.GzipFile(fileobj=f) as bytestream:
            magic = self._read32(bytestream)
            if magic != 2051:
                raise ValueError('Invalid magic number %d in MNIST image file: %s' % (magic, f.name))
            num_images = self._read32(bytestream)
            rows = self._read32(bytestream)
            cols = self._read32(bytestream)
            buf = bytestream.read(rows * cols * num_images)
            data = np.frombuffer(buf, dtype=np.uint8)
            data = data.reshape(num_images, rows, cols, 1)
            return data

    def _extract_labels(self, f):
        """Extract the labels into a 1D uint8 numpy array [index].
        Args:
          f: A file object that can be passed into a gzip reader.
        Returns:
          labels: a 1D uint8 numpy array.
        Raises:
          ValueError: If the bystream doesn't start with 2049.
        """
        print('Extracting', f.name)
        with gzip.GzipFile(fileobj=f) as bytestream:
            magic = self._read32(bytestream)
            if magic != 2049:
                raise ValueError('Invalid magic number %d in MNIST label file: %s' % (magic, f.name))
            num_items = self._read32(bytestream)
            buf = bytestream.read(num_items)
            labels = np.frombuffer(buf, dtype=np.uint8)
            return labels
