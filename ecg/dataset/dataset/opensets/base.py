""" Contains the base class for open datasets """

from .. import Dataset
from ..image import ImagesBatch


class Openset(Dataset):
    """ The base class for open datasets """
    def __init__(self, index=None, batch_class=None, train_test=False):
        self.train_test = train_test
        self._train_index, self._test_index = None, None
        self._data = self.download()
        preloaded = self._data if not train_test else None
        super().__init__(index, batch_class, preloaded=preloaded)

    @staticmethod
    def build_index(index):
        """ Create an index """
        if index is not None:
            return super().build_index(index)
        else:
            return None

    def download(self):
        """ Download a dataset from the source web-site """
        return None

    def cv_split(self, shares=0.8, shuffle=False):
        if self.train_test:
            train_data, test_data = self._data  # pylint:disable=unpacking-non-sequence
            self.train = Dataset(self._train_index, self.batch_class, preloaded=train_data)
            self.test = Dataset(self._test_index, self.batch_class, preloaded=test_data)
        else:
            super().cv_split(shares, shuffle)


class ImagesOpenset(Openset):
    """ The base class for open datasets with images """
    def __init__(self, index=None, batch_class=ImagesBatch, train_test=False):
        super().__init__(index, batch_class, train_test)
