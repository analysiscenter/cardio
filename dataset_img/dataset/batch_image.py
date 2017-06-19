""" Contains Batch classes for images """

import os   # pylint: disable=unused-import
import traceback

try:
    import blosc   # pylint: disable=unused-import
except ImportError:
    pass
import numpy as np
try:
    import PIL.Image
except ImportError:
    pass
try:
    import scipy.ndimage
except ImportError:
    pass
try:
    import cv2
except ImportError:
    pass

from .batch import Batch
from .decorators import action, inbatch_parallel, any_action_failed


class ImagesBatch(Batch):
    """ Batch class for 2D images """
    def __init__(self, index, preloaded=None):
        super().__init__(index, preloaded)
        self._new_attr = None

    @property
    def components(self):
        return "images", "labels", "masks"

    def assemble(self, all_res, *args, **kwargs):
        """ Assemble the batch after a parallel action """
        _ = args, kwargs
        if any_action_failed(all_res):
            all_errors = self.get_errors(all_res)
            print(all_errors)
            traceback.print_tb(all_errors[0].__traceback__)
            raise RuntimeError("Could not assemble the batch")

        attr = kwargs.get('attr', 'images')
        if isinstance(all_res[0], PIL.Image.Image):
            setattr(self, attr, all_res)
        else:
            setattr(self, attr, np.transpose(np.dstack(all_res), (2, 0, 1)))
        return self

    @action
    def convert_to_PIL(self, attr='images'):   # pylint: disable=invalid-name
        """ Convert batch data to PIL.Image format """
        self._new_attr = list(None for _ in self.indices)
        self.apply_transform('_new_attr', attr, PIL.Image.fromarray)
        setattr(self, attr, self._new_attr)
        return self

    @action
    @inbatch_parallel(init='images', post='assemble')
    def resize(self, image, shape, method=None):
        """ Resize all images in the batch
        if batch contains PIL images or if method is 'PIL',
        uses PIL.Image.resize, otherwise scipy.ndimage.zoom
        We recommend to install a very fast Pillow-SIMD fork """
        if isinstance(image, PIL.Image.Image):
            return image.resize(shape, PIL.Image.ANTIALIAS)
        else:
            if method == 'PIL':
                new_image = PIL.Image.fromarray(image).resize(shape, PIL.Image.ANTIALIAS)
                new_arr = np.fromstring(new_image.tobytes(), dtype=image.dtype)
                if len(image.shape) == 2:
                    new_arr = new_arr.reshape(new_image.height, new_image.width)
                elif len(image.shape) == 3:
                    new_arr = new_arr.reshape(new_image.height, new_image.width, -1)
                return new_arr
            elif method == 'cv2':
                new_shape = shape[1], shape[0]
                return cv2.resize(image, new_shape, interpolation=cv2.INTER_CUBIC)
            else:
                factor = 1. * np.asarray(shape) / np.asarray(image.shape)
                return scipy.ndimage.zoom(image, factor, order=3)

    @action
    def load(self, src, fmt=None):
        """ Load data """
        return super().load(src, fmt)

    @action
    def dump(self, dst, fmt=None):
        """ Saves data to a file or a memory object """
        _ = dst, fmt
        return self
