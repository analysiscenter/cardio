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
    from numba import njit
except ImportError:
    pass

from ..batch import Batch
from ..decorators import action, inbatch_parallel, any_action_failed



CROP_CENTER = -1
CROP_00 = -2


@njit(nogil=True)
def random_crop_numba(images, shape):
    """ Fill-in new_images with random crops from images """
    new_images = np.zeros((images.shape[0],) + shape, dtype=images.dtype)
    if images.shape[2] - shape[0] > 0:
        origin_x = np.random.randint(0, images.shape[2] - shape[0], size=images.shape[0])
    else:
        origin_x = np.zeros(images.shape[0], dtype=np.array(images.shape).dtype)
    if images.shape[1] - shape[1] > 0:
        origin_y = np.random.randint(0, images.shape[1] - shape[1], size=images.shape[0])
    else:
        origin_y = np.zeros(images.shape[0], dtype=np.array(images.shape).dtype)
    for i in range(images.shape[0]):
        x = slice(origin_x[i], origin_x[i] + shape[0])
        y = slice(origin_y[i], origin_y[i] + shape[1])
        new_images[i, :, :] = images[i, y, x]
    return new_images

@njit(nogil=True)
def calc_origin(image_coord, shape_coord, crop):
    """ Return origin for preserve_shape """
    if crop == CROP_00:
        origin = 0
    elif crop == CROP_CENTER:
        origin = np.abs(shape_coord - image_coord) // 2
    return origin

@njit(nogil=True)
def calc_coords(image_coord, shape_coord, crop):
    """ Return coords for preserve_shape """
    if image_coord < shape_coord:
        new_image_origin = calc_origin(image_coord, shape_coord, crop)
        image_origin = 0
        image_len = image_coord
    else:
        new_image_origin = 0
        image_origin = calc_origin(image_coord, shape_coord, crop)
        image_len = shape_coord
    return image_origin, new_image_origin, image_len

@njit(nogil=True)
def preserve_shape_numba(image_shape, shape, crop=CROP_CENTER):
    """ Change the image shape by cropping and adding empty pixels to fit the given shape """
    x, new_x, len_x = calc_coords(image_shape[0], shape[0], crop)
    y, new_y, len_y = calc_coords(image_shape[1], shape[1], crop)
    new_x = new_x, new_x + len_x
    x = x, x + len_x
    new_y = new_y, new_y + len_y
    y = y, y + len_y

    return new_x, new_y, x, y



class BasicImagesBatch(Batch):
    """ Batch class for 2D images """
    @property
    def components(self):
        return "images", "labels"

    def assemble(self, all_res, *args, **kwargs):
        """ Assemble the batch after a parallel action """
        _ = all_res, args, kwargs
        raise NotImplementedError("Use ImagesBatch")

    def _assemble_load(self, all_res, *args, **kwargs):
        """ Build the batch data after loading data from files """
        _ = all_res, args, kwargs
        return self

    @action
    def load(self, src, fmt=None, components=None, *args, **kwargs):
        """ Load data """
        return super().load(src, fmt, components, *args, **kwargs)

    @action
    def dump(self, dst, fmt=None, components=None, *args, **kwargs):
        """ Save data to a file or a memory object """
        return super().dump(dst, fmt, components=None, *args, **kwargs)

    @action
    def noop(self):
        """ Do nothing """
        return self

    def get_image_size(self, image):
        """ Return an image size (width, height) """
        raise NotImplementedError("Should be implemented in child classes")

    @action
    def crop(self, component='images', origin=None, shape=None):
        """ Crop all images in the batch
        Args:
            component: string - a component name which data should be cropped
            origin: can be one of:
                    - tuple - a starting point in the form of (x, y)
                    - CROP_00 - to crop from left top edge (0,0)
                    - CROP_CENTER - to crop from center of each image
            shape: tuple - a crop size in the form of (width, height)
        """
        if origin is not None or shape is not None:
            origin = origin if origin is not None else (0, 0)
            self._crop(component, origin, shape)
        return self

    @action
    def random_crop(self, component='images', shape=None):
        """ Crop all images to a given shape and a random origin
        Args:
            component: string - a component name which data should be cropped
            shape: tuple - a crop size in the form of (width, height)

        Origin will be chosen at random to fit the required shape
        """
        if shape is None:
            raise ValueError("shape cannot be None")
        else:
            self._random_crop(component, shape)
        return self

    @action
    @inbatch_parallel(init='indices', post='assemble')
    def resize(self, ix, component='images', shape=(64, 64)):
        """ Resize all images in the batch to the given shape
        Args:
            component: string - a component name which data should be cropped
            shape: tuple - a crop size in the form of (width, height)
        """
        return self._resize_one(ix, component, shape)

    @action
    @inbatch_parallel(init='indices', post='assemble')
    def random_scale(self, ix, component='images', p=1., factor=None, preserve_shape=True, crop=CROP_CENTER):
        """ Scale the content of each image in the batch with a random scale factor
        Args:
            component: string - a component name
            p: float - a probability to apply scale
                      (0. - don't scale, .5 - scale half of images, 1 - scale all images)
            factor: tuple - min and max scale; the scale factor for each image
                              will be sampled from the uniform distribution
        """
        if factor is None:
            factor = 0.9, 1.1
        image = self.get(ix, component)
        if np.random.binomial(1, p) > 0:
            _factor = np.random.uniform(*factor)
            image_size = self.get_image_size(image)
            shape = np.round(np.array(image_size) * _factor).astype(np.int16)
            new_image = self._resize_one(ix, component, shape)
            if preserve_shape:
                new_image = self._preserve_shape(new_image, image_size, crop)
        else:
            new_image = image
        return new_image

    def _preserve_shape(self, image, shape, crop=CROP_CENTER):
        """ Change the image shape by cropping and adding empty pixels to fit the given shape """
        raise NotImplementedError()

    @action
    @inbatch_parallel(init='indices', post='assemble')
    def rotate(self, ix, component='images', angle=0, preserve_shape=True, **kwargs):
        """ Rotate all images in the batch at the given angle
        Args:
            component: string - a component name which data should be rotated
            angle: float - the rotation angle in degrees.
            preserve_shape: bool - whether to keep shape after rotating
                                   (always True for images as arrays, can be False for PIL.Images)
        """
        return self._rotate_one(ix, component, angle, preserve_shape, **kwargs)

    @action
    @inbatch_parallel(init='indices', post='assemble')
    def random_rotate(self, ix, component='images', p=1., angle=None, **kwargs):
        """ Rotate each image in the batch at a random angle
        Args:
            component: string - a component name which data should be rotated
            p: float - a probability to apply rotate
                       (0. - don't rotate, .5 - rotate half of images, 1 - rotate all images)
            angle: tuple - an angle range in the form of (min_angle, max_angle), in radians
        """
        if np.random.binomial(1, p) > 0:
            angle = angle or (-45., 45.)
            _angle = np.random.uniform(*angle)
            preserve_shape = kwargs.pop('preserve_shape', True)
            return self._rotate_one(ix, component, _angle, preserve_shape=preserve_shape, **kwargs)
        else:
            image = self.get(ix, component)
            return image

    @action
    def flip(self, axis=None):
        """ Flip images
        Args:
            axis: 'h' for horizontal (left/right) flip
                  'v' for vertical (up/down) flip
            direction: 'l', 'r', 'u', 'd'
        """
        if axis == 'h':
            return self.fliplr()
        elif axis == 'v':
            return self.flipud()
        else:
            raise ValueError("Parameter axis can be 'h' or 'v' only")


class ImagesBatch(BasicImagesBatch):
    """ Batch class for 2D images

    images are stored as numpy arrays (N, H, W) or (N, H, W, C)

    """
    def get_image_size(self, image):
        """ Return image size (width, height) """
        return image.shape[:2][::-1]

    def assemble(self, all_res, *args, **kwargs):
        """ Assemble the batch after a parallel action """
        _ = args, kwargs
        if any_action_failed(all_res):
            all_errors = self.get_errors(all_res)
            print(all_errors)
            traceback.print_tb(all_errors[0].__traceback__)
            raise RuntimeError("Could not assemble the batch")

        component = kwargs.get('component', 'images')
        try:
            new_images = np.stack(all_res)
        except ValueError as e:
            message = str(e)
            if "must have the same shape" in message:
                min_shape = np.array([x.shape for x in all_res]).min(axis=0)
                all_res = [arr[:min_shape[0], :min_shape[1]].copy() for arr in all_res]
                new_images = np.stack(all_res)
        setattr(self, component, new_images)
        return self

    @action
    def convert_to_pil(self):
        """ Convert batch data to PIL.Image format """
        if self.images is None:
            new_images = None
        else:
            new_images = np.asarray(list(None for _ in self.indices))
            self.apply_transform(new_images, 'images', PIL.Image.fromarray)
        new_data = (new_images, self.labels)
        new_batch = ImagesPILBatch(np.arange(len(self)), preloaded=new_data)
        return new_batch

    def _resize_one(self, ix, component='images', shape=None):
        """ Resize one image """
        image = self.get(ix, component)
        factor = 1. * np.asarray([*shape]) / np.asarray(image.shape[:2])
        new_image = scipy.ndimage.interpolation.zoom(image, factor, order=3)
        return new_image

    def _preserve_shape(self, image, shape, crop=CROP_CENTER):
        """ Change the image shape by cropping and/or adding empty pixels to fit the given shape """
        image_size = self.get_image_size(image)
        new_x, new_y, x, y = preserve_shape_numba(image_size, shape, crop)
        new_image_shape = shape[::-1] + image.shape[2:]
        new_image = np.zeros(new_image_shape, dtype=image.dtype)
        new_image[slice(*new_y), slice(*new_x)] = image[slice(*y), slice(*x)]
        return new_image

    def _rotate_one(self, ix, component='images', angle=0, preserve_shape=True, **kwargs):
        """ Rotate one image """
        image = self.get(ix, component)
        kwargs['reshape'] = not preserve_shape
        new_image = scipy.ndimage.interpolation.rotate(image, angle, **kwargs)
        return new_image

    @staticmethod
    def _calc_origin(image, origin, shape):
        if origin is None or origin == CROP_00:
            origin = 0, 0
        elif origin == CROP_CENTER:
            origin_x = (image.shape[1] - shape[0]) // 2 if image.shape[1] > shape[0] else 0
            origin_y = (image.shape[0] - shape[1]) // 2 if image.shape[0] > shape[1] else 0
            origin = origin_x, origin_y
        return origin

    def _crop(self, component='images', origin=None, shape=None):
        """ Crop all images in the batch """
        if origin is not None or shape is not None:
            images = self.get(None, component)

            origin = self._calc_origin(images[0], origin, shape)
            if shape is None:
                shape = images.shape[2], images.shape[1]
            if np.all(np.array(origin) + np.array(shape) > np.array(images.shape[1:3])):
                shape = images.shape[2] - origin[0], images.shape[1] - origin[1]

            x = slice(origin[0], origin[0] + shape[0])
            y = slice(origin[1], origin[1] + shape[1])
            new_images = images[:, y, x].copy()

            setattr(self, component, new_images)

    def _crop_image(self, image, origin, shape):
        origin = self._calc_origin(image, origin, shape)
        new_image = image[origin[1]:origin[1] + shape[1], origin[0]:origin[1] + shape[0]].copy()
        return new_image

    def _random_crop(self, component='images', shape=None):
        if shape is not None:
            images = self.get(None, component)
            new_images = random_crop_numba(images, shape)
            setattr(self, component, new_images)
        return self

    @action
    def fliplr(self, component='images'):
        """ Flip image horizontaly (left / right) """
        images = self.get(None, component)
        setattr(self, component, images[:, :, ::-1])
        return self

    @action
    def flipud(self, component='images'):
        """ Flip image verticaly (up / down) """
        images = self.get(None, component)
        setattr(self, component, images[:, ::-1])
        return self


class ImagesPILBatch(BasicImagesBatch):
    """ Batch class for 2D images in PIL format """
    def get_image_size(self, image):
        """ Return image size (width, height) """
        return image.size

    def assemble(self, all_res, *args, **kwargs):
        """ Assemble the batch after a parallel action """
        _ = args, kwargs
        if any_action_failed(all_res):
            all_errors = self.get_errors(all_res)
            print(all_errors)
            traceback.print_tb(all_errors[0].__traceback__)
            raise RuntimeError("Could not assemble the batch")

        component = kwargs.get('component', 'images')
        new_data = np.array(all_res, dtype='object')
        setattr(self, component, new_data)
        return self

    @action
    def convert_to_array(self, dtype=np.uint8):
        """ Convert images from PIL.Image format to an array """
        if self.images is not None:
            new_images = list(None for _ in self.indices)
            self.apply_transform(new_images, 'images', self._convert_to_array_one, dtype=dtype)
            new_images = np.stack(new_images)
        else:
            new_images = None
        new_data = new_images, self.labels
        new_batch = ImagesBatch(np.arange(len(self)), preloaded=new_data)
        return new_batch

    def _convert_to_array_one(self, image, dtype=np.uint8):
        if image is not None:
            arr = np.fromstring(image.tobytes(), dtype=dtype)
            if image.palette is None:
                new_shape = (image.height, image.width)
            else:
                new_shape = (image.height, image.width, -1)
            new_image = arr.reshape(*new_shape)
        else:
            new_image = None
        return new_image

    def _resize_one(self, ix, component='images', shape=None, **kwargs):
        """ Resize one image """
        _ = kwargs
        image = self.get(ix, component)
        new_image = image.resize(shape, PIL.Image.ANTIALIAS)
        return new_image

    def _preserve_shape(self, image, shape, crop=CROP_CENTER):
        new_x, new_y, x, y = preserve_shape_numba(image.size, shape, crop)
        new_image = PIL.Image.new(image.mode, shape)
        box = x[0], y[0], x[1], y[1]
        new_image.paste(image.crop(box), (new_x[0], new_y[0]))
        return new_image

    def _rotate_one(self, ix, component='images', angle=0, preserve_shape=True, **kwargs):
        """ Rotate one image """
        image = self.get(ix, component)
        kwargs['expand'] = not preserve_shape
        new_image = image.rotate(angle, **kwargs)
        return new_image

    def _crop_image(self, image, origin, shape):
        """ Crop one image """
        if origin is None or origin == CROP_00:
            origin = 0, 0
        elif origin == CROP_CENTER:
            origin = (image.width - shape[0]) // 2, (image.height - shape[1]) // 2
        origin_x, origin_y = origin
        shape = shape if shape is not None else (image.width - origin_x, image.height - origin_y)
        box = origin_x, origin_y, origin_x + shape[0], origin_y + shape[1]
        return image.crop(box)

    @inbatch_parallel('indices', post='assemble')
    def _crop(self, ix, component='images', origin=None, shape=None):
        """ Crop all images """
        image = self.get(ix, component)
        return self._crop_image(image, origin, shape)

    @inbatch_parallel('indices', post='assemble')
    def _random_crop(self, ix, component='images', shape=None):
        """ Crop all images with a given shape and a random origin """
        image = self.get(ix, component)
        origin_x = np.random.randint(0, image.width - shape[0])
        origin_y = np.random.randint(0, image.height - shape[1])
        return self._crop_image(image, (origin_x, origin_y), shape)

    @action
    @inbatch_parallel('indices', post='assemble')
    def fliplr(self, ix, component='images'):
        """ Flip image horizontaly (left / right) """
        image = self.get(ix, component)
        return image.transpose(PIL.Image.FLIP_LEFT_RIGHT)

    @action
    @inbatch_parallel('indices', post='assemble')
    def flipud(self, ix, component='images'):
        """ Flip image verticaly (up / down) """
        image = self.get(ix, component)
        return image.transpose(PIL.Image.FLIP_TOP_BOTTOM)
