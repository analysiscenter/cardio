# pylint: skip-file
import os
import sys
import numpy as np
import scipy.ndimage
from time import time

sys.path.append("../..")
from dataset import Dataset
from dataset.image import ImagesBatch, CROP_CENTER

if __name__ == "__main__":
    # number of items in the dataset
    K = 6
    S = 10

    # Fill-in dataset with sample data
    def gen_data(num_items, shape):
        index = np.arange(num_items)
        data = np.random.randint(0, 255, size=num_items * shape[0] * shape[1])
        data = data.reshape(num_items, shape[0], shape[1]).astype('uint8')
        ds = Dataset(index=index, batch_class=ImagesBatch)
        return ds, data


    # Create a dataset
    print("Generating...")
    dataset, images = gen_data(K, (S, S))

    pipeline = (dataset.p
                .load((images, None, None))
                .resize(shape=(384, 384))
                .crop(shape=(360, 360))
                .crop(origin=CROP_CENTER, shape=(300, 300))
                .random_scale(p=.5, factor=(.8, 1.2))
                .rotate(angle=np.pi/8, preserve_shape=True)
                .random_rotate(p=1., angle=(-np.pi/4, np.pi/4), preserve_shape=True)
                .random_crop(shape=(200, 200))
                .convert_to_pil()
                .rotate(angle=np.pi/8, preserve_shape=True)
                .resize(shape=(384, 384))
                .random_rotate(angle=(-np.pi/4, np.pi/4), preserve_shape=True)
                .crop(origin=CROP_CENTER, shape=(300, 300))
                .random_crop(shape=(200, 200))
                .random_scale(p=.5, factor=(.8, 1.2))
                .convert_to_array()
                .apply_transform('images', 'images', scipy.ndimage.filters.gaussian_filter, sigma=3)
                .crop(shape=(128, 128))
    )

    print("Start...")
    t = time()
    pipeline.run(K//3, n_epochs=1, prefetch=0, aarget='mpc')
    print("End", time() - t)
