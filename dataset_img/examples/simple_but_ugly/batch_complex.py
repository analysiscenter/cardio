# pylint: skip-file
import os
import sys
import numpy as np
from numba import njit
from time import time

sys.path.append("../..")
from dataset import DatasetIndex, Dataset, Batch, action, inbatch_parallel, any_action_failed


# Example of custom Batch class which defines some actions
class MyBatch(Batch):
    @property
    def components(self):
        return "images", "corners", "shapes", "image_no"

    def get_pos(self, data, component, index):
        if data is None:
            data = self.data
            pos = self.index.get_pos(index)
        else:
            pos = index

        if component == 'images':
            return data.image_no[pos]
        else:
            return pos

    def get_items(self, index, data=None):
        item = super().get_items(index, data)
        l, t, w, h = *item.corners, *item.shapes
        item = item._replace(images=item.images[l:l+w, t:t+h], corners=(0,0), image_no=0)
        return item

    @action
    def load(self, src, fmt=None):
        print("load")
        self._data = src
        return self

    @action
    @inbatch_parallel('indices')
    def sto(self, ix):
        self.images[self.image_no[ix]] = np.diag(np.diag(self.images[self.image_no[ix]]))
        pass

    @action
    @inbatch_parallel('items')
    def some(self, item):
        print(item)

    @action
    @inbatch_parallel('indices')
    def other(self, ix):
        item = self[ix]
        print()
        print("item")
        print("    ", type(item))
        print("image")
        print("     ", item.images) #s[self.image_no[ix]])


if __name__ == "__main__":
    # number of items in the dataset
    K = 12
    S = 12

    # Fill-in dataset with sample data
    def gen_data():
        ix = np.arange(K)
        images = np.random.randint(0, 255, size=K//2*S*S).reshape(-1, S, S).astype('uint8')
        top = np.random.randint(0, 3, size=K*2).reshape(-1, 2).astype('uint8')
        size = np.random.randint(3, 7, size=K*2).reshape(-1, 2).astype('uint8')
        pos = np.random.choice(range(len(images)), replace=True, size=K)
        data = images, top, size, pos

        dsindex = DatasetIndex(ix)
        ds = Dataset(index=dsindex, batch_class=MyBatch)
        return ds, data


    # Create datasets
    print("Generating...")
    ds_data, data = gen_data()

    #res = ds_data.p.load(data).convert_to_PIL('images').resize((384, 384))
    #res = ds_data.p.load(data).resize((384, 384), method='cv2')
    res = ds_data.p.load(data).some().other().sto().some()
    #res = ds_data.p.load(data).transform(embarassingly_parallel_one)

    print("Start...")
    t = time()
    res.run(2, n_epochs=1, prefetch=0)
    print("End", time() - t)
