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
            _data = self.data
            pos = self.index.get_pos(index)
        else:
            _data = data
            pos = index

        if data is None and component == 'images':
            return data.image_no[pos]
        else:
            return pos

    def get_items(self, index, data=None):
        item = super().get_items(index, data)
        if data is None and item.images.ndim == 2:
            l, t, w, h = *item.corners, *item.shapes
            item = self._item_class((item.images[l:l+w, t:t+h], (0, 0), (h, w), 0))
        return item

    @action
    def print(self):
        print(self.items)
        return self

    @action
    @inbatch_parallel('indices')
    def sto(self, ix):
        pos = self.get_pos(None, 'image_no', ix)
        self.images[self.image_no[pos]] = np.diag(np.diag(self.images[self.image_no[pos]]))

    @action
    @inbatch_parallel('items', target='for')
    def some(self, item):
        print("some:", type(item), "len:", len(item.images)) #, item.corners, item.shapes)

    @action
    @inbatch_parallel('indices')
    def some2(self, ix):
        print("some:") #, type(self[ix])) #, item.corners, item.shapes)

    @action
    @inbatch_parallel('indices')
    def other(self, ix):
        item = self[ix]
        print()
        print("item", ix)
        print("    ", type(item))
        print("image")
        print("     len: ", len(item.images)) #, item.image_no)


if __name__ == "__main__":
    # number of items in the dataset
    K = 8
    S = 12

    # Fill-in dataset with sample data
    def gen_data():
        ix = np.arange(K)
        images = np.random.randint(0, 255, size=K//2*S*S).reshape(-1, S, S).astype('uint8')
        top = np.random.randint(0, 3, size=K*2).reshape(-1, 2).astype('uint8')
        size = np.random.randint(3, 7, size=K*2).reshape(-1, 2).astype('uint8')
        pos = np.random.choice(range(len(images)), replace=True, size=K)
        data = images, top, size, pos

        #dsindex = DatasetIndex(ix)
        ds = Dataset(index=ix, batch_class=MyBatch)
        return ds, data


    # Create datasets
    print("Generating...")
    ds_data, data = gen_data()

    res = ds_data.p.load(data).print().other() #.other() #.sto().some()

    print("Start...")
    t = time()
    res.run(2, n_epochs=1, prefetch=0, target='threads')
    print("End", time() - t)
