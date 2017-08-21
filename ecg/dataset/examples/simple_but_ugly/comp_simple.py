# pylint: skip-file
import os
import sys
import numpy as np
import dill
from time import time

sys.path.append("../..")
from dataset import DatasetIndex, Dataset, ArrayBatch, action, inbatch_parallel, any_action_failed


def mpc_some(item):
    print("some:",)
    dill.dumps(item)
    #print(type(item), item.images.ndim)



# Example of custom Batch class which defines some actions
class MyBatch(ArrayBatch):
    @property
    def components(self):
        return "images", "labels", "masks", "targets"

    @action
    def print(self, txt=None):
        if txt is not None:
            print(txt)
        for i in self:
            print(i)
        print("--------------------")
        return self

    @action
    @inbatch_parallel('indices', target='for')
    def other(self, ix):
        item = self[ix]
        pos = self.get_pos(None, 'images', ix)
        print("other:", ix, pos, type(item), item.images.ndim)


    @action
    @inbatch_parallel('items', target='t')
    def some(self, item=None):
        print("some:", type(item))
        print(item)
        print("len", len(dill.dumps(item.as_tuple())))
        return mpc_some


if __name__ == "__main__":
    # number of items in the dataset
    K = 4
    S = 3

    # Fill-in dataset with sample data
    def gen_data():
        ix = np.arange(K)
        images = np.random.randint(0, 255, size=K*S*S).reshape(-1, S, S).astype('uint8')
        labels = np.random.randint(0, 3, size=K).astype('uint8')
        masks = np.random.randint(0, 10, size=K).astype('uint8') + 100
        targets = np.random.randint(0, 10, size=K).astype('uint8') + 1000
        data = images, labels, masks, targets

        ds = Dataset(index=ix, batch_class=MyBatch)
        return ds, data


    # Create datasets
    print("Generating...")
    ds_data, data = gen_data()

    #res = ds_data.p.print().other().some()
    res = (ds_data.p
            .load(data)
            .print('before dump')
            .dump('../data/data2', 'blosc')
    )

    res2 = (ds_data.p
            .load('../data/data2', 'blosc', components=['images'])
            .print('after loading images')
            .load('../data/targets.csv', 'csv', components=['targets', 'masks'], header=0, index_col=False, names=['Target', 'N'])
            #.load('../data/data2', 'blosc', components=['masks', 'targets'])
            .print('after targets')
    )

    print("Start...")
    t = time()
    res.run(2, n_epochs=1, prefetch=0, target='t')
    print("======================")
    res2.run(2, n_epochs=1, prefetch=0, target='t')
    print("End", time() - t)
