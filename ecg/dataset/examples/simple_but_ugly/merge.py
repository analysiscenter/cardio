# pylint: skip-file
import os
import sys
import numpy as np
import dill
from time import time

sys.path.append("../..")
from dataset import DatasetIndex, Dataset, ArrayBatch, action, inbatch_parallel, any_action_failed



# Example of custom Batch class which defines some actions
class MyBatch(ArrayBatch):
    @property
    def components(self):
        return "images", "labels"

    @classmethod
    def merge(cls, batches, batch_size=None):
        batch, rest = super().merge(batches, batch_size)
        print("merge")
        for b in batches:
            print("   ", b.indices)
        print(batch.indices)
        return batch, rest

    @action
    def print(self, txt=None):
        print("--------------------")
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
        return None


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
    ds1, data1 = gen_data()

    res1 = (ds1.p
            .load(data1)
            .print('res1')
            .some()
            .run(2, shuffle=False, lazy=True)
    )

    res2 = (ds1.p
            .load(data1)
            .print('res2')
            .other()
            .merge(res1)
            .print("after merge")
            .run(2, shuffle=False, lazy=True)
    )

    print("Start...")
    t = time()
    res2.run() #2, shuffle=False)
    print("======================")
    #res2.run(2, n_epochs=1, prefetch=0, target='t')
    print("End", time() - t)
