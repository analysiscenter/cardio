# pylint: skip-file
import os
import sys
import numpy as np
import pandas as pd

sys.path.append("../..")
from dataset import *

# Example of custome Batch class which defines some actions
class MyArrayBatch(ArrayBatch):
    @action
    def after_join(self, batch):
        print("after join", batch.indices)
        return self

    @action
    def after_multijoin(self, batch1, batch2):
        print("after multi join", batch1.indices, batch2.indices)
        return self


if __name__ == "__main__":
    # number of items in the dataset
    K = 10

    # Fill-in dataset with sample data
    def gen_data():
        ix = np.arange(K)
        data = np.arange(K * 3).reshape(K, -1)
        dsindex = DatasetIndex(ix)
        ds = Dataset(index=dsindex, batch_class=MyArrayBatch)
        return ds, data.copy()

    def gen_labels(dsindex):
        data = np.arange(K)
        ds = Dataset(index=dsindex, batch_class=MyArrayBatch)
        return ds, data.copy()

    def gen_masks(dsindex):
        data = np.arange(K) + 100
        ds = Dataset(index=dsindex, batch_class=MyArrayBatch)
        return ds, data.copy()

    # Create datasets
    ds_data, data = gen_data()
    ds_labels, labels = gen_labels(ds_data.index)
    ds_masks, masks = gen_masks(ds_data.index)

    pp_labels = ds_labels.p.load(labels)

    pp_masks = ds_masks.p.load(masks)

    pp_data = (ds_data.p
                .load(data)
                .join(pp_labels)
                .after_join()
                .join(pp_labels, pp_masks)
                .after_multijoin())


    print("Start iterating...")
    pp_data.run(3, shuffle=False, n_epochs=1)
    print("End iterating")
