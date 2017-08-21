# pylint: skip-file
import os
import sys
import numpy as np
import pandas as pd

sys.path.append("../..")
from dataset import * # pylint: disable=wrong-import-

def my_shuffle(indices):
    return np.random.permutation(np.concatenate((np.array([1,1,1,2,2]), indices)))

if __name__ == "__main__":
    # number of items in the dataset
    K = 10

    # Fill-in dataset with sample data
    def pd_data():
        ix = np.arange(K)
        data = np.arange(K * 3).reshape(K, -1)
        dsindex = DatasetIndex(ix)
        ds = Dataset(index=dsindex, batch_class=ArrayBatch)
        return ds, data.copy()

    # Create datasets
    ds_data, data = pd_data()

    for drop_last in [False, True]:
        print("Start iterating... drop_last =", drop_last)
        i = 0
        for batch in ds_data.gen_batch(4, shuffle=my_shuffle, n_epochs=1, drop_last=drop_last):
            print("batch", i, ":", batch.indices)
            i += 1
        print("End iterating\n")

    print("Start iterating...")
    for i in range(K + 5):
        batch = ds_data.next_batch(3, shuffle=False, n_epochs=2, drop_last=True)
        print("batch", i, ":", batch.indices)
    print("End iterating")
