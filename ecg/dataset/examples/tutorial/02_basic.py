# pylint: skip-file
import os
import sys
import numpy as np
import pandas as pd

sys.path.append("../..")
from dataset import Dataset, ArrayBatch


# Make a dataset with sample data
def gen_data(num_items):
    index = np.arange(num_items)
    data = np.arange(num_items * 3).reshape(num_items, -1)
    # when your data fits into memory, just preload it
    dataset = Dataset(index=index, batch_class=ArrayBatch, preloaded=data)
    return dataset


if __name__ == "__main__":
    # number of items in the dataset
    NUM_ITEMS = 10
    BATCH_SIZE = 3

    # Create datasets
    dataset = gen_data(NUM_ITEMS)

    print("Start iterating...")
    i = 0
    for batch in dataset.gen_batch(BATCH_SIZE, n_epochs=1):
        i += 1
        print()
        print("batch", i, " contains items", batch.indices)
        print("and batch data is")
        print(batch.data)

        print()
        print("  You can iterate over batch items:")
        for item in batch:
            print("      ", item)
    print("End iterating")
