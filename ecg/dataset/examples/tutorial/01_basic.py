# pylint: skip-file
import os
import sys
import numpy as np
import pandas as pd

sys.path.append("../..")
from dataset import Dataset, ArrayBatch


# Make a dataset without data
def gen_data(num_items):
    index = np.arange(num_items)
    dataset = Dataset(index=index, batch_class=ArrayBatch)
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
        print("batch", i, " contains items", batch.indices)
    print("End iterating")

    print()
    print("And now with drop_last=True")
    print("Start iterating...")
    i = 0
    for batch in dataset.gen_batch(BATCH_SIZE, n_epochs=1, drop_last=True):
        i += 1
        print("batch", i, " contains items", batch.indices)
    print("End iterating")

    print()
    print("And one more time, but with next_batch(...) and too many iterations, so we will get a StopIteration")
    print("Start iterating...")
    for i in range(NUM_ITEMS * 3):
        try:
            batch = dataset.next_batch(BATCH_SIZE, shuffle=True, n_epochs=2, drop_last=True)
            print("batch", i + 1, "contains items", batch.indices)
        except StopIteration:
            print("got StopIteration")
            break
    print("End iterating")

    print()
    print("And finally with shuffle, n_epochs=None and variable batch size")
    print("Start iterating...")
    # don't forget to reset iterator to start next_batch'ing from scratch
    dataset.reset_iter()
    for i in range(int(NUM_ITEMS * 1.3)):
        batch = dataset.next_batch(BATCH_SIZE + (-1)**i * i % 3, shuffle=True, n_epochs=None)
        print("batch", i + 1, "contains items", batch.indices)
    print("End iterating")
