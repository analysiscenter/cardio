# pylint: skip-file
import os
import sys
import numpy as np
import pandas as pd

sys.path.append("../..")
from dataset import Dataset, Batch


class MyBatch(Batch):
    @property
    def components(self):
        return "features", "labels"


# Make a dataset with sample data
def gen_data(num_items):
    features_array = np.arange(num_items * 3).reshape(num_items, -1)
    labels_array = np.random.choice(10, size=num_items)
    data = features_array, labels_array

    index = np.arange(num_items)
    # when your data fits into memory, just preload it
    dataset = Dataset(index=index, batch_class=MyBatch, preloaded=data)
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
        print("and batch data consists of features:")
        print(batch.features)
        print("and labels:", batch.labels)

        print()
        print("    You can iterate over batch items:")
        for item in batch:
            print("      item features:", item.features, "    item label:", item.labels)

        print()
        print("    You can change batch data, even scalars:")
        item = batch[batch.indices[0]]
        item.labels = 100
        print("     ", batch.labels)

    print("End iterating")
