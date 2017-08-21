# pylint: skip-file
import os
import sys
import numpy as np
import pandas as pd

sys.path.append("../..")
from dataset import * # pylint: disable=wrong-import-


# Example of custome Batch class which defines some actions
class MyDataFrameBatch(DataFrameBatch):
    @action
    def print(self, ds=None, text=None):
        if text is not None:
            print(text)
        print(self.data)
        if ds is not None:
           print('Joined data')
           print(ds[0].data)
        return self

    @action
    def action1(self):
        print("action 1")
        return self

    @action
    def action2(self):
        print("action 2")
        return self

    @action
    def action3(self):
        print("action 3")
        return self

    @action
    def add(self, inc):
        self._data += inc
        return self

# number of items in the dataset
K = 10

# Fill-in dataset with sample data
def pd_data():
    ix = np.arange(K).astype('str')
    data = pd.DataFrame(np.arange(K * 3).reshape(K, -1), index=ix)
    dsindex = DatasetIndex(data.index)
    ds = Dataset(index=dsindex, batch_class=MyDataFrameBatch)
    return ds, data.copy()

# Fill-in target dataset with sample data
def pd_target(dsindex):
    data = pd.DataFrame(np.arange(K).reshape(K, -1) * 10, index=dsindex.index)
    ds = Dataset(index=dsindex, batch_class=MyDataFrameBatch)
    return ds, data.copy()


# Create datasets
ds_data, data = pd_data()
ds_target, target = pd_target(ds_data.index)


# Batch size
BATCH_SIZE = 3

# Load data and take some actions
print("\nFull preprocessing")
fp_data = (ds_data.pipeline()
            .load(data)
            .action1()
            .action2())
# nothing has been done yet, all the actions are lazy
# Now run the actions once for each batch
fp_data.run(BATCH_SIZE, shuffle=False)
# The last batch has fewer items as run makes only one pass through the dataset

print("\nLoad and preprocess target")
# Define target preprocessing procedure and run it
fp_target = (ds_target.pipeline()
                .load(target)
                .add(100)
                .print()
                .run(BATCH_SIZE, shuffle=False))

# Preprocessing does not mute the source data
print("\nOriginal target left unchanged")
print(target)


fp_t2 = (ds_target.pipeline()
                .load(target)
                .add(1000)
                .print(text="   T2"))

# Now define some processing pipeline which will run during training
lazy_pp_data = (ds_data.pipeline()
                .load(data)
                .action1())
lazy_pp_target = (ds_target.pipeline()
                    .load(target)
                    .add(5)
                    .add(1)
                    .join(fp_t2)
                    .print(text="   PP Target"))
# Nothing has been done yet

# Define dataset which is based on lazy processing
ds_full = FullDataset(lazy_pp_data, lazy_pp_target)

print("\n\nPreproces one batch at a time")
for i in range(5):
    print("\n\nNext batch")
    # all the actions are fired when you call next_batch
    b_data, b_target = ds_full.next_batch(BATCH_SIZE, shuffle=True, n_epochs=None)
    # Because of `n_epochs=None` all the batches will have equal size
    # and shuffle dictates randomly change the order of items


ds_data.cv_split([0.5, 0.3])
print("\nTrain full preprocessing")
pp = ds_data.train.pipeline().load(data).action1().action2().dump("./data", "csv").run(BATCH_SIZE, shuffle=False)
