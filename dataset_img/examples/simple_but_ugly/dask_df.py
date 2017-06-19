# pylint: skip-file
import os
import sys
import numpy as np
import dask.dataframe as dd
import pandas as pd
import glob

sys.path.append("../..")
from dataset import * # pylint: disable=wrong-import-


# Example of custome Batch class which defines some actions
class MyDataFrameBatch(DataFrameBatch):
    @action
    def print(self):
        print("print")
        print(self.data)
        return self

    @action
    def action1(self):
        print("action 1")
        return self


df = dd.read_csv("./data/data/*.csv").set_index('i')
print(df.compute())

index = pd.Series([3,5,9]).tolist()
res = df.loc[index]
if isinstance(df, dd.DataFrame):
  print(res.compute())

dsindex = DatasetIndex(df.index.compute())
ds = Dataset(index=dsindex, batch_class=MyDataFrameBatch)

# Batch size
BATCH_SIZE = 3

# Load data and take some actions
print("\nFull preprocessing")
fp_data = (ds.pipeline()
            .load(df)
            .action1()
            .print()
            .run(BATCH_SIZE, shuffle=True))
