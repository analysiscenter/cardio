# pylint: skip-file
import os
import sys
import shutil
import numpy as np
import pandas as pd

sys.path.append("../..")
from dataset import * # pylint: disable=wildcard-import


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
        self.data += inc
        return self


# remove directory with subdirectories
DIR_PATH = './data/dirs'
shutil.rmtree(DIR_PATH, ignore_errors=True)
# create new dirs
for i in range(3):
    for j in range(5):
        os.makedirs(os.path.join(DIR_PATH, 'dir' + str(i), str(i*5 + j)))


# Create index from ./data/dirs
dindex = FilesIndex(path=os.path.join(DIR_PATH, 'dir*/*'), dirs=True, sort=False)
# print list of subdirectories
print("Dir Index:")
print(dindex.index)

align = False
if align:
    oindex = DatasetIndex(np.arange(len(dindex))+100)
else:
    oindex = FilesIndex(path=os.path.join(DIR_PATH, 'dir*/*'), dirs=True, sort=False)
print("\nOrder Index:")
print(oindex.index)

ds1 = Dataset(dindex, MyDataFrameBatch)
ds2 = Dataset(oindex, MyDataFrameBatch)
jds = JointDataset((ds1,ds2), align='order' if align else 'same')

K = 5

print("\nGenerating batches")
for b1, b2 in jds.gen_batch(K, n_epochs=1):
	print(b1.indices)
	print(b2.indices)


print("\nSplit")
jds.cv_split([0.5, 0.35])
for dsi in [jds.train, jds.test, jds.validation]:
    if dsi is not None:
        print("Joint index:", dsi.index.index)
        b1, b2 = jds.create_batch(dsi.index.index)
        print("DS1:", b1.indices)
        print("DS2:", b2.indices)
        print()

print("\nTrain batches")
for b1, b2 in jds.train.gen_batch(3, shuffle=False, n_epochs=1):
    print("DS1:", b1.indices)
    print("DS2:", b2.indices)
    print()
