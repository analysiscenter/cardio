# pylint: skip-file
import os
import sys
import shutil
import numpy as np
import pandas as pd

sys.path.append("../..")
from dataset import * # pylint: disable=wildcard-import


# Create index from ./data
findex = FilesIndex(path='./data/data/*', no_ext=True)
# print list of files
print("File Index:")
print(findex.index)

print("\nSplit")
findex.cv_split([0.35, 0.35])
for dsi in [findex.train, findex.test, findex.validation]:
    if dsi is not None:
        print(dsi.index)

print("\nprint batches:")
for dsi in [findex.train, findex.test, findex.validation]:
    print("---")
    for b in dsi.gen_batch(2, n_epochs=1):
        print(b.index)


# remove directory with subdirectories
DIR_PATH = './data/dirs'
shutil.rmtree(DIR_PATH, ignore_errors=True)
# create new dirs
for i in range(3):
    for j in range(5):
        os.makedirs(os.path.join(DIR_PATH, 'dir' + str(i), str(i*5 + j)))


# Create index from ./data/dirs
p = os.path.join(DIR_PATH, 'dir*/*')
print(p)
dindex = FilesIndex(path=p, dirs=True, sort=True)
# print list of subdirectories
print("\n\nDir Index:")
print(dindex.index)

print("\nSplit")
dindex.cv_split([0.35, 0.35])
for dsi in [dindex.train, dindex.test, dindex.validation]:
    if dsi is not None:
    	for dir in dsi.indices:
           print(dir, dindex.get_fullpath(dir))
    print("---")


# Create index from several dirs in ./data/dirs
print("\nSeveral dirs")
paths = []
for i in [0, 2]:
    paths.append(os.path.join(DIR_PATH, 'dir' + str(i), '*'))
print(paths)
dindex = FilesIndex(path=paths, dirs=True, sort=True)
for dir in dindex.indices:
    print(dir, dindex.get_fullpath(dir))


# Create index from non-existent dir
findex = FilesIndex(path='sadfsdf/*')
# print list of subdirectories
print("\n\nIndex:")
print(findex.index)
