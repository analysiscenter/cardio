# pylint: skip-file
import os
import sys
import numpy as np

sys.path.append("../..")
from dataset import Dataset, Batch, Pipeline, action, inbatch_parallel


# Example of custome Batch class which defines some actions
class MyBatch(Batch):
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
        print("action 1", self.indices)
        return self

    @action
    def action2(self):
        """ action2 """
        print("  action 2", self.indices)
        return self

    @action
    def action3(self):
        print("action 3", self.indices)
        return self

    #@action
    @inbatch_parallel('indices')
    def par(self):
        """ Parallel like nobody's watching """
        return self

# number of items in the dataset
K = 10

# Fill-in dataset with sample data
def gen_data(num_items):
    ix = np.arange(num_items).astype('str')
    data = np.arange(num_items * 3).reshape(num_items, -1)
    ds = Dataset(index=ix, batch_class=MyBatch, preloaded=data)
    return ds, data


# Create datasets
ds_data, data = gen_data(K)


# Batch size
BATCH_SIZE = 3

# Load data and take some actions
print("\nFull preprocessing")
with Pipeline() as p:
    pipe1 = p.action1().action2() @ .5 + p.action3() * 2 @ .5 + p.action3() * 3 @ .2
    #pipe1 = p.action1() @ .7 + p.action2() * 2 @ 0.5 * 3
    #pipe2 = p.action2() @ .3 * 3 * 2 + p.action3() * 4 @ .4
    pipe = pipe1
    #pipe = pipe1 * 2  + pipe2
    #pipe = p.action1() @ .3 * 2 + p.action2()

fp_data = pipe << ds_data


def print_pipe(level, pipe):
    #print("  " * level, pipe.proba, pipe.repeat)
    for a in pipe._action_list:
        if 'pipeline' in a:
            print("  " * level, a)
            print_pipe(level + 1, a['pipeline'])
        else:
            print("  " * level, a)

print_pipe(0, pipe)
#ds_data.p.action1().action2().run(BATCH_SIZE)


# nothing has been done yet, all the actions are lazy
# Now run the actions once for each batch
fp_data.run(BATCH_SIZE, shuffle=False)
# The last batch has fewer items as run makes only one pass through the dataset
