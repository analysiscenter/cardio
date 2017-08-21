# pylint: skip-file
import os
import sys
import asyncio
import numpy as np
import pandas as pd
from numba import njit
from time import time

sys.path.append("../..")
from dataset import * # pylint: disable=wrong-import-

@njit(nogil=True)
def numba_fn(k, a1=0, a2=0, a3=0):
    print("   action numba", k, "started", a1, a2, a3)
    for i in range(k * 3000):
        x = np.random.normal(0, 1, size=10000)
    print("   action numba", k, "ended")
    return x


# Example of custome Batch class which defines some actions
class MyArrayBatch(ArrayBatch):
    def __init__(self, index, *args, **kwargs):
        super().__init__(index)

    @property
    def components(self):
        return "images", "labels"

    @action
    def print(self, text=None):
        if text is not None:
            print("\n====", text, self.indices, "======\n")
        print("data:", type(self.data))
        return self

    def parallel_init(self, *args, **kwargs):
        r = self.indices.tolist()
        print("\n Init:", r)
        return r

    def parallel_post(self, results, not_done=None):
        print(" Post:", results)
        return self

    @action
    def action0(self, *args):
        print("   batch", self.indices, "   action 0", args)
        return self

    @action
    @inbatch_parallel(init="parallel_init") #, post="parallel_post")
    def action1(self, i, *args):
        print("   batch", self.indices, "   action 1", i, args)
        return i

    def action_n_init(self, *args, **kwargs):
        r = self.indices.astype('int').tolist()
        print("Parallel:", r)
        return r

    @action
    @inbatch_parallel(init="action_n_init", target="nogil")
    def action_n(self):
        return numba_fn

    @action
    @inbatch_parallel(init="parallel_init", post="parallel_post", target='async')
    async def action2(self, i, *args, **kwargs):
        print("   batch", self.indices, "action 2", i, "started", args)
        await asyncio.sleep(5)
        print("   batch", self.indices, "   action 2", i, "ended")
        return i

    @action
    def add(self, inc):
        self.data += inc
        return self


if __name__ == "__main__":
    # number of items in the dataset
    K = 10

    # Fill-in dataset with sample data
    def pd_data():
        ix = np.arange(K)
        data = np.arange(K * 3).reshape(K, -1)
        dsindex = DatasetIndex(ix)
        ds = Dataset(index=dsindex, batch_class=MyArrayBatch)
        return ds, data.copy()


    # Create datasets
    ds_data, data = pd_data()

    res = (ds_data.pipeline()
            .load(data)
            .print("Start batch")
            .action0()
            .action1()
            .action2() #loop=asyncio.get_event_loop())
            .action_n()
            #.add(1000)
            .print("End batch"))

    #res.run(4, shuffle=False)
    print("Start iterating...")
    t = time()
    #res.run(3, shuffle=False, n_epochs=1, drop_last=True, prefetch=3, target='mpc')
    print("End:", time() - t)

    i = 0
    for batch_res in res.gen_batch(3, shuffle=False, n_epochs=1, prefetch=1): #, target='mpc'):
        print('-------------------------------------------------')
        print("====== Iteration ", i, "batch:", batch_res.indices)
        i += 1
    print("====== Stop iteration ===== ")

