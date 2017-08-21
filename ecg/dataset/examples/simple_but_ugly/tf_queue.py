# pylint: skip-file
import os
import sys
import numpy as np
import pandas as pd

sys.path.append("../..")
from dataset import *

# Example of custome Batch class which defines some actions
class MyArrayBatch(ArrayBatch):
    def get_tensor(self):
        return self.data

    @action
    def after_join(self, batch):
        print("after join", batch.indices)
        return self

    @action
    def after_multijoin(self, batch1, batch2):
        print("after multi join", batch1.indices, batch2.indices)
        return self


if __name__ == "__main__":
    # number of items in the dataset
    K = 100
    # queue length
    Q = 5

    input_queue = tf.FIFOQueue(capacity=Q, dtypes='float')

    next_batch_tensor = input_queue.dequeue()
    model_output = tf.square(tf.reduce_sum(next_batch_tensor))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())


    # Fill-in dataset with sample data
    def gen_data():
        ix = np.arange(K)
        data = np.arange(K * 3).reshape(K, -1) * 1.
        dsindex = DatasetIndex(ix)
        ds = Dataset(index=dsindex, batch_class=MyArrayBatch)
        return ds, data.astype('float32').copy()

    # Create datasets
    ds_data, data = gen_data()

    pp_data = (ds_data.p
                .load(data)
                .put_into_tf_queue(session=sess, queue=input_queue, get_tensor=MyArrayBatch.get_tensor))

    print("Start iterating...")

    for batch in pp_data.gen_batch(3, shuffle=False, n_epochs=1, prefetch=Q-1):
        # run one model step
        print(sess.run([model_output]))

    print("End iterating")
