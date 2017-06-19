# Working with TensorFlow queues

A pipeline might send data directly into [TensorFlow](https://www.tensorflow.org) queues.


To begin with, you need a [batch class](batch.md):
```python
class MyBatch(Batch):
    def get_tensor(self):
        return np.asarray(self.data)

    @action
    def load(path):
        self._data = ...

    @action
    def preprocessing_action(self):
        # do something with self._data
        return self

    @action
    def another_action(self):
        # do something else with self._data
        return self

    @action
    def one_more_action(self):
        # do something with self._data
        return self
```
Take a look at `get_tensor` method. It should return a numpy array that will be fed into a TensorFlow queue.

Create a queue:
```python
input_queue = tf.FIFOQueue(capacity=5, dtypes='float')
```

Define a tensorflow model:
```python
next_batch_tensor = input_queue.dequeue()
model_output = ... # define your operations
cost = tf.reduce_mean(model_output)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
```

Here comes a [pipeline](pipeline.md) definition:
```python
my_pipeline = my_dataset.p
                .load('/some/path')
                .preprocessing_action()
                .another_action()
                .one_more_action()
                .put_into_tf_queue(queue=input_queue, get_tensor=MyBatch.get_tensor)
```
So there is a queue at the end of the pipeline - after the last action.

After we have defined batch actions, a computational graph and a pipeline, we create and initialize a TensorFlow session:
```python
sess = tf.Session()
sess.run(tf.initialize_all_variables())
```


And now let's iterate over batches and train the model:
```python
for i in range(MAX_ITER):
    batch = my_pipeline.next_batch(BATCH_SIZE, n_epochs=None, prefetch=4, tf_session=sess)
    # run one optimization step for the current batch
    sess.run([optimizer])
```
Note that we don't use `batch` here as it was already put into the input queue (in fact, not the batch itself, but `batch.get_tensor()`).

Also since actions are executed in different threads, a TensorFlow session should be stated explicitly when calling pipeline's `next_batch`, `gen_batch` or `run` methods.

### Attention!

1. There is not much point in using TensorFlow queues without batch `prefetch`ing.
1. For a greater performance put `next_batch` and `sess.run([optimizer])` into different threads.
1. As actions are fired in parallel and asyncronously, the order of batches in the queue cannot be preserved.
More specificaly, in the example above `optimizer` might be evalated on another batch, not the one stored in the `batch` variable.
This is one more reason to separate `next_batch` and `sess.run(...)` into different threads.
1. If you don't separate batch generation and model evaluation in different threads, than you should be aware that a pipeline will eventually hang up if `prefetch` is equal or greater then a queue capacity. So you should not prefetch more batches than the queue might accomodate.
