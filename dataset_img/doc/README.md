# Working with large datasets

## Index
Index holds a sequence of data item ids. As a dataset is split into batches you should have a mechanism to uniquely address each data item.
In simple cases it can be just a `numpy.arange`:
```python
dataset_index = DatasetIndex(np.arange(my_array.shape[0]))
```
`FilesIndex` is helpful when your data comes from multiple files.

See [index.md](index.md)


## Dataset
A dataset consists of an index (1-d sequence with unique keys per each data item) and a batch class which processes small subsets of data.
```python
client_ds = Dataset(dataset_index, batch_class=ArrayBatch)
```
Now you can iterate over sequential or random batches:
```python
batch = client_ds.next_batch(BATCH_SIZE, shuffle=True, n_epochs=3)
```
See [dataset.md](dataset.md)


## Batch
Batch class holds the data and contains processing functions.
Normally, you never create batch instances, as they are created in the `Dataset` or `Pipeline` batch generators.
See [batch.md](batch.md)


## Pipeline
After a batch class is created, you can define a processing workflow for the whole dataset:
```python
my_pipeline = my_dataset.pipeline()
                .load('/some/path')
                .some_processing()
                .another_processing()
                .save('/other/path')
                .run(BATCH_SIZE, shuffle=False)
```
All the methods here are actions from the [batch class](batch.md).
See further in [pipeline.md](pipeline.md).


## Within-batch parallelism
In order to accelerate data processing you can run batch methods in parallel:
```python
from dataset import Batch, inbatch_parallel, action

class MyBatch(Batch):
    ...
    @action
    @inbatch_parallel(init='_init_fn', post='_post_fn', target='threads')
    def some_action(self, item):
        # process just one item from the batch
        return some_value
```
See [parallel.md](parallel.md)


## Inter-batch parallelism
To further increase pipeline performance and eliminate inter batch delays you may process several batches in parallel:
```python
some_pipeline.next_batch(BATCH_SIZE, prefetch=3)
```
The parameter `prefetch` defines how many additional batches will be processed in the background.
See [prefetch.md](prefetch.md)


## Tensorflow queues
A pipeline might send data directly into [TensorFlow](https://www.tensorflow.org) queues:
```python
my_pipeline = my_dataset.p
                .load('/some/path')
                .preprocessing_action()
                .another_action()
                .put_into_tf_queue(queue=input_queue)
```
For details see [Working with Tensorflow queues](tf_queue.md).