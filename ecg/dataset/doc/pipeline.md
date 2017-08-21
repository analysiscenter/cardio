# Pipeline

Quite often you can't just use the data itself. You need to preprocess it beforehand. And not too rarely you end up with several processing workflows which you have to use simultaneously. That is the situation when pipelines might come in handy.

```python
class ClientTransactions(Batch):
    ...
    @action
    def some_action(self):
        ...
        return self

    @action
    def other_action(self, param):
        ...
        return self

    @action
    def yet_other_action(self):
        ...
        return self

```
To begin with, you create a dataset (client_index is an instance of [DatasetIndex](index.md)):
```python
ct_ds = Dataset(client_index, batch_class=ClientTranasactions)
```

And then you can define a workflow pipeline:
```python
trans_pipeline = (ct_ds.pipeline()
                    .some_action()
                    .other_action(param=2)
                    .yet_other_action())
```
And nothing happens! Because all the actions are lazy.
Let's run them.
```python
trans_pipeline.run(BATCH_SIZE, shuffle=False, n_epochs=1)
```
Now the dataset is split into batches and then all the actions are executed for each batch independently.

In the very same way you can define an augmentation workflow
```python
aug_wf = (image_ds.pipeline()
            .load('/some/path')
            .random_rotate(angle=(-pi/4, pi/4))
            .random_resize(factor=(0.8, 1.2))
            .random_crop(factor=(0.5, 0.8))
            .resize(shape=(256, 256)))
```
And again, no action is executed until its result is needed.
```python
NUM_ITERS = 1000
for i in range(NUM_ITERS):
    image_batch = augm_wf.next_batch(BATCH_SIZE, shuffle=True, n_epochs=None)
    # only now the actions are fired and data is changed with the workflow defined earlier
```
The original data stored in `image_ds` is left unchanged. You can call `image_ds.next_batch` in order to iterate over the source dataset without any augmentation.

## Public API

Pipelines are created from datasets.
```python
my_pipeline = my_dataset.pipeline()
```
or the shorter version:
```python
my_pipeline = my_dataset.p
```

### `next_batch(batch_size, shuffle=False, n_epochs=1, drop_last=False, prefetch=0)`
Gets a batch from the dataset, executes all the actions defined in the pipeline and then returns the result of the last action.

Args:
`batch_size` - number of items in each batch.

`shuffle` - whether to randomize items order before splitting into batches. Can be  
- `bool`: `True` / `False`
- a `RandomState` object which has an inplace shuffle method (see [numpy.random.RandomState](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.RandomState.html)):
- `int` - a random seed number which will be used internally to create a `numpy.random.RandomState` object
- `sample function` - any callable which gets an order and returns a shuffled order.

Default - `False`.

`n_epochs` - number of iterations around the whole index. If `None`, then you will get an infinite sequence of batches. Default value - 1.

`drop_last` - whether to skip the last batch if it has fewer items (for instance, if an index contains 10 items and the batch size is 3, then there will 3 batches of 3 items and the last batch with just 1 item).

`prefetch` - the number of batches processed in advance (see [details](prefetch.md))

Returns:
an instance of the batch class returned from the last action in the pipeline

Usage:
```python
for i in range(MAX_ITERS):
    batch = my_pipeline.next_batch(BATCH_SIZE, n_epochs=None)
```

### `gen_batch(batch_size, shuffle=False, n_epochs=1, drop_last=False, prefetch=0)`
Returns a batch generator.

Usage:
```python
for batch in my_pipeline.gen_batch(BATCH_SIZE, shuffle=True, n_epochs=1):
    # do something
```

### `run(batch_size, shuffle=False, n_epochs=1, drop_last=False, prefetch=0)`
Runs a pipeline.

The same as:
```python
for _ in my_pipeline.gen_batch(...):
    pass
```

### `join(another_source, one_more_source, ...)`
Joins corresponding batches from several sources (datasets or pipelines).

If you have a pipeline `images` and a pipeline `labels`, you might join them for a more convenient processing:

```python
images_with_labels = (images.p
                            .load(...)
                            .resize(shape=(256, 256))
                            .random_rotate(angle=(-pi/4, pi/4))
                            .join(labels)
                            .some_action())
```
When this pipeline is run, the following will happen for each batch of `images`:
- the actions `load`, `resize` and `random_rotate` will be executed
- a batch of `labels` with the same index will be created
- the `labels` batch will be passed into `some_action` as a first argument (after `self`, of course).

So, images batch class should look as follows:
```python
class ImagesBatch(Batch):
    def load(self, src, fmt):
        ...

    def resize(self, shape):
        ...

    def random_rotate(self, angle):
        ...

    def some_actions(self, labels_batch):
        ...
```

You can join several sources:
```python
full_images = (images.p
                     .load(...)
                     .resize(shape=(256, 256))
                     .random_rotate(angle=(-pi/4, pi/4))
                     .join(labels, masks)
                     .some_action())
```
Thus, the batches from `labels` and `masks` will be passed into `some_action` as the first and the second arguments (as always, after `self`).

### `put_into_tf_queue(session, queue, get_tensor)`
Puts the batches into a tensorflow queue.

Arguments:
    session: a tensorflow session in which a graph with the queue is executed
    queue:   the queue where the tensors will be put
    get_tensor: a callable which receives a batch as an argument and returns a tensor to put into the queue

For a detailed exlpanation see [Working with tensorflow queues](tf_queue.md).
