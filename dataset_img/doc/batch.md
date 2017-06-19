# Batch class

Batch class holds the data and contains processing functions.
Normally, you never create batch instances, as they are created in the `Dataset` or `Pipeline` batch generators.


## Index
`Batch` class stores the [index](index.md) of all data items which belong to the batch. You can access the index through `self.index`. The sequence of indices is also available as `self.indices`.


## Data
The base `Batch` class has a private property `_data` which you can use to store your data in. So you can access the data within batch class methods through `self._data`. But you may also create new properties for your specific data. For example, an image batch class for segmentation purposes may have properties `_images` and `_masks` which point to corresponding image arrays.

There is also a public property `data` defined as:
```python
@property
def data(self):
    return self._data
```
This approach allows to conceal an internal data structure and provides for a more convenient and (perhaps) more stable public interface to access the data.

An earlier mentioned batch with images may redefine `data` as:
```python
@property
def data(self):
    return self._images, self._masks
```

Nevertheless, this is just a convention and you are not obliged to follow it.


## Action methods
`Action` methods form a public API of the batch class which is available in the [pipeline](pipeline.md). If you operate directly with the batch class instances, you don't need `action` methods. However, pipelines provide the most convenient interface to process the whole dataset and to separate data processing steps and model training / validation cycles.

In order to convert a batch class method to an action you add `@action` decorator:
```python
from dataset import Batch, action

class MyBatch(Batch):
    ...
    @action
    def some_action(self):
        # process your data
        return self
```
Take into account that an `action` method should return an instance of some `Batch`-class: the very same one or some other class.
If an `action` changes the instance's data directly, it may simply return `self`.


## Model definitions and model-based actions
Models and model training methods can also be a part of a batch class.

```python
class MyArrayBatch(ArrayBatch):
    ...
    @model()
    def basic_model():
        input_data = tf.placeholder('float', [None, 28])
        model_output = ...
        return [input_data, model_output]

    @action(model='basic_model')
    def train_model(self, model):
        input_data, optimizer = model
        # update gradients
        return self
```
For details see [Working with models](models.md).


## Running methods in parallel
As a batch can be quite large it might make sense to parallel the computations. And it is pretty easy to do:
```python
from dataset import Batch, inbatch_parallel, action

class MyBatch(Batch):
    ...
    @action
    @inbatch_parallel(init='_init_fn', post='_post_fn', target='threads')
    def some_action(self, item, arg1, arg2):
        # process just one item
        return some_value
```
For further details how to make parallel actions see [parallel.md](parallel.md).




## Writing your own Batch

### Constructor should include `*args` and `*kwargs`
```python
class MyBatch(Batch):
    ...
    def __init__(self, index, your_param1, your_param2, *args, **kwargs):
        super().__init__()
        # process your data
```
It is not so important if you are extremely carefull when calling batch generators and parallelizing actions, so you are absolutly sure that a batch cannot get unexpected arguments.
But usually it is just easier to add `*args` and `*kwargs` and have a guarantee that your program will not break or hang up (as it most likely will do if you do batch prefetching with multiprocessing).

### Don't load data in the constructor
The constructor should just intialize properties.
`Action`-method `load` is the best place for reading data from files or other sources.

So DON'T do this:
```python
class MyBatch(Batch):
    ...
    def __init__(self, index, your_param1, your_param2, *args, **kwargs):
        super().__init__()
        ...
        self._data = read(file)
```

Instead DO that:
```python
class MyBatch(Batch):
    ...
    def __init__(self, index, your_param1, your_param2, *args, **kwargs):
        super().__init__()
        ...

    @action
    def load(self, source, format):
        # load data from source
        ...
        self._data = read(file)
        return self
```

### (optional) Store your data in `_data` property
It is just a convenient convention which makes your life more consistent.

### (optional) Define `__getitem__` method
If you want to address batch items easily as well as iterate over your batch, you need `__getitem__` method. The default `Batch.__getitem__`  just returns an item from `data`:
```python
class MyBatch(Batch):
    ...
    def some_method(self):
        ...
        some_value = self[some_item]  # equals self.data[some_item]
        ...
```
Thus you are able to address batch items as `self[index_id]` internally (in the batch class methods) or as `batch[index_id]` externally.

If you have defined `data` as a tuple, then base `__getitem__` will also work:
```python
class MyBatch(Batch):
    ...
    def data(self):
        return self._images, self._mask
...
for i in range(MAX_ITER):
    batch = some_pipeline.next_batch(BATCH_SIZE)
    image0, mask0 = batch[index0]:
    # process one image and one corresponding mask
```
If your data has a more complex structure, you will need to write your own `__getitem__`.

###  Make `actions` whenever possible
If you create some method transforming batch data, you might want to call it as a step in the whole dataset processing `pipeline`. So make it an `action`:
```python
class MyBatch(Batch):
    ...
    @action
    def change_data(self, arg1, arg2):
        # process your data
        return self
```
`Actions` should return an instance of some batch class.

### Parallel everyting you can
If you want a really fast data processing you can't do without `numba` or `cython`.
And don't forget about input/output operations.
For more details see [parallel.md](parallel.md).

### Define `load` and `dump` action-methods
`load` and `dump` allows for a convenient and managable data flow.
```python
class MyBatch(Batch):
    ...
    @action
    def load(self, src, fmt='raw'):
        if fmt == 'raw':
            self._data = ... # load from a raw file
        elif fmt == 'blosc':
            self._data = ... # load from a blosc file
        else:
            raise ValueError("Unknown format '%s'" % fmt)
        return self

    @action
    def dump(self, dst, fmt='raw'):
        if fmt == 'raw':
            # write self.data to a raw file
        elif fmt == 'blosc':
            # write self.data to a blosc file
        else:
            raise ValueError("Unknown format '%s'" % fmt)
        return self
```
This lets you create explicit pipeline workflows:
```python
batch
   .load('/some/path', 'raw')
   .some_action(param1)
   .other_action(param2)
   .one_more_action()
   .dump('/other/path', 'blosc')
```

### Make all I/O in `async` methods
This is extremely important if you read batch data from many files.
```python
class MyBatch(Batch):
    ...
    @action
    def load(self, fmt='raw'):
        if fmt == 'raw':
            self._data = self._load_raw()
        elif fmt == 'blosc':
            self._data = self._load_blosc()
        else:
            raise ValueError("Unknown format '%s'" % fmt)
        return self

    @inbatch_parallel(init='_init_io', post='_post_io', target='async')
    async def _load_raw(self, item):
        # load one data item from a raw format file
        return loaded_item

    def _init_io(self):
        return [[item_id, self.index.get_fullpath(item_id)] for item_id in self.indices]

    def _post_io(self, all_res):
        if any_action_failed(all_res):
            raise IOError("Could not load data.")
        else:
            self._data = np.concatenate(all_res)
        return self
```

### Make all I/O in `async` methods even if there is nothing to parallelize
```python
class MyBatch(Batch):
    ...
    @inbatch_parallel(init='run_once', target='async')
    async def read_some_data(self, src, fmt='raw'):
        ...
...
some_pipeline
    .do_whatever_you_want()
    .read_some_data('/some/path')
    .do_something_else()
```
Init-function `run_once` runs the decorated method once (so no parallelism whatsoever).
Besides, the methods does not receive any additional arguments, only those passed to it directly.
However, an `action` defined as asynchronous will be waited for.
You may define your own `post`-method in order to check the result and process the exceptions if they arise.
