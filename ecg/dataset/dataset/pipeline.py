""" Pipeline classes """
import traceback
import concurrent.futures as cf
import threading
#import multiprocessing as mpc
import asyncio
import queue as q
import numpy as np
try:
    import tensorflow as tf
except ImportError:
    pass

from .batch_base import BaseBatch
from .base import Baseset
from .exceptions import SkipBatchException

PIPELINE_ID = '#_pipeline'
JOIN_ID = '#_join'
MERGE_ID = '#_merge'
REBATCH_ID = '#_rebatch'


def mult_option(a, b):
    """ Multiply even if any arg is None """
    return a * b if a is not None and b is not None else a if a is not None else b

class Pipeline:
    """ Pipeline """
    def __init__(self, dataset=None, pipeline=None, proba=None, repeat=None):
        if pipeline is None:
            self.dataset = dataset
            self._action_list = []
        else:
            self.dataset = pipeline.dataset
            self._action_list = pipeline._action_list[:]  # pylint: disable=protected-access
            if self.num_actions == 1:
                if proba is not None:
                    if self.get_last_action_repeat() is None:
                        self._action_list[-1]['proba'] = mult_option(proba, self.get_last_action_proba())
                elif repeat is not None:
                    if self.get_last_action_proba() is None:
                        self._action_list[-1]['repeat'] = mult_option(repeat, self.get_last_action_repeat())

        self.variables = dict() #mpc.Manager().dict()
        self._tf_session = None

        self._stop_flag = False
        self._executor = None
        self._service_executor = None
        self._prefetch_count = None
        self._prefetch_queue = None
        self._batch_queue = None
        self._batch_generator = None

        self._rest_batch = None
        self._lazy_run = None
        self.reset_iter()


    def __enter__(self):
        """ Create a context and return an empty pipeline non-bound to any dataset """
        return type(self)()

    def __exit__(self, exc_type, exc_value, trback):
        pass

    @classmethod
    def from_pipeline(cls, pipeline, proba=None, repeat=None):
        """ Create a pipeline from another pipeline """
        if proba is None:
            if repeat is None:
                new_p = cls(pipeline=pipeline)
            else:
                if pipeline.num_actions == 1 and pipeline.get_last_action_proba() is None:
                    new_p = cls(pipeline=pipeline, repeat=repeat)
                else:
                    new_p = cls()
                    new_p.append_pipeline(pipeline, repeat=repeat)
        else:
            if pipeline.num_actions == 1 and pipeline.get_last_action_repeat() is None:
                new_p = cls(pipeline=pipeline, proba=proba)
            else:
                new_p = cls()
                new_p.append_pipeline(pipeline, proba=proba)
        return new_p

    @classmethod
    def concat(cls, pipe1, pipe2):
        """ Create a new pipeline concatenating two given pipelines """
        # pylint: disable=protected-access
        if pipe1.dataset != pipe2.dataset and pipe1.dataset is not None and pipe2.dataset is not None:
            raise ValueError("Cannot add pipelines with different datasets")

        new_p1 = cls.from_pipeline(pipe1)
        new_p2 = cls.from_pipeline(pipe2)
        new_p1._action_list += new_p2._action_list[:]
        new_p1.dataset = pipe1.dataset or pipe2.dataset
        return new_p1

    def get_last_action_proba(self):
        """ Return a probability of the last action """
        return self._action_list[-1]['proba']

    def get_last_action_repeat(self):
        """ Return a repeat count of the last action """
        return self._action_list[-1]['repeat']

    def __add__(self, other):
        if not isinstance(other, Pipeline):
            raise TypeError("Both operands should be Pipelines")
        if other.num_actions > 0:
            return self.concat(self, other)
        else:
            return self

    def __matmul__(self, other):
        if self.num_actions == 0:
            raise ValueError("Cannot add probability to an empty pipeline")
        if not isinstance(other, float) and other not in [0, 1]:
            raise TypeError("Probability should be float or 0 or 1")
        other = float(other) if int(other) != 1 else None
        return self.from_pipeline(self, proba=other)

    def __mul__(self, other):
        if other < 0:
            raise ValueError("Repeat count cannot be negative. Use as pipeline * positive_number")
        elif isinstance(other, float):
            raise ValueError("Repeat count cannot be float. Use as pipeline * integer")
        elif isinstance(other, int):
            new_p = self.from_pipeline(self, repeat=other)
        return new_p

    def __lshift__(self, other):
        if not isinstance(other, Baseset):
            raise TypeError("Pipelines might take only Datasets. Use as pipeline << dataset")
        new_p = self.from_pipeline(self)
        new_p.dataset = other
        return new_p

    @staticmethod
    def _is_batch_method(name, cls=None):
        cls = BaseBatch if cls is None else cls
        if hasattr(cls, name) and callable(getattr(cls, name)):
            return True
        else:
            return any(Pipeline._is_batch_method(name, subcls) for subcls in cls.__subclasses__())

    def __getattr__(self, name):
        """ Check if an unknown attr is an action from some batch class """
        if self._is_batch_method(name):
            self._action_list.append({'name': name})
            return self.append_action
        else:
            raise AttributeError("%s not found in class %s" % (name, self.__class__.__name__))

    @property
    def num_actions(self):
        """ Return index length """
        return len(self._action_list)

    def append_action(self, *args, **kwargs):
        """ Add new action to the log of future actions """
        self._action_list[-1].update({'args': args, 'kwargs': kwargs, 'proba': None, 'repeat': None})
        new_p = self.from_pipeline(self)
        self._action_list = self._action_list[:-1]
        return new_p

    def append_pipeline(self, pipeline, proba=None, repeat=None):
        """ Add a nested pipeline to the log of future actions """
        self._action_list.append({'name': PIPELINE_ID, 'pipeline': pipeline,
                                  'proba': proba, 'repeat': repeat})

    def __getstate__(self):
        return {'dataset': self.dataset, 'action_list': self._action_list, 'variables': self.variables}

    def __setstate__(self, state):
        self.dataset = state['dataset']
        self._action_list = state['action_list']
        self.variables = state['variables']

    @property
    def index(self):
        """ Return index of the source dataset """
        return self.dataset.index

    @property
    def indices(self):
        """ Return the sequence of indices of the source dataset """
        return self.index.indices

    def __len__(self):
        """ Return index length """
        return len(self.index)

    def get_variable(self, name):
        """ Return a variable value """
        res = self.variables.get(name, None)
        return res

    def init_variable(self, name, value):
        """ Create a variable if not exists """
        if name not in self.variables:
            self.variables[name] = value

    def set_variable(self, name, value):
        """ Set a variable value """
        self.variables[name] = value

    @staticmethod
    def _get_action_method(batch, name):
        if hasattr(batch, name):
            attr = getattr(batch, name)
            if attr.__self__ == batch:
                # action decorator with arguments
                # attr is bounded to the batch
                action_method = attr
                action_attr = attr
            else:
                # action decorator wihout arguments
                action_method = attr
                action_attr = attr.__self__

            if callable(action_attr):
                if hasattr(action_attr, 'action'):
                    action_spec = getattr(action_attr, 'action')
                else:
                    raise ValueError("Method %s is not marked with @action decorator" % name)
            else:
                raise TypeError("%s is not a method" % name)
        else:
            raise AttributeError("Method '%s' has not been found in the %s class" % (name, type(batch).__name__))
        return action_method, action_spec

    def _exec_one_action(self, batch, action, args, kwargs):
        if self._needs_exec(action):
            for _ in range(action['repeat'] or 1):
                action_method, _ = self._get_action_method(batch, action['name'])
                batch.pipeline = self
                batch = action_method(*args, **kwargs)
        return batch

    def _exec_nested_pipeline(self, batch, action):
        if self._needs_exec(action):
            for _ in range(action['repeat'] or 1):
                batch = self._exec_all_actions(batch, action['pipeline']._action_list)  # pylint: disable=protected-access
        return batch

    def _exec_all_actions(self, batch, action_list=None):
        # pylint: disable=too-many-branches
        join_batches = None
        action_list = action_list or self._action_list
        for _action in action_list:
            if _action['name'] in [JOIN_ID, MERGE_ID]:
                join_batches = []
                for pipe in _action['pipelines']:   # pylint: disable=not-an-iterable
                    if _action['mode'] == 'i':
                        jbatch = pipe.create_batch(batch.index)
                    elif _action['mode'] == 'n':
                        jbatch = pipe.next_batch()
                    join_batches.append(jbatch)

                if _action['name'] == MERGE_ID:
                    if _action['merge_fn'] is None:
                        batch, _ = batch.merge([batch] + join_batches)
                    else:
                        batch, _ = _action['merge_fn']([batch] + join_batches)
                    join_batches = None
            elif _action['name'] == REBATCH_ID:
                if self._rest_batch is None:
                    cur_len = 0
                    batches = []
                else:
                    cur_len = len(self._rest_batch)
                    batches = [self._rest_batch]
                    self._rest_batch = None
                while cur_len < _action['batch_size']:
                    new_batch = _action['pipeline'].next_batch(*self._lazy_run[0], **self._lazy_run[1])
                    batches.append(new_batch)
                    cur_len += len(new_batch)
                if _action['merge_fn'] is None:
                    batch, self._rest_batch = batches[0].merge(batches, batch_size=_action['batch_size'])
                else:
                    batch, self._rest_batch = _action['merge_fn'](batches, batch_size=_action['batch_size'])
            elif _action['name'] == PIPELINE_ID:
                batch = self._exec_nested_pipeline(batch, _action)
            else:
                if join_batches is None:
                    _action_args = _action['args']
                else:
                    _action_args = tuple(join_batches) + _action['args']
                    join_batches = None

                batch = self._exec_one_action(batch, _action, _action_args, _action['kwargs'])

                if 'tf_queue' in _action:
                    self._put_batch_into_tf_queue(batch, _action)
        return batch

    def _needs_exec(self, action):
        if action['proba'] is None:
            return True
        else:
            return np.random.binomial(1, action['proba']) == 1

    def _exec(self, batch, new_loop=False):
        if new_loop:
            asyncio.set_event_loop(asyncio.new_event_loop())
        return self._exec_all_actions(batch)

    def join(self, *pipelines):
        """ Join pipelines
        Args:
            one or several pipelines
            mode:
                - 'i' - by index, i.e. get from each pipeline batches with the same indices
                - 'r' - by lazy run, i.e. get the next batch from each pipeline.
                        In this mode batches could have different size and non-matching indices.
                        For this to work pipelines must end with run(..., lazy=True).
        """
        self._action_list.append({'name': JOIN_ID, 'pipelines': pipelines, 'mode': 'i'})
        return self.append_action()

    def merge(self, *pipelines, merge_fn=None):
        """ Merge pipelines """
        self._action_list.append({'name': MERGE_ID, 'pipelines': pipelines,    # pylint: disable=protected-access
                                  'mode': 'n', 'merge_fn': merge_fn})
        return self.append_action()

    def rebatch(self, batch_size, merge_fn=None):
        """ Set the output batch size """
        new_p = type(self)()
        new_p._action_list.append({'name': REBATCH_ID, 'batch_size': batch_size,  # pylint: disable=protected-access
                                   'pipeline': self, 'merge_fn': merge_fn})
        return new_p.append_action()

    def put_into_tf_queue(self, session=None, queue=None, get_tensor=None):
        """ Insert a tensorflow queue after the action"""
        if len(self._action_list) > 0:
            action = dict()
            action['tf_session'] = session
            action['tf_queue'] = queue
            action['get_tensor'] = get_tensor
            action['tf_enqueue_op'] = None
            action['tf_placeholders'] = None
            action['tf_action_lock'] = threading.Lock()
            self._action_list[-1].update(action)
        else:
            raise RuntimeError('tf_queue should be precedeed by at least one action')
        return self

    @staticmethod
    def _get_dtypes(tensors=None, action=None):
        if tensors:
            return [tensor.dtype for tensor in tensors]
        else:
            return [placeholder.dtype for placeholder in action['tf_placeholders']]

    def _create_tf_queue(self, tensors, action):
        if action['tf_session'] is None:
            action['tf_session'] = self._tf_session
        if action['tf_session'] is None:
            raise ValueError("Tensorflow session cannot be None")
        maxsize = 1 if self._prefetch_queue is None else self._prefetch_queue.maxsize
        with action['tf_session'].graph.as_default():
            action['tf_queue'] = tf.FIFOQueue(capacity=maxsize, dtypes=self._get_dtypes(tensors, action))

    @staticmethod
    def _get_tf_placeholders(tensors, action):
        tensors = tensors if isinstance(tensors, tuple) else tuple([tensors])
        with action['tf_session'].graph.as_default():
            placeholders = [tf.placeholder(dtype=tensor.dtype) for tensor in tensors]
        return placeholders

    @staticmethod
    def _get_tensor(batch, action):
        if action['get_tensor'] is None:
            return batch.data
        else:
            return action['get_tensor'](batch)

    def _put_batch_into_tf_queue(self, batch, action):
        tensors = self._get_tensor(batch, action)
        tensors = tensors if isinstance(tensors, tuple) else tuple([tensors])
        if action['tf_queue'] is None:
            with action['tf_action_lock']:
                if action['tf_queue'] is None:
                    self._create_tf_queue(tensors, action)
        if action['tf_enqueue_op'] is None:
            with action['tf_action_lock']:
                if action['tf_enqueue_op'] is None:
                    action['tf_placeholders'] = self._get_tf_placeholders(tensors, action)
                    action['tf_enqueue_op'] = action['tf_queue'].enqueue(action['tf_placeholders'])
        action['tf_session'].run(action['tf_enqueue_op'], feed_dict=dict(zip(action['tf_placeholders'], tensors)))


    def _put_batches_into_queue(self, gen_batch):
        while not self._stop_flag:
            self._prefetch_count.put(1, block=True)
            try:
                batch = next(gen_batch)
            except StopIteration:
                break
            else:
                future = self._executor.submit(self._exec, batch, new_loop=True)
                self._prefetch_queue.put(future, block=True)
        self._prefetch_queue.put(None, block=True)

    def _run_batches_from_queue(self):
        skip_batch = False
        while not self._stop_flag:
            future = self._prefetch_queue.get(block=True)
            if future is None:
                self._prefetch_queue.task_done()
                self._batch_queue.put(None)
                break
            else:
                try:
                    batch = future.result()
                except SkipBatchException:
                    skip_batch = True
                except Exception:   # pylint: disable=broad-except
                    exc = future.exception()
                    print("Exception in a thread:", exc)
                    traceback.print_tb(exc.__traceback__)
                finally:
                    if not skip_batch:
                        self._batch_queue.put(batch, block=True)
                        skip_batch = False
                    self._prefetch_queue.task_done()
        return None

    def create_batch(self, batch_index, *args, **kwargs):
        """ Create a new batch by given indices and execute all previous lazy actions """
        batch = self.dataset.create_batch(batch_index, *args, **kwargs)
        batch_res = self._exec(batch)
        return batch_res


    def reset_iter(self):
        """ Clear all iteration metadata in order to start iterating from scratch """
        def _clear_queue(queue):
            if queue is not None:
                while not queue.empty():
                    queue.get(block=True)
                    queue.task_done()

        def _stop_executor(executor):
            if executor is not None:
                executor.shutdown()

        self._stop_flag = True

        _clear_queue(self._prefetch_queue)
        _clear_queue(self._batch_queue)
        _clear_queue(self._prefetch_count)

        _stop_executor(self._executor)
        _stop_executor(self._service_executor)

        self._executor = None
        self._service_executor = None
        self._prefetch_count = None
        self._prefetch_queue = None
        self._batch_queue = None
        self._batch_generator = None
        self.dataset.reset_iter()
        self.variables = dict() #mpc.Manager().dict()


    def gen_batch(self, batch_size, shuffle=True, n_epochs=1, drop_last=False, prefetch=0, *args, **kwargs):
        """ Generate batches """
        self.reset_iter()

        target = kwargs.pop('target', 'threads')
        self._tf_session = kwargs.pop('tf_session', None)

        batch_generator = self.dataset.gen_batch(batch_size, shuffle, n_epochs, drop_last, *args, **kwargs)

        if prefetch > 0:
            # pool cannot have more than 63 workers
            prefetch = min(prefetch, 62)

            if target in ['threads', 't']:
                self._executor = cf.ThreadPoolExecutor(max_workers=prefetch + 1)
            elif target in ['mpc', 'm']:
                self._executor = cf.ProcessPoolExecutor(max_workers=prefetch + 1)   # pylint: disable=redefined-variable-type
            else:
                raise ValueError("target should be one of ['threads', 'mpc']")

            self._stop_flag = False
            self._prefetch_count = q.Queue(maxsize=prefetch + 1)
            self._prefetch_queue = q.Queue(maxsize=prefetch)
            self._batch_queue = q.Queue(maxsize=1)
            self._service_executor = cf.ThreadPoolExecutor(max_workers=2)
            self._service_executor.submit(self._put_batches_into_queue, batch_generator)
            self._service_executor.submit(self._run_batches_from_queue)

            while not self._stop_flag:
                batch_res = self._batch_queue.get(block=True)
                self._batch_queue.task_done()
                if batch_res is not None:
                    yield batch_res
                    self._prefetch_count.get(block=True)
                    self._prefetch_count.task_done()
                else:
                    self._stop_flag = True
        else:
            for batch in batch_generator:
                try:
                    batch_res = self._exec(batch)
                except SkipBatchException:
                    pass
                else:
                    yield batch_res

    def next_batch(self, *args, **kwargs):
        """ Get the next batch and execute all previous lazy actions
        next_batch(self, batch_size, shuffle=True, n_epochs=1, drop_last=False, prefetch=0, *args, **kwargs):
        """
        if len(args) == 0 and len(kwargs) == 0:
            if self._lazy_run is None:
                raise RuntimeError("next_batch without arguments requires a lazy run at the end of the pipeline")
            batch_res = self.next_batch(*self._lazy_run[0], **self._lazy_run[1])
        elif kwargs.get('prefetch', 0) > 0:
            if self._batch_generator is None:
                self._lazy_run = args, kwargs
                self._batch_generator = self.gen_batch(*args, **kwargs)
            batch_res = next(self._batch_generator)
        else:
            self._lazy_run = args, kwargs
            _kwargs = kwargs.copy()
            # target is not used here, but people tend to forget removing it when set prefetch to 0
            _ = _kwargs.pop('target', 'threads')
            # prefetch could be 0
            _ = _kwargs.pop('prefetch', 0)
            batch_res = None
            while batch_res is None:
                batch_index = self.index.next_batch(*args, **_kwargs)
                try:
                    batch_res = self.create_batch(batch_index, **_kwargs)
                except SkipBatchException:
                    pass
        return batch_res

    def run(self, *args, **kwargs):
        """ Execute all lazy actions for each batch in the dataset
        run(self, batch_size, shuffle=True, n_epochs=1, drop_last=False, prefetch=0, *args, **kwargs):
        """
        if kwargs.pop('lazy', False):
            self._lazy_run = args, kwargs
        else:
            if len(args) == 0 and len(kwargs) == 0:
                args, kwargs = self._lazy_run
            for _ in self.gen_batch(*args, **kwargs):
                pass
        return self
