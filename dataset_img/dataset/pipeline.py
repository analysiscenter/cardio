""" Pipeline classes """
import sys
import traceback
import concurrent.futures as cf
import threading
import asyncio
import queue as q
try:
    import tensorflow as tf
except ImportError:
    pass
from .batch_base import BaseBatch


class Pipeline:
    """ Pipeline """
    def __init__(self, dataset):
        self.dataset = dataset
        self._action_list = []
        self._prefetch_queue = None
        self._batch_queue = None
        self._service_executor = None
        self._executor = None
        self._batch_generator = None
        self._tf_session = None


    @staticmethod
    def _is_batch_method(name, cls=None):
        cls = BaseBatch if cls is None else cls
        if hasattr(cls, name) and callable(getattr(cls, name)):
            return True
        else:
            return any(Pipeline._is_batch_method(name, subcls) for subcls in cls.__subclasses__())

    def __getattr__(self, name, *args, **kwargs):
        """ Check if an unknown attr is an action from the batch class """
        if self._is_batch_method(name):
            self._action_list.append({'name': name})
            return self._append_action
        else:
            raise AttributeError("%s not found in class %s" % (name, self.__class__.__name__))

    def _append_action(self, *args, **kwargs):
        """ Add new action to the log of future actions """
        self._action_list[-1].update({'args': args, 'kwargs': kwargs})
        return self

    def __getstate__(self):
        return {'dataset': self.dataset, 'action_list': self._action_list}

    def __setstate__(self, state):
        self.dataset = state['dataset']
        self._action_list = state['action_list']

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


    def _exec_all_actions(self, batch, new_loop=False):
        if new_loop:
            asyncio.set_event_loop(asyncio.new_event_loop())

        joined_sets = None
        for _action in self._action_list:
            if _action['name'] == 'join':
                joined_sets = _action['datasets']
            else:
                action_method, _ = self._get_action_method(batch, _action['name'])

                if joined_sets is not None:
                    joined_data = []
                    for jset in joined_sets:   # pylint: disable=not-an-iterable
                        joined_data.append(jset.create_batch(batch.index))
                    _action_args = tuple(joined_data) + _action['args']
                    joined_sets = None
                else:
                    _action_args = _action['args']

                batch = action_method(*_action_args, **_action['kwargs'])

                if 'tf_queue' in _action:
                    self._put_batch_into_tf_queue(batch, _action)
        return batch

    def join(self, *datasets):
        """ Join other datasets """
        self._action_list.append({'name': 'join', 'datasets': datasets})
        return self

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
        for batch in gen_batch:
            future = self._executor.submit(self._exec_all_actions, batch, True)
            self._prefetch_queue.put(future, block=True)
        self._prefetch_queue.put(None, block=True)

    def _run_batches_from_queue(self):
        while True:
            future = self._prefetch_queue.get(block=True)
            if future is None:
                self._prefetch_queue.task_done()
                self._batch_queue.put(None)
                break
            else:
                try:
                    batch = future.result()
                except Exception:   # pylint: disable=broad-except
                    print("Exception in a thread:", future.exception())
                    _, _, exc_traceback = sys.exc_info()
                    traceback.print_tb(exc_traceback, limit=1, file=sys.stdout)
                self._batch_queue.put(batch)
                self._prefetch_queue.task_done()
        return None

    def run(self, batch_size, shuffle=False, n_epochs=1, drop_last=False, prefetch=0, *args, **kwargs):
        """ Execute all lazy actions for each batch in the dataset """
        batch_generator = self.gen_batch(batch_size, shuffle, n_epochs, drop_last, prefetch, *args, **kwargs)
        for _ in batch_generator:
            pass
        return self

    def create_batch(self, batch_index, *args, **kwargs):
        """ Create a new batch by given indices and execute all previous lazy actions """
        batch = self.dataset.create_batch(batch_index, *args, **kwargs)
        batch_res = self._exec_all_actions(batch)
        return batch_res

    def reset_iter(self):
        """ Clear all iteration metadata in order to start iterating from scratch """
        if self._prefetch_queue is not None:
            self._prefetch_queue.put(None, block=True)
        if self._batch_queue is not None:
            self._batch_queue.put(None, block=True)
        if self._executor is not None:
            self._executor.shutdown()
            self._executor = None
        if self._service_executor is not None:
            self._service_executor.shutdown()
            self._service_executor = None
        self._prefetch_queue = None
        self._batch_queue = None
        self._batch_generator = None
        self.dataset.reset_iter()

    def gen_batch(self, batch_size, shuffle=False, n_epochs=1, drop_last=False, prefetch=0, *args, **kwargs):
        """ Generate batches """
        target = kwargs.pop('target', 'threads')
        self._tf_session = kwargs.pop('tf_session', None)

        batch_generator = self.dataset.gen_batch(batch_size, shuffle, n_epochs, drop_last, *args, **kwargs)

        if prefetch > 0:
            # pool cannot have more than 63 workers
            prefetch = min(prefetch, 60)

            if target == 'threads':
                self._executor = cf.ThreadPoolExecutor(max_workers=prefetch + 1)
            elif target == 'mpc':
                self._executor = cf.ProcessPoolExecutor(max_workers=prefetch + 1)   # pylint: disable=redefined-variable-type
            else:
                raise ValueError("target should be one of ['threads', 'mpc']")

            self._prefetch_queue = q.Queue(maxsize=prefetch + 1)
            self._batch_queue = q.Queue()
            self._service_executor = cf.ThreadPoolExecutor(max_workers=2)
            self._service_executor.submit(self._put_batches_into_queue, batch_generator)
            self._service_executor.submit(self._run_batches_from_queue)
            is_end = False
            while not is_end:
                batch_res = self._batch_queue.get(block=True)
                if batch_res is not None:
                    self._batch_queue.task_done()
                    yield batch_res
                else:
                    is_end = True
        else:
            for batch in batch_generator:
                yield self._exec_all_actions(batch)

        self.reset_iter()
        return self

    def next_batch(self, batch_size, shuffle=False, n_epochs=1, drop_last=False, prefetch=0, *args, **kwargs):
        """ Get the next batch and execute all previous lazy actions """
        if prefetch > 0:
            if self._batch_generator is None:
                self._batch_generator = self.gen_batch(batch_size, shuffle, n_epochs,
                                                       drop_last, prefetch, *args, **kwargs)
            batch_res = next(self._batch_generator)
        else:
            # target is not used here, but people tend to forget removing it when set prefetch to 0
            _ = kwargs.pop('target', 'threads')
            batch_index = self.index.next_batch(batch_size, shuffle, n_epochs, drop_last, *args, **kwargs)
            batch_res = self.create_batch(batch_index, *args, **kwargs)
        return batch_res
