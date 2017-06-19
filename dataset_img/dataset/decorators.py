""" Pipeline decorators """
import os
import inspect
import traceback
import threading
import concurrent.futures as cf
import asyncio


def _cpu_count():
    cpu_count = 0
    try:
        cpu_count = len(os.sched_getaffinity(0))
    except AttributeError:
        cpu_count = os.cpu_count()
    return cpu_count


def make_method_key(module_name, method_name):
    """ Build a full method name 'module.method' """
    return module_name + '.' + method_name

def get_method_key(method):
    """ Retrieve a full method name from a callable """
    return make_method_key(inspect.getmodule(method).__name__, method.__qualname__)

def infer_method_key(action_method, model_name):
    """ Infer a full model method name from a given action method and a model name """
    return make_method_key(inspect.getmodule(action_method).__name__,
                           action_method.__qualname__.rsplit('.', 1)[0] + '.' + model_name)


class ModelDecorator:
    """ Decorator for model definition methods in Batch classes """
    models = dict()

    def __init__(self, mode='global', engine='tf'):
        self.mode = mode
        self.engine = engine
        self.method = None

    @staticmethod
    def get_model(method):
        """ Return a model specification for a given model method """
        full_method_name = get_method_key(method)
        return ModelDecorator.models[full_method_name]

    @staticmethod
    def add_model(method, model_spec):
        """ Add a model specification into the model directory """
        full_method_name = get_method_key(method)
        ModelDecorator.models.update({full_method_name: model_spec})

    def run_model(self):
        """ Run the model method and save the model into the model directory """
        model_spec = self.method()
        self.add_model(self.method, model_spec)

    def __call__(self, method):
        self.method = method
        if self.mode == 'global':
            self.run_model()

        def method_call(*args, **kwargs):
            """ Do nothing if the method is called explicitly """
            _ = args, kwargs
            return None
        method_call.model_method = self.method
        return method_call

def model(*args, **kwargs):
    """ Decorator for model methods

    Usage:
        @model()
        def some_model():
            ...
            return my_model
    """
    return ModelDecorator(*args, **kwargs)


class ActionDecorator:
    """ Decorator for Batch class actions """
    def __init__(self, *args, **kwargs):
        self.method = None
        self.model_name = None
        self.model_method = None
        self.action_self = None
        self.singleton = False
        self.singleton_lock = None

        if len(args) == 1 and callable(args[0]):
            # @action without arguments
            self.add_action(args[0])
        else:
            # @action with arguments
            self.singleton = kwargs.pop('singleton', False)
            self.model_name = kwargs.pop('model', None)

    def add_action(self, method):
        """ Add an action specification into an action method """
        self.method = method
        full_method_name = get_method_key(self.method)
        if self.model_name is None:
            full_model_name = None
        else:
            full_model_name = infer_method_key(self.method, self.model_name)

        self.singleton_lock = None if not self.singleton else threading.Lock()
        action_spec = dict(method=self.method, full_method_name=full_method_name,
                           singleton=self.singleton, singleton_lock=self.singleton_lock,
                           has_model=self.model_name is not None,
                           model_name=self.model_name, full_model_name=full_model_name)
        self.action = action_spec

    def _action_with_model(self):
        """ Return a callable for a decorator call """
        def _call_with_model(action_self, *args, **kwargs):
            """ Call an action with a model specification """
            if hasattr(action_self, self.model_name):
                try:
                    self.model_method = getattr(action_self, self.model_name).model_method
                except AttributeError:
                    raise ValueError("The method '%s' is not marked with @model" % self.model_name)
            else:
                raise ValueError("There is no such method '%s'" % self.model_name)

            model_spec = ModelDecorator.get_model(self.model_method)
            return self.call_action(action_self, model_spec, *args, **kwargs)
        _call_with_model.action = self.action
        return _call_with_model

    def _action_wo_model(self):
        """ Return a callable for a decorator call """
        def _call_action(action_self, *args, **kwargs):
            return self.call_action(action_self, *args, **kwargs)
        _call_action.action = self.action
        return _call_action

    def call_action(self, action_self, *args, **kwargs):
        """ Call an action """
        if self.singleton_lock is not None:
            self.singleton_lock.acquire(blocking=True)

        res = self.method(action_self, *args, **kwargs)

        if self.singleton_lock is not None:
            self.singleton_lock.release()

        return res

    def __call__(self, *args, **kwargs):
        if self.method is None:
            # @action with arguments
            self.add_action(args[0])
            if self.model_name is not None:
                # return a function that will be called when a decorated method is called
                return self._action_with_model()
            else:
                return self._action_wo_model()
        else:
            # @action without arguments
            return self.call_action(self.action_self, *args, **kwargs)

    def __get__(self, instance, owner):
        _ = owner
        self.action_self = instance
        return self.__call__


def action(*args, **kwargs):
    """ Decorator for action methods in Batch classes

    Usage:
        @action
        def some_action(self, arg1, arg2):
            ...

        @action(model='some_model')
        def train_model(self, model, another_arg):
            ...
    """
    return ActionDecorator(*args, **kwargs)


def any_action_failed(results):
    """ Return True if some parallelized invocations threw exceptions """
    return any(isinstance(res, Exception) for res in results)

def inbatch_parallel(init, post=None, target='threads', **dec_kwargs):
    """ Make in-batch parallel decorator """
    if target not in ['nogil', 'threads', 'mpc', 'async']:
        raise ValueError("target should be one of 'nogil', threads', 'mpc', 'async'")

    def inbatch_parallel_decorator(method):
        """ Return a decorator which run a method in parallel """
        def _check_functions(self):
            """ Check dcorator's `init` and `post` parameters """
            if init is None:
                raise ValueError("init cannot be None")
            else:
                try:
                    init_fn = getattr(self, init)
                except AttributeError:
                    raise ValueError("init should refer to a method or property of the class", type(self).__name__,
                                     "returning the list of arguments")
            if post is not None:
                try:
                    post_fn = getattr(self, post)
                except AttributeError:
                    raise ValueError("post should refer to a method of the class", type(self).__name__)
                if not callable(post_fn):
                    raise ValueError("post should refer to a method of the class", type(self).__name__)
            else:
                post_fn = None
            return init_fn, post_fn

        def _call_init_fn(init_fn, args, kwargs):
            if callable(init_fn):
                return init_fn(*args, **kwargs)
            else:
                return init_fn

        def _call_post_fn(self, post_fn, futures, args, kwargs):
            all_results = []
            for future in futures:
                try:
                    result = future.result()
                except Exception as exce:  # pylint: disable=broad-except
                    result = exce
                finally:
                    all_results += [result]

            if post_fn is None:
                if any_action_failed(all_results):
                    all_errors = [error for error in all_results if isinstance(error, Exception)]
                    print(all_errors)
                    traceback.print_tb(all_errors[0].__traceback__)
                return self
            else:
                return post_fn(all_results, *args, **kwargs)

        def _make_args(init_args, args, kwargs):
            """ Make args, kwargs tuple """
            if isinstance(init_args, tuple) and len(init_args) == 2:
                margs, mkwargs = init_args
            elif isinstance(init_args, dict):
                margs = list()
                mkwargs = init_args
            else:
                margs = init_args
                mkwargs = dict()
            margs = margs if isinstance(margs, (list, tuple)) else [margs]
            if len(args) > 0:
                margs = list(margs) + list(args)
            if len(kwargs) > 0:
                mkwargs.update(kwargs)
            return margs, mkwargs

        def wrap_with_threads(self, args, kwargs, nogil=False):
            """ Run a method in parallel """
            init_fn, post_fn = _check_functions(self)

            n_workers = kwargs.pop('n_workers', _cpu_count() * 4)
            with cf.ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = []
                if nogil:
                    nogil_fn = method(self, *args, **kwargs)
                full_kwargs = {**kwargs, **dec_kwargs}
                for arg in _call_init_fn(init_fn, args, full_kwargs):
                    margs, mkwargs = _make_args(arg, args, kwargs)
                    if nogil:
                        one_ft = executor.submit(nogil_fn, *margs, **mkwargs)
                    else:
                        one_ft = executor.submit(method, self, *margs, **mkwargs)
                    futures.append(one_ft)

                timeout = kwargs.get('timeout', None)
                cf.wait(futures, timeout=timeout, return_when=cf.ALL_COMPLETED)

            return _call_post_fn(self, post_fn, futures, args, full_kwargs)

        def wrap_with_mpc(self, args, kwargs):
            """ Run a method in parallel """
            init_fn, post_fn = _check_functions(self)

            n_workers = kwargs.pop('n_workers', _cpu_count() * 4)
            with cf.ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = []
                mpc_func = method(self, *args, **kwargs)
                full_kwargs = {**kwargs, **dec_kwargs}
                for arg in _call_init_fn(init_fn, args, full_kwargs):
                    margs, mkwargs = _make_args(arg, args, kwargs)
                    one_ft = executor.submit(mpc_func, *margs, **mkwargs)
                    futures.append(one_ft)

                timeout = kwargs.pop('timeout', None)
                cf.wait(futures, timeout=timeout, return_when=cf.ALL_COMPLETED)

            return _call_post_fn(self, post_fn, futures, args, full_kwargs)

        def wrap_with_async(self, args, kwargs):
            """ Run a method in parallel with async / await """
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                # this is a new thread where there is no loop
                loop = kwargs.get('loop', None)
                asyncio.set_event_loop(loop)
            else:
                loop = kwargs.get('loop', loop)

            init_fn, post_fn = _check_functions(self)

            futures = []
            full_kwargs = {**kwargs, **dec_kwargs}
            for arg in _call_init_fn(init_fn, args, full_kwargs):
                margs, mkwargs = _make_args(arg, args, kwargs)
                futures.append(asyncio.ensure_future(method(self, *margs, **mkwargs)))

            loop.run_until_complete(asyncio.gather(*futures, loop=loop, return_exceptions=True))

            return _call_post_fn(self, post_fn, futures, args, full_kwargs)

        def wrapped_method(self, *args, **kwargs):
            """ Wrap a method in a required parallel engine """
            if target == 'threads':
                return wrap_with_threads(self, args, kwargs)
            elif target == 'nogil':
                return wrap_with_threads(self, args, kwargs, nogil=True)
            elif target == 'mpc':
                return wrap_with_mpc(self, args, kwargs)
            elif target == 'async':
                return wrap_with_async(self, args, kwargs)
            raise ValueError('Wrong parallelization target:', target)
        return wrapped_method
    return inbatch_parallel_decorator


parallel = inbatch_parallel  # pylint: disable=invalid-name
