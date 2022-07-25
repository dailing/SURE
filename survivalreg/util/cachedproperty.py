import inspect
import logging
import os
import pickle
from functools import partial
from hashlib import md5
from multiprocessing import RLock

from .logger import get_logger

logger = get_logger(__name__, print_level=logging.WARN)

_NOT_FOUND = object()
logger.debug(__name__)


class CacheDependency:
    """
    Used to monitor changed of a member function. if the code changes, all
    CachedPropertied will be invalidated and re-calculated
    """

    def __init__(self, func):
        self.func = func
        self.attrname = None
        self.__doc__ = func.__doc__
        self.code_hash = md5(inspect.getsource(
            func).encode("utf-8")).hexdigest()
        self.lock = RLock()
        self._cache_folder = None
        self._cache_file = None
        self.dependency = {}  # dependency is a name:func_hash map

    def report_dependency(self, instance):
        logger.debug('reporting')
        if '__cached_property_dep_stack' not in instance.__dict__:
            return
        if len(instance.__dict__['__cached_property_dep_stack']) > 0:
            instance.__dict__['__cached_property_dep_stack'][-1].dependency.update(self.dependency)
            instance.__dict__['__cached_property_dep_stack'][-1].dependency.update({
                self.func.__name__: self.code_hash
            })
            logger.debug(f'appending {self.func.__name__} {self.code_hash}')

    def __get__(self, instance, owner):
        self.report_dependency(instance)
        return partial(self.func, instance)


class CachedProperty:
    """
    Cache the result of a member-function, act like a normal
    cached_property, but it saves changes to disk. When re-run
    the script, the data cached to local disk will be loaded to
    reduce time.
    """

    def __init__(self, func):
        self.func = func
        self.attrname = None
        self.__doc__ = func.__doc__
        self.code_hash = md5(inspect.getsource(
            func).encode("utf-8")).hexdigest()
        self.lock = RLock()
        self._cache_folder = None
        self._cache_file = None
        self.dependency = {}  # dependency is a name:func_hash map

    def __set_name__(self, owner, name):
        if self.attrname is None:
            self.attrname = name
        elif name != self.attrname:
            raise TypeError(
                "Cannot assign the same cached_property to two different names "
                f"({self.attrname!r} and {name!r})."
            )

    def search_dep(self, klass, name):
        if name in klass.__dict__:
            return klass.__dict__[name]
        for base in klass.__bases__:
            res = self.search_dep(base, name)
            if res is not None:
                return res
        return None

    def check_dependency(self, instance):
        logger.debug(self.dependency)
        for k, v in self.dependency.items():
            inst = self.search_dep(instance.__class__, k)
            if inst is None or inst.code_hash != v:
                return False
        return True

    def report_dependency(self, instance):
        logger.debug('reporting')
        if '__cached_property_dep_stack' not in instance.__dict__:
            return
        if len(instance.__dict__['__cached_property_dep_stack']) > 0:
            instance.__dict__['__cached_property_dep_stack'][-1].dependency.update(self.dependency)
            instance.__dict__['__cached_property_dep_stack'][-1].dependency.update({
                self.func.__name__: self.code_hash
            })
            logger.debug(f'appending {self.func.__name__} {self.code_hash}')

    def handle_cache_not_find(self, instance, owner=None):
        if not os.path.exists(self._cache_file):
            if os.path.exists(self._cache_folder):
                # cache folder while file not exist, clear cache folder
                for f in os.listdir(self._cache_folder):
                    if os.path.isfile(os.path.join(self._cache_folder, f)):
                        os.remove(os.path.join(
                            self._cache_folder, f))
            else:
                os.makedirs(self._cache_folder, exist_ok=True)
            # append self to dependency stack
            if '__cached_property_dep_stack' not in instance.__dict__:
                instance.__dict__['__cached_property_dep_stack'] = []
            instance.__dict__['__cached_property_dep_stack'].append(self)
            logger.debug(f'before, {instance.__dict__["__cached_property_dep_stack"]}')
            val = self.func(instance)  # after this, the dependency map should be established
            logger.debug(f'after, {instance.__dict__["__cached_property_dep_stack"]}')
            # update parent dependency if there is one
            assert instance.__dict__['__cached_property_dep_stack'].pop() is self
            self.report_dependency(instance)
            pickle.dump((val, self.dependency), open(self._cache_file, "wb"))
            return val
        val, self.dependency = pickle.load(open(self._cache_file, 'rb'))
        if self.check_dependency(instance):
            return val
        # dependency changed, delete cache file and re_calculate
        os.remove(self._cache_file)
        logger.debug('cache_dependency broke')
        return self.handle_cache_not_find(instance, owner)

    def __get__(self, instance, owner=None):
        logger.debug(f'enter {self.func.__name__}')
        func = instance

        if instance is None:
            return self
        if self.attrname is None:
            raise TypeError(
                "Cannot use cached_property instance without calling __set_name__ on it.")
        try:
            if '__cached_objs__' not in instance.__dict__:
                instance.__dict__['__cached_objs__'] = {}
            cache = instance.__dict__['__cached_objs__']
        # not all objects have __dict__ (e.g. class defines slots)
        except AttributeError:
            msg = (
                f"No '__dict__' attribute on {type(instance).__name__!r} "
                f"instance to cache {self.attrname!r} property."
            )
            raise TypeError(msg) from None
        val = cache.get(self.attrname, _NOT_FOUND)
        if val is _NOT_FOUND:
            with self.lock:
                if self._cache_file is None:
                    self._cache_folder = os.path.join(
                        './.output/',
                        inspect.getfile(instance.__class__).split('/')[-1],
                        func.__class__.__name__, self.func.__name__)
                    self._cache_file = os.path.join(self._cache_folder, self.code_hash)
                # check if another thread filled cache while we awaited lock
                val = cache.get(self.attrname, _NOT_FOUND)
                if val is _NOT_FOUND:
                    val = self.handle_cache_not_find(instance, owner)
                    try:
                        cache[self.attrname] = val
                        # return val
                    except TypeError:
                        msg = (
                            f"The '__dict__' attribute on {type(instance).__name__!r} instance "
                            f"does not support item assignment for caching {self.attrname!r} property."
                        )
                        raise TypeError(msg) from None
        self.report_dependency(instance)
        logger.debug(f'exit {self.func.__name__}')
        return val
