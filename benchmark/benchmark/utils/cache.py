import builtins
import inspect
import hashlib
import logging
import pandas as pd

from diskcache import Cache
from pathlib import Path


from ..baselines.base import Baseline
from ..config import DEFAULT_N_SAMPLES, RESULT_CACHE_PATH


def get_method_cache_name(method_callable):
    """
    Convert a method to a string that can be used as a cache path.

    Parameters:
    -----------
    method_callable: callable
        A callable that receives a task instance and a number of samples and returns
        a prediction samples. The callable should expect the following kwargs:
        task_instance, n_samples

    """
    if method_callable.__class__.__name__ == "function":
        return f"{method_callable.__module__}.{method_callable.__qualname__}"
    elif isinstance(method_callable, Baseline):
        return method_callable.cache_name
    else:
        raise ValueError("Unable to infer cache name for method.")


def get_source(obj) -> str:
    """
    Get the source code of an object as a string.

    Parameters:
    -----------
    obj: callable
        Any object

    Notes:
    ------
    If the object is a class, the source code of all parent classes is also included.

    """
    if obj.__class__.__name__ == "function":
        return inspect.getsource(obj)
    else:
        # Get all parent classes that are not built-in
        parent_classes = [
            c
            for c in inspect.getmro(obj.__class__)
            if c.__module__ != builtins.__name__
        ]
        # Concatenate source code of all parent classes and the method's class
        return "".join(inspect.getsource(c) for c in parent_classes + [obj.__class__])


class CacheMissError(Exception):
    pass


class ResultCache:
    """
    A cache to avoid recomputing costly results. Basically acts as a wrapper
    around a method callable that caches the results.

    Parameters:
    -----------
    method_callable: callable
        A callable that receives a task instance and a number of samples and returns
        a prediction samples. The callable should expect the following kwargs:
        task_instance, n_samples
    method_name: str, optional
        Name of the method callable. Used to create a directory in the cache.
        Must be provided if method_callable is an instance of a class.
    cache_path: str, optional
        Path to the cache directory. Default is taken from RESULT_CACHE.
    raise_on_miss: bool, optional
        Whether to raise a CacheMissError if the cache is not found. Default is False.

    """

    def __init__(
        self, method_callable, method_name=None, cache_path=RESULT_CACHE_PATH, raise_on_miss=False
    ) -> None:
        self.logger = logging.getLogger("Result cache")
        self.method_callable = method_callable
        self.cache_dir = Path(cache_path) / (
            get_method_cache_name(method_callable)
            if method_name is None
            else method_name
        )
        self.cache = Cache(self.cache_dir)
        self.raise_on_miss = raise_on_miss

    def get_cache_key(self, task_instance, n_samples):
        """
        Get cache key by hashing the task instance (including all data and code)
        as well as the method callable's source code.

        Any change in the task instance or the method callable will result in recomputation.

        """

        def get_attr_hash(attr):
            if isinstance(attr, pd.DataFrame) or isinstance(attr, pd.Series):
                # Convert DataFrame to string representation of its data
                return attr.to_csv().encode("utf-8")
            elif isinstance(attr, str):
                # Directly encode strings
                return attr.encode("utf-8")
            else:
                # Convert other attribute types to string and encode
                return str(attr).encode("utf-8")

        # Initialize the hash object
        hasher = hashlib.sha256()

        # Hash task source code
        hasher.update(get_source(task_instance.__class__).encode("utf-8"))

        # Hash method source code
        hasher.update(get_source(self.method_callable).encode("utf-8"))

        # Hash the number of samples
        hasher.update(str(n_samples).encode("utf-8"))

        # Iterate over all attributes of the task instance
        for attr, value in task_instance.__dict__.items():
            # Hash the attribute name
            hasher.update(attr.encode("utf-8"))

            # Hash the attribute value
            hasher.update(get_attr_hash(value))

        # Return the hex digest of the hash
        return hasher.hexdigest()

    def __call__(self, task_instance, n_samples=DEFAULT_N_SAMPLES):
        self.logger.info("Attempting to load from cache.")
        cache_key = self.get_cache_key(task_instance, n_samples)

        if cache_key in self.cache:
            self.logger.info("Cache hit.")
            return self.cache[cache_key]

        if self.raise_on_miss:
            raise CacheMissError()
        
        self.logger.info("Cache miss. Running inference.")
        samples = self.method_callable(task_instance, n_samples)

        # Update cache on disk
        self.logger.info("Updating cache.")
        self.cache[cache_key] = samples

        return samples

    def __str__(self) -> str:
        return str(self.method_callable)
