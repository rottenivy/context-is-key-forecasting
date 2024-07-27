import inspect
import hashlib
import logging
import os
import pandas as pd
import pickle

from pathlib import Path

from ..config import DEFAULT_N_SAMPLES, RESULT_CACHE_PATH


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

    """

    def __init__(self, method_callable, cache_path=RESULT_CACHE_PATH) -> None:
        self.logger = logging.getLogger("Result cache")
        self.method_callable = method_callable
        self.cache_dir = Path(cache_path) / self.get_method_path_name(method_callable)
        self.cache_path = self.cache_dir / "cache.pkl"

        if not self.cache_path.exists():
            self.logger.info("Cache file does not exist. Creating new cache.")
            self.cache = {}
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.logger.info(f"Loading cache from {self.cache_path}.")
            with self.cache_path.open("rb") as f:
                if os.path.getsize(self.cache_path) == 0:
                    self.cache = {}
                else:
                    self.cache = pickle.load(f)

    def get_method_path_name(self, obj):
        """
        Convert a method to a string that can be used as directory name in a path.

        """
        if obj.__class__.__name__ == "function":
            return f"{obj.__module__}.{obj.__qualname__}"
        else:
            return obj.__class__.__name__

    def get_method_source(self, obj):
        """
        Get the source code of a method as a string

        """
        if obj.__class__.__name__ == "function":
            return inspect.getsource(obj)
        else:
            return inspect.getsource(obj.__class__)

    def get_cache_key(self, task_instance, n_samples):
        """
        Get cache key by hashing the task instance (including all data and code)
        as well as the method callable's source code.

        Any change in the task instance or the method callable will result in recomputation.

        """

        def get_attr_hash(attr):
            if isinstance(attr, pd.DataFrame):
                # Convert DataFrame to string representation of its data
                return attr.to_csv(index=False).encode("utf-8")
            elif isinstance(attr, str):
                # Directly encode strings
                return attr.encode("utf-8")
            else:
                # Convert other attribute types to string and encode
                return str(attr).encode("utf-8")

        # Initialize the hash object
        hasher = hashlib.sha256()

        # Hash task source code
        hasher.update(inspect.getsource(task_instance.__class__).encode("utf-8"))

        # Hash method source code
        hasher.update(self.get_method_source(self.method_callable).encode("utf-8"))

        # Hash the number of samples
        hasher.update(str(n_samples).encode("utf-8"))

        # Iterate over all attributes of the object
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

        self.logger.info("Cache miss. Running inference.")
        samples = self.method_callable(task_instance, n_samples)

        # Update cache on disk
        self.logger.info("Updating cache.")
        self.cache[cache_key] = samples
        with self.cache_path.open("wb") as f:
            pickle.dump(self.cache, f)

        return samples
