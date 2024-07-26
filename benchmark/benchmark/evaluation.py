import inspect
import hashlib
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle

from collections import defaultdict
from pathlib import Path

from . import ALL_TASKS
from .config import DEFAULT_N_SAMPLES, RESULT_CACHE_PATH


logger = logging.getLogger("Evaluation")


def plot_forecast_univariate(task, samples, filename):
    """
    Plot the first variable of a forecast.

    Parameters:
    -----------
    task: a BaseTask
        The task associated with the forecast
    samples: np.array
        The forecast of shape [samples, time dimension, number of variables]
    filename: Pathlike
        Where to save the figure
    """
    samples = samples[:, :, 0]
    past_timesteps = task.past_time.index
    past_values = task.past_time.to_numpy()[:, 0]
    future_timesteps = task.future_time.index
    future_values = task.future_time.to_numpy()[:, 0]

    # The fill_between method is only ok with pd.DatetimeIndex
    if isinstance(past_timesteps, pd.PeriodIndex):
        past_timesteps = past_timesteps.to_timestamp()
    if isinstance(future_timesteps, pd.PeriodIndex):
        future_timesteps = future_timesteps.to_timestamp()

    timesteps = np.concatenate([past_timesteps, future_timesteps])
    values = np.concatenate([past_values, future_values])

    plt.figure()

    for zorder, quant, color, label in [
        [1, 0.05, (0.75, 0.75, 1), "5%-95%"],
        [2, 0.10, (0.25, 0.25, 1), "10%-90%"],
        [3, 0.25, (0, 0, 0.75), "25%-75%"],
    ]:
        plt.fill_between(
            future_timesteps,
            np.quantile(samples, quant, axis=0).astype(float),
            np.quantile(samples, 1 - quant, axis=0).astype(float),
            facecolor=color,
            interpolate=True,
            label=label,
            zorder=zorder,
        )

    plt.plot(
        future_timesteps,
        np.quantile(samples, 0.5, axis=0),
        color=(0.5, 0.5, 0.5),
        linewidth=3,
        label="50%",
        zorder=4,
    )

    plt.plot(
        timesteps, values, color=(0, 0, 0), linewidth=2, zorder=5, label="ground truth"
    )

    handles, labels = plt.gca().get_legend_handles_labels()
    order = [4, 0, 1, 2, 3]
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])

    plt.title(task.name)

    plt.savefig(filename)


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


def evaluate_all_tasks(
    method_callable,
    seeds=5,
    n_samples=DEFAULT_N_SAMPLES,
    plot_folder=None,
    use_cache=True,
):
    """
    Evaluates a method on all tasks for a number of seeds and samples

    Parameters:
    -----------
    method_callable: callable
        A callable that receives a task instance and returns a prediction samples.
        The callable should expect the following kwargs: task_instance, n_samples
    seeds: int
        Number of seeds to evaluate the method
    n_samples: int
        Number of samples to generate for each prediction
    plot_folder: None or Pathlike
        If not None, save figure for each forecast in this folde
    use_cache: bool
        If True, use cached results when available. Otherwise, re-run the evaluation.

    Returns:
    --------
    results: dict
        A dictionary with the results of the evaluation.
        Keys are task names and values are lists of dictionaries
        with metrics and relevant information.

    """
    logger.info(
        f"Evaluating method {method_callable} with {seeds} seeds and {n_samples} samples on {len(ALL_TASKS)} tasks."
    )

    if plot_folder:
        logger.info(f"Saving plots to {plot_folder}")
        plot_folder = Path(plot_folder)
        plot_folder.mkdir(parents=True, exist_ok=True)

    if use_cache:
        method_callable = ResultCache(method_callable)

    results = defaultdict(list)
    for task_cls in ALL_TASKS:
        logger.info(f"Task {task_cls.__name__}.")
        for seed in range(1, seeds + 1):
            task = task_cls(seed=seed)
            samples = method_callable(task_instance=task, n_samples=n_samples)

            results[task_cls.__name__].append(
                {
                    "seed": seed,
                    "score": task.evaluate(samples),
                }
            )

            if plot_folder:
                task_folder = plot_folder / task.name
                task_folder.mkdir(parents=True, exist_ok=True)
                task_filename = task_folder / f"{seed}.pdf"
                plot_forecast_univariate(
                    task=task, samples=samples, filename=task_filename
                )

    return results
