import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from collections import defaultdict
from pathlib import Path

from . import ALL_TASKS
from .config import DEFAULT_N_SAMPLES
from .utils.cache import ResultCache


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


def evaluate_all_tasks(
    method_callable,
    seeds=5,
    n_samples=DEFAULT_N_SAMPLES,
    plot_folder=None,
    use_cache=True,
    cache_name=None,
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
    cache_name: str, optional
        Name of the method to use as cache key. If the method_callable is an instance
        of a class, the cache key must be provided. Otherwise, the cache key is the
        method_callable's function name.

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
        method_callable = ResultCache(method_callable, method_name=cache_name)

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
