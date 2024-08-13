import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import traceback

from collections import defaultdict
from pathlib import Path

from . import ALL_TASKS
from .config import DEFAULT_N_SAMPLES
from .utils.cache import ResultCache


logger = logging.getLogger("Evaluation")


def plot_forecast_univariate(task, samples, path):
    """
    Plot the first variable of a forecast.

    Parameters:
    -----------
    task: a BaseTask
        The task associated with the forecast
    samples: np.array
        The forecast of shape [samples, time dimension, number of variables]
    path: Pathlike
        Directory in which to save the figure

    """
    samples = samples[:, :, 0]

    future_timesteps = task.future_time.index
    if isinstance(future_timesteps, pd.PeriodIndex):
        future_timesteps = future_timesteps.to_timestamp()

    fig = task.plot()
    ax = fig.gca()

    for zorder, quant, color, label in [
        [1, 0.05, (0.75, 0.75, 1), "5%-95%"],
        [2, 0.10, (0.25, 0.25, 1), "10%-90%"],
        [3, 0.25, (0, 0, 0.75), "25%-75%"],
    ]:
        ax.fill_between(
            future_timesteps,
            np.quantile(samples, quant, axis=0).astype(float),
            np.quantile(samples, 1 - quant, axis=0).astype(float),
            facecolor=color,
            interpolate=True,
            label=label,
            zorder=zorder,
        )

    ax.plot(
        future_timesteps,
        np.quantile(samples, 0.5, axis=0),
        color=(0.5, 0.5, 0.5),
        linewidth=3,
        label="50%",
        zorder=4,
    )

    ax.set_ylim(
        np.min([np.min(task.past_time), np.min(task.future_time), np.min(samples)]),
        np.max([np.max(task.past_time), np.max(task.future_time), np.max(samples)]),
    )

    handles, labels = ax.get_legend_handles_labels()
    order = [2, 0, 1, 3, 4, 5, 6]
    ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order])

    ax.set_title(task.name)

    fig.savefig(path / "forecast.pdf")
    fig.savefig(path / "forecast.png", bbox_inches="tight")
    plt.close(fig)


def save_context(task, path):
    """
    Save the context of a task to a file for future reference.

    """
    with open(path / "context", "w") as f:
        f.write(
            f"""
Background:
{task.background}

Constraints:
{task.constraints}

Scenario:
{task.scenario}
"""
        )


def evaluate_all_tasks(
    method_callable,
    seeds=5,
    n_samples=DEFAULT_N_SAMPLES,
    output_folder=None,
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
    output_folder: None or Pathlike
        Directory to output results (figures and task context). If not None,
        results will not be saved.
    use_cache: bool
        If True, use cached results when available. Otherwise, re-run the evaluation.
    cache_name: str, optional
        Name of the method to use as cache key. If not provided, will be inferred
        automatically.

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

    if output_folder:
        logger.info(f"Saving outputs to {output_folder}")
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
    else:
        logger.info("No output folder provided. Results will not be saved.")

    if use_cache:
        method_callable = ResultCache(method_callable, method_name=cache_name)

    results = defaultdict(list)
    for task_cls in ALL_TASKS:
        for seed in range(1, seeds + 1):

            # Instantiate the task
            task = task_cls(seed=seed)

            if output_folder:
                task_folder = output_folder / task.name
                seed_folder = task_folder / f"{seed}"
                seed_folder.mkdir(parents=True, exist_ok=True)

            try:
                logger.info(
                    f"Method {method_callable} - Task {task.name} - Seed {seed}"
                )
                samples = method_callable(task_instance=task, n_samples=n_samples)
                results[task_cls.__name__].append(
                    {
                        "seed": seed,
                        "score": task.evaluate(samples),
                    }
                )

                if output_folder:
                    # Save forecast plots
                    plot_forecast_univariate(
                        task=task, samples=samples, path=seed_folder
                    )

                    # Save context
                    save_context(task=task, path=seed_folder)

            except Exception as e:
                logger.error(
                    f"Error evaluating task {task_cls.__name__} - Seed {seed}: {e}"
                )
                # Save the error to the seed folder
                if output_folder:
                    with open(seed_folder / "error", "w") as f:
                        f.write(str(e))
                        f.write("\n")
                        f.write(traceback.format_exc())

    return results
