import logging

logging.basicConfig(level=logging.INFO)
import numpy as np
import pandas as pd

from benchmark.baselines.lag_llama import lag_llama
from benchmark.evaluation import evaluate_all_tasks


def random_baseline(task_instance, n_samples=50):
    """
    A baseline is just some callable that receives a task instance and returns a prediction.

    """
    # This is only valid for a univariate forecast, of shape [samples, time dimension, 1]
    return np.random.rand(n_samples, task_instance.future_time.shape[0], 1)


def oracle_baseline(task_instance, n_samples=50):
    """
    A perfect baseline that looks at the future and returns it in multiple copies with a tiny jitter (like perfect samples)

    """
    # This is only valid for a univariate forecast, of shape [samples, time dimension, 1]
    return (
        np.tile(task_instance.future_time.values[None, :, None], (n_samples, 1, 1))
        + np.random.rand(n_samples, task_instance.future_time.shape[0], 1) * 1e-6
    )


if __name__ == "__main__":
    n_samples = 25

    random_results = evaluate_all_tasks(random_baseline, n_samples=n_samples)
    oracle_results = evaluate_all_tasks(oracle_baseline, n_samples=n_samples)
    lag_llama_results = evaluate_all_tasks(lag_llama, n_samples=n_samples)

    results = pd.DataFrame(
        {
            "Task": [task for task in random_results],
            "Random": [
                np.mean([res["score"] for res in random_results[task]])
                for task in random_results
            ],
            "Oracle": [
                np.mean([res["score"] for res in oracle_results[task]])
                for task in oracle_results
            ],
            "Lag-Llama": [
                np.mean([res["score"] for res in lag_llama_results[task]])
                for task in lag_llama_results
            ],
        }
    )
    print(results)
