"""
Trivial baselines

"""

import numpy as np


def random_baseline(task_instance, n_samples=50):
    """
    A baseline is just some callable that receives a task instance and returns a prediction.

    """
    # This is valid for forecasts of shape [samples, time dimension, number of variables]
    return np.random.rand(
        n_samples,
        task_instance.future_time.shape[0],
        task_instance.future_time.shape[1],
    )


def oracle_baseline(task_instance, n_samples=50):
    """
    A perfect baseline that looks at the future and returns it in multiple copies with a tiny jitter (like perfect samples)

    """
    # This is valid for forecasts of shape [samples, time dimension, number of variables]
    target = (
        task_instance.future_time.to_numpy()
    )  # [time dimension, number of variables]
    return (
        target[None, :, :]
        + np.random.rand(n_samples, target.shape[0], target.shape[1]) * 1e-6
    )
