"""
Base classes for the benchmark

"""

import numpy as np

from abc import ABC, abstractmethod

from .metrics.crps import crps_quantile


class BaseTask(ABC):
    """
    Base class for a task

    Parameters:
    -----------
    seed: int
        Seed for the random number generator
    fixed_config: dict
        Fixed configuration for the task

    """

    def __init__(self, seed: int = None, fixed_config: dict = None):
        self.random = np.random.RandomState(seed)

        # Instantiate task parameters
        if fixed_config is not None:
            self.past_time = fixed_config["past_time"]
            self.future_time = fixed_config["future_time"]
            self.constraints = fixed_config["constraints"]
            self.background = fixed_config["background"]
            self.scenario = fixed_config["scenario"]
        else:
            self.random_instance()

    @abstractmethod
    def evaluate(self, samples: np.ndarray):
        """
        Evaluate success based on samples from the inferred distribution

        Parameters:
        -----------
        samples: np.ndarray, shape (n_samples, n_time, n_dim)
            Samples from the inferred distribution

        Returns:
        --------
        metric: float
            Metric value

        """
        pass

    @abstractmethod
    def random_instance(self):
        """
        Generate a random instance of the task and instantiate its data

        """
        pass


class UnivariateCRPSTask(BaseTask):
    """
    A base class for tasks that require forecasting a single series and that use CRPS for evaluation

    """

    def evaluate(self, samples):
        if len(samples.shape) == 3:
            samples = samples[:, :, 0]

        return crps_quantile(target=self.future_time, samples=samples)[0].sum()
