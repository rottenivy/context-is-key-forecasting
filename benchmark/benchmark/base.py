"""
Base classes for the benchmark

"""

import numpy as np
import pandas as pd
from typing import Optional
import statsmodels.tsa.tsatools

from abc import ABC, abstractmethod

from .metrics.roi_metric import region_of_interest_constraint_metric
from .utils.plot import plot_task


class ContextFlags(dict):
    """
    Custom dictionnary to store context flags

    """

    allowed_keys = ["c_i", "c_h", "c_f", "c_cov", "c_causal"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize all flags to False
        for key in self.allowed_keys:
            self[key] = False

    def __setitem__(self, key, value):
        # Check if flag is valid
        if key not in self.allowed_keys:
            raise KeyError(
                f"Invalid context type: {key}. Allowed types: {self.allowed_keys}."
            )
        # Check if value if boolean
        if not isinstance(value, bool):
            raise ValueError(
                f"Context flag must be a boolean, not a {value.__class__.__name__}."
            )
        super().__setitem__(key, value)


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

    def __init__(self, seed: int = None, fixed_config: Optional[dict] = None):
        self.random = np.random.RandomState(seed)

        # Instantiate task parameters
        if fixed_config is not None:
            self.past_time = fixed_config["past_time"]
            self.future_time = fixed_config["future_time"]
            self.constraints = fixed_config["constraints"]
            self.background = fixed_config["background"]
            self.scenario = fixed_config["scenario"]
        else:
            self.constraints = None
            self.background = None
            self.scenario = None
            self.random_instance()

        # Context flags
        self.context_flags = ContextFlags()

        config_errors = self.verify_config()
        if config_errors:
            raise RuntimeError(
                f"Incorrect config for {self.__class__.__name__}: {config_errors}"
            )

    @property
    def name(self) -> str:
        """
        Give the name of the task, for reporting purpose

        Returns:
        --------
        name: str
            The name of the task
        """
        return self.__class__.__name__

    @property
    def seasonal_period(self) -> int:
        """
        This returns the period which should be used by statistical models for this task.
        If negative, this means that the data either has no period, or the history is shorter than the period.
        """
        # By default, uses the frequency of the data to guess the period.
        # This should be overriden for tasks for which this guess fails.
        freq = self.past_time.index.freq
        if not freq:
            freq = pd.infer_freq(self.past_time.index)
        period = statsmodels.tsa.tsatools.freq_to_period(freq)
        return period

    def verify_config(self) -> list[str]:
        """
        Check whether the task satisfy the correct format for its parameters.

        Returns:
        --------
        errors: list[str]
            A list of textual descriptions of all errors in the format
        """
        errors = []
        # A few tests to make sure that all tasks use a compatible format for the parameters
        # Note: Only the parameters which are used elsewhere are current tested
        if not isinstance(self.past_time, pd.DataFrame):
            errors.append(
                f"past_time is not a pd.DataFrame, but a {self.past_time.__class__.__name__}"
            )
        if not isinstance(self.future_time, pd.DataFrame):
            errors.append(
                f"future_time is not a pd.DataFrame, but a {self.future_time.__class__.__name__}"
            )
        return errors

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

    def plot(self):
        """
        Plot the task

        Returns:
        --------
        fig: matplotlib.figure.Figure
            The figure containing the plot

        """
        return plot_task(self)


class UnivariateCRPSTask(BaseTask):
    """
    A base class for tasks that require forecasting a single series and that use CRPS for evaluation
    We use the last column of `future_time` as the ground truth for evaluation
    """

    def __init__(self, seed: int = None, fixed_config: Optional[dict] = None):
        # Instantiate task parameters
        if fixed_config is not None:
            self.region_of_interest = fixed_config["region_of_interest"]
            self.roi_weight = fixed_config["roi_weight"]
            self.metric_constraints = fixed_config["metric_constraints"]
            self.metric_constraints_tolerance = fixed_config[
                "metric_constraints_tolerance"
            ]
        else:
            # These will be filled during by random_instance(), called in BaseTask.__init__
            self.region_of_interest = None
            self.roi_weight = 0.5
            self.metric_constraints = None
            self.metric_constraints_tolerance = 0.05

        super().__init__(seed=seed, fixed_config=fixed_config)

    def verify_config(self) -> list[str]:
        """
        Check whether the task satisfy the correct format for its parameters.

        Returns:
        --------
        errors: list[str]
            A list of textual descriptions of all errors in the format
        """
        errors = super().verify_config()
        if self.roi_weight < 0 or self.roi_weight > 1:
            errors.append(f"roi weight ({self.roi_weight}) is not between 0 and 1")
        return errors

    def evaluate(self, samples):
        if len(samples.shape) == 3:
            samples = samples[:, :, 0]

        # This is the dual of pd.Series.to_frame(), compatible with any series name
        only_column = self.future_time.columns[-1]
        target = self.future_time[only_column]
        return region_of_interest_constraint_metric(
            target=target,
            forecast=samples,
            region_of_interest=self.region_of_interest,
            roi_weight=self.roi_weight,
            constraints=self.metric_constraints,
            tolerance_percentage=self.metric_constraints_tolerance,
        )
