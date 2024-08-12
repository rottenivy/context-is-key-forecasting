"""
Base classes for the benchmark

"""

import numpy as np
import pandas as pd
from typing import Optional
import statsmodels.tsa.tsatools

from abc import ABC, abstractmethod

from .metrics.roi_metric import region_of_interest_constraint_metric


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
            self.roi_constraints = fixed_config["roi_constraints"]
            self.roi_tolerance = fixed_config["roi_tolerance"]
        else:
            # These will be filled during by random_instance(), called in BaseTask.__init__
            self.region_of_interest = None
            self.roi_weight = 0
            self.roi_constraints = None
            self.roi_tolerance = 0.05

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
        if self.region_of_interest is None and self.roi_weight != 0:
            errors.append("region_of_interest is not set, yet roi_weight is not 0")
        if self.region_of_interest is not None and self.roi_weight == 0:
            errors.append("region_of_interest is set, yet roi_weight is 0")
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
            constraints=self.roi_constraints,
            tolerance_percentage=self.roi_tolerance,
        )
