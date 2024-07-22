import scipy
import numpy as np
import pandas as pd
from abc import abstractmethod

from .base import BaseTask
from tactis.gluon.dataset import get_dataset
from gluonts.dataset.util import to_pandas

from .utils import get_random_window_univar


class OraclePredUnivariateConstraintsTask(BaseTask):
    """
    Task that creates constraints from the ground truth, makes synthetic context that
    describes these constraints and evaluates the forecast based on these constraints.
    Time series: real, electricity_hourly but dataset agnostic
    Context: synthetic, by looking at the ground truth forecast
    Parameters:
    -----------
    constraints: dict
        Dictionary of constraints to be satisfied by the forecast. If None, constraints
        are sampled from the ground truth forecast.
    possible_constraints: list
        List of possible constraints to be used.
        Default is ["min", "max", "median", "mode", "mean"]
    max_constraints: int
        Maximum number of constraints to be used. Default is 2.
    fixed_config: dict
        Fixed configuration for the task

    """

    def __init__(
        self,
        possible_constraints=["min", "max", "median", "mode", "mean"],
        max_constraints: int = 2,
        fixed_config: dict = None,
        seed: int = None,
    ):
        self.possible_constraints = possible_constraints
        self.max_constraints = max_constraints

        super().__init__(seed=seed, fixed_config=fixed_config)

    def random_instance(self):
        """
        Create a random instance of the OraclePredUnivariateConstraintsTask task.
        Selects a random dataset, a random time series, and a random window.
        Samples constraints from the ground truth forecast.
        Instantiates the class variables.
        """
        datasets = ["electricity_hourly"]

        # Select a random dataset
        dataset_name = self.random.choice(datasets)
        dataset = get_dataset(dataset_name, regenerate=False)

        assert len(dataset.train) == len(
            dataset.test
        ), "Train and test sets must contain the same number of time series"

        # Get the dataset metadata
        metadata = dataset.metadata

        # Select a random time series
        ts_index = self.random.choice(len(dataset.train))
        full_series = to_pandas(list(dataset.test)[ts_index])

        # Select a random window
        window = get_random_window_univar(
            full_series,
            prediction_length=metadata.prediction_length,
            history_factor=self.random.randint(2, 5),
            random=self.random,
        )

        # Extract the history and future series
        history_series = window.iloc[: -metadata.prediction_length]
        future_series = window.iloc[-metadata.prediction_length :]

        # Generate constraints from the ground truth
        self.constraints = self.sampleConstraintsFromGroundTruth(future_series)

        # Instantiate the class variables
        self.past_time = history_series.to_frame()
        self.future_time = future_series.to_frame()
        self.background = None
        self.scenario = None

    def sampleConstraintsFromGroundTruth(self, future_series):
        """
        Sample constraints from the ground truth.
        Parameters:
        -----------
        future_series: pd.Series
            Ground truth forecast
        Returns:
        --------
        constraints: dict
            Dictionary of constraints to be satisfied by the forecast
        """
        constraints = {}
        constraints = self.random.choice(
            self.possible_constraints,
            self.random.randint(1, self.max_constraints + 1),
            replace=False,
        )
        for constraint in constraints:
            if constraint == "min":
                constraints["min"] = future_series.min()
            elif constraint == "max":
                constraints["max"] = future_series.max()
            elif constraint == "median":
                constraints["median"] = future_series.median()
            elif constraint == "mode":
                constraints["mode"] = future_series.mode().iloc[0]  ## TODO: generalize
            elif constraint == "mean":
                constraints["mean"] = future_series.mean()

        return constraints

    def evaluate(self, samples):
        """
        Since we don't know the ground truth distribution, the evaluation should
        be done by comparing the forecast with the constraints
        The score is the proportion of samples that satisfy each constraint,
        averaged over all constraints.
        As a side-effect, sets the attribute prop_satisfied_constraints to the
        proportion of samples that satisfy each constraint.
        Parameters:
        -----------
        samples: np.ndarray
            Samples from the inferred distribution
        Returns:
        --------
        prop_satisfied_constraint: float
            Proportion of samples that satisfy the constraints
        """
        if len(samples.shape) == 3:
            samples = samples[:, :, 0]  # (n_samples, n_time)

        prop_satisfied_constraints = {
            constraint: False for constraint in self.constraints.keys()
        }
        for constraint, value in self.constraints.items():
            if constraint == "min":
                good_samples = np.all(samples >= value, axis=1)
            elif constraint == "max":
                good_samples = np.all(samples <= value, axis=1)
            elif constraint == "median":
                good_samples = np.abs(np.median(samples, axis=1) - value) < 1e-6
            elif constraint == "mode":
                good_samples = (
                    np.abs(scipy.stats.mode(samples, axis=1).mode.flatten() - value)
                    < 1e-6
                )
            elif constraint == "mean":
                good_samples = np.abs(np.mean(samples, axis=1) - value) < 1e-6

            prop_satisfied_constraint = np.mean(good_samples)
            prop_satisfied_constraints[constraint] = prop_satisfied_constraint

        prop_satisfied_constraint = np.mean(
            np.array(list(prop_satisfied_constraints.values()))
        )
        prop_satisfied_constraints["satisfaction_rate"] = prop_satisfied_constraint
        self.prop_satisfied_constraints = prop_satisfied_constraints

        return prop_satisfied_constraint


__TASKS__ = [OraclePredUnivariateConstraintsTask]
