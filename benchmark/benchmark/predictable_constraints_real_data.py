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
        assert max_constraints <= len(
            possible_constraints
        ), "max_constraints cannot be greater than the total available constraints"
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

        self._constraints = self.sampleConstraintsFromGroundTruth(future_series)

        # Instantiate the class variables
        self.past_time = history_series.to_frame()
        self.future_time = future_series.to_frame()
        self.constraints = self.verbalize_context_from_constraints(self._constraints)
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
        constraints_dict = {}
        sampled_constraint_types = self.random.choice(
            self.possible_constraints,
            self.random.randint(1, self.max_constraints + 1),
            replace=False,
        )
        for constraint_type in sampled_constraint_types:
            if constraint_type == "min":
                constraints_dict["min"] = future_series.min()
            elif constraint_type == "max":
                constraints_dict["max"] = future_series.max()
            elif constraint_type == "median":
                constraints_dict["median"] = future_series.median()
            elif constraint_type == "mode":
                constraints_dict["mode"] = future_series.mode().iloc[0]
                ## TODO: How to handle multimodal distributions?
            elif constraint_type == "mean":
                constraints_dict["mean"] = future_series.mean()

        return constraints_dict

    def verbalize_context_from_constraints(self, constraints):
        """
        Generate a synthetic context that describes the constraints.
        Parameters:
        -----------
        constraints: dict
            Dictionary of constraints to be satisfied by the forecast
        Returns:
        --------
        context: str
            Synthetic context that describes the constraints
        """
        context = "Assume that in the forecast, "
        for constraint, value in constraints.items():
            # context += f"{constraint}: {value}, "
            if constraint == "min":
                context += f"the minimum possible value is {value} and anything below that being bounded to {value}, "
            elif constraint == "max":
                context += f"the maximum possible value is {value} and anything above that being bounded to {value}, "
            elif constraint == "median":
                context += f"the median is {value}, "
            elif constraint == "mode":
                context += f"the mode is {value}, "
            elif constraint == "mean":
                context += f"the mean is {value}, "
            else:
                raise ValueError(f"Unknown constraint type: {constraint}")

        # remove trailing whitespace and comma
        context = context[:-2]

        # add period
        context += "."

        return context

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

        prop_satisfied_constraints = {}
        for constraint, value in self._constraints.items():
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


class BoundedPredConstraintsBasedOnPredQuantilesTask(
    OraclePredUnivariateConstraintsTask
):
    """
    A task where the data is modified to be bounded (upper or lower) in the prediction part, and the context specifies the bounds.
    This task is dataset-independent.
    """

    def __init__(
        self,
        possible_constraints=["min", "max"],
        max_constraints: int = 2,
        fixed_config: dict = None,
        seed: int = None,
    ):
        super().__init__(
            possible_constraints=possible_constraints,
            max_constraints=max_constraints,
            seed=seed,
            fixed_config=fixed_config,
        )

    def random_instance(self):
        """
        Create a random instance of the BoundedConstraintsTask task.
        Selects a random dataset, a random time series, and a random window.
        Calculates appropriate bounds from the window. Applies the bound constraints on just the prediction part, so you would need the context to perform a perfect forecast.
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

        # Apply bound constraints to the time series
        self._constraints = self.calculateConstraintsFromGroundTruth(future_series)

        # Instantiate the class variables
        self.past_time = history_series.to_frame()
        self.future_time = future_series.to_frame()
        self.constraints = self.verbalize_context_from_constraints(self._constraints)
        self.background = None
        self.scenario = None

    def calculateConstraintsFromGroundTruth(self, window):
        """
        Sample constraints from the ground truth.
        Parameters:
        -----------
        window: pd.Series
            Window
        Returns:
        --------
        constraints: dict
            Dictionary of constraints to be satisfied by the forecast
        """
        constraints_dict = {}
        sampled_constraint_types = self.random.choice(
            self.possible_constraints,
            self.random.randint(1, self.max_constraints + 1),
            replace=False,
        )
        # Define the quantiles you want to calculate (0th, 10th, ..., 100th)
        quantiles = list(range(0, 100, 10))
        # Define quantiles used for min and max bounding
        min_quantile_index = 3  # 30th Quantile
        max_quantile_index = 7  # 70th Quantile
        # Calculate the quantile values
        quantile_values = np.percentile(
            window, quantiles
        )  # TODO: Is a pandas series; may need .values
        # Apply constraints
        for constraint_type in sampled_constraint_types:
            if constraint_type == "min":
                window[window <= quantile_values[min_quantile_index]] = quantile_values[
                    min_quantile_index
                ]
                constraints_dict["min"] = quantile_values[min_quantile_index]
            elif constraint_type == "max":
                window[window >= quantile_values[max_quantile_index]] = quantile_values[
                    max_quantile_index
                ]
                constraints_dict["max"] = quantile_values[max_quantile_index]

        return constraints_dict


__TASKS__ = [
    OraclePredUnivariateConstraintsTask,
    BoundedPredConstraintsBasedOnPredQuantilesTask,
]
