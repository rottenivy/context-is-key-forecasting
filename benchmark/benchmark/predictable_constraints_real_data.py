import numpy as np

from .base import UnivariateCRPSTask
from .metrics.constraints import ListConstraint, MaxConstraint, MinConstraint
from tactis.gluon.dataset import get_dataset
from gluonts.dataset.util import to_pandas

from .utils import get_random_window_univar

from .window_selection import (
    intersection_over_union_is_low,
    quartile_intersection_over_union_is_low,
    median_absolute_deviation_intersection_is_low,
    is_baseline_prediction_poor,
)


class OraclePredUnivariateConstraintsTask(UnivariateCRPSTask):
    """
    Task that creates constraints from the ground truth, makes synthetic context that
    describes these constraints and evaluates the forecast based on these constraints.
    Time series: real, electricity_hourly but dataset agnostic
    Context: synthetic, by looking at the ground truth forecast
    Parameters:
    -----------
    possible_constraints: list
        List of possible constraints to be used.
        Default is ["min", "max"]
        Possible values are ["min", "max"]
    max_constraints: int
        Maximum number of constraints to be used. Default is 2.
    baselines: list
        List of baseline models to be used. Default is None.
    baseline_evaluation_criteria: str
        Criteria to evaluate the baseline models. Default is "all".
    window_selection: str
        Method to select the window. Default is "robust_iou".
    fixed_config: dict
        Fixed configuration for the task
    seed: int
        Seed for the random number generator.
    """

    _context_sources = UnivariateCRPSTask._context_sources + ["c_f"]
    _skills = UnivariateCRPSTask._skills + ["instruction following"]

    def __init__(
        self,
        possible_constraints=["min", "max"],
        max_constraints: int = 2,
        baselines=None,
        baseline_evaluation_criteria="all",
        window_selection="robust_iou",
        fixed_config: dict = None,
        seed: int = None,
    ):
        assert max_constraints <= len(
            possible_constraints
        ), "max_constraints cannot be greater than the total available constraints"
        self.possible_constraints = possible_constraints
        self.max_constraints = max_constraints
        self.seed = seed
        self.baselines = baselines
        self.baseline_evaluation_criteria = baseline_evaluation_criteria
        self.window_selection = window_selection

        super().__init__(seed=seed, fixed_config=fixed_config)

    def random_instance(self):
        """
        Create a random instance of the OraclePredUnivariateConstraintsTask task.
        Selects a random dataset, a random time series, and a random window.
        Samples constraints from the ground truth forecast.
        Instantiates the class variables.
        """
        history_series, future_series = self.find_interesting_window(
            how=self.window_selection
        )

        # Instantiate the class variables
        self.past_time = history_series.to_frame()
        self.future_time = future_series.to_frame()
        self.constraints = self.verbalize_context_from_constraints(
            self.constraints_dict
        )
        self.background = None
        self.scenario = None

    def find_interesting_window(self, how="iou"):
        """
        Selects a window from one of the gluonts datasets.
        The window is selected according to the performance of a baseline model.
        If the baseline model performs poorly, the window is considered interesting.
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

        window_is_interesting = False
        while not window_is_interesting:
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
            self.past_time = history_series.to_frame()
            self.future_time = future_series.to_frame()

            self.constraints_dict, self.metric_constraint = (
                self.sampleConstraintsFromGroundTruth(future_series)
            )

            # Check if the constraints are interesting
            if how == "baseline":
                period = self.seasonal_period
                window_is_interesting = is_baseline_prediction_poor(
                    history_series, future_series, self.constraints_dict, period
                )
            elif how == "iou":
                window_is_interesting = intersection_over_union_is_low(
                    history_series, future_series
                )

            elif how == "robust_iou":
                window_is_interesting = quartile_intersection_over_union_is_low(
                    history_series, future_series
                )

            elif how == "robust_mad_iou":
                window_is_interesting = median_absolute_deviation_intersection_is_low(
                    history_series, future_series
                )

        return history_series, future_series

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
        constraints_objects = []
        sampled_constraints = self.random.choice(
            self.possible_constraints,
            self.random.randint(1, self.max_constraints + 1),
            replace=False,
        )
        for constraint_type in sampled_constraints:
            if constraint_type == "min":
                constraints_dict["min"] = future_series.min()
                constraints_objects.append(MinConstraint(future_series.min()))
            elif constraint_type == "max":
                constraints_dict["max"] = future_series.max()
                constraints_objects.append(MaxConstraint(future_series.max()))

        if len(constraints_objects) >= 2:
            metric_constraint = ListConstraint(constraints_objects)
        else:
            metric_constraint = constraints_objects[0]
        return constraints_dict, metric_constraint

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
        parts = ["Suppose that in the forecast,"]

        for constraint, value in constraints.items():
            if constraint == "min":
                parts.append(f"the values are bounded below by {value:.2f}")
            elif constraint == "max":
                parts.append(f"the values are bounded above by {value:.2f}")

        context = ", ".join(parts) + "."
        return context


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

        # ROI metrics parameter
        constraints_dict, self.metric_constraint = (
            self.calculateConstraintsFromGroundTruth(future_series)
        )

        # Instantiate the class variables
        self.past_time = history_series.to_frame()
        self.future_time = future_series.to_frame()
        self.constraints = self.verbalize_context_from_constraints(constraints_dict)
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
        constraints_objects = []
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
                constraints_objects.append(
                    MinConstraint(quantile_values[min_quantile_index])
                )
            elif constraint_type == "max":
                window[window >= quantile_values[max_quantile_index]] = quantile_values[
                    max_quantile_index
                ]
                constraints_dict["max"] = quantile_values[max_quantile_index]
                constraints_objects.append(
                    MaxConstraint(quantile_values[max_quantile_index])
                )

        if len(constraints_objects) >= 2:
            metric_constraint = ListConstraint(constraints_objects)
        else:
            metric_constraint = constraints_objects[0]
        return constraints_dict, metric_constraint


__TASKS__ = [
    OraclePredUnivariateConstraintsTask,
    BoundedPredConstraintsBasedOnPredQuantilesTask,
]
