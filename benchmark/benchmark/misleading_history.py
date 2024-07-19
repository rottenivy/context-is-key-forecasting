"""
Tasks that involve time series whose past series contains misleading information
that can only be detected by understanding the contextual information provided
with the data.

"""

import numpy as np

from tactis.gluon.dataset import get_dataset
from gluonts.dataset.util import to_pandas

from .base import UnivariateCRPSTask
from .utils import get_random_window_univar


class SensorPeriodicMaintenanceTask(UnivariateCRPSTask):
    """
    A task where the history contains misleading information due to periodic
    sensor maintenance. The maintenance periods should not be reflected in
    the forecast.

    """

    def __init__(self, fixed_config: dict = None, seed: int = None):
        super().__init__(seed=seed, fixed_config=fixed_config)

    def random_instance(self):
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
            history_factor=self.random.randint(3, 7),
            random=self.random,
        )

        # Extract the history and future series
        history_series = window.iloc[: -metadata.prediction_length]
        future_series = window.iloc[-metadata.prediction_length :]

        if dataset_name == "electricity_hourly":
            # Duration: between 2 and 6 hours
            duration = self.random.randint(2, 7)
            start_hour = self.random.randint(0, 24 - duration)
            start_time = f"{start_hour:02d}:00"
            end_time = f"{(start_hour + duration):02d}:00"

            # Add the maintenance period to the window
            history_series.index = history_series.index.to_timestamp()
            history_series.loc[
                history_series.between_time(start_time, end_time).index
            ] = 0

            # Convert future index to timestamp for consistency
            future_series.index = future_series.index.to_timestamp()

            background = f"The sensor was offline for maintenance every day between {start_time} and {end_time}, which resulted in zero readings. This should be disregarded in the forecast."

        else:
            raise NotImplementedError(f"Dataset {dataset_name} is not supported.")

        # Instantiate the class variables
        self.past_time = history_series.to_frame()
        self.future_time = future_series.to_frame()
        self.constraints = None
        self.background = background
        self.scenario = None


class SensorTrendAccumulationTask(UnivariateCRPSTask):
    """
    A task where the history contains misleading information due to the
    measurement sensor accumulating a trend over time due to a calibration
    issue. The trend should not be reflected in the forecast.

    """

    def __init__(self, fixed_config: dict = None, seed: int = None):
        super().__init__(seed=seed, fixed_config=fixed_config)

    def random_instance(self):
        datasets = ["traffic"]

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
            history_factor=self.random.randint(3, 7),
            random=self.random,
        )

        # Extract the history and future series
        history_series = window.iloc[: -metadata.prediction_length]
        future_series = window.iloc[-metadata.prediction_length :]

        if dataset_name == "traffic":
            # Sample a starting point in the first half of the history's index
            history_series.index = history_series.index.to_timestamp()
            start_point = self.random.choice(
                history_series.index[: len(history_series) // 2]
            )
            n_points_slope = len(history_series) - history_series.index.get_loc(
                start_point
            )

            # Slope: make sure the mean increases by something between 1 and 1.5
            mean = history_series.mean()
            min_slope = mean / (len(history_series) - n_points_slope)
            max_slope = 1.5 * min_slope
            slope = self.random.uniform(min_slope, max_slope)

            # Add slope to history series based on number of measurements
            # XXX: Assumes a constant frequency
            history_series.loc[start_point:] = history_series.loc[
                start_point:
            ] + np.float32(slope * np.arange(n_points_slope))

            # Convert future index to timestamp for consistency
            future_series.index = future_series.index.to_timestamp()

            background = f"The sensor had a calibration problem starting from {start_point}, which resulted in an upward trend. This should be disregarded in the forecast."

        else:
            raise NotImplementedError(f"Dataset {dataset_name} is not supported.")

        # Instantiate the class variables
        self.past_time = history_series.to_frame()
        self.future_time = future_series.to_frame()
        self.constraints = None
        self.background = background
        self.scenario = None


class SensorSpikeTask(UnivariateCRPSTask):
    """
    A task where the history contains misleading information due to the
    measurement sensor having random spikes due to an unexpected glitch.
    This should not affect the forecast.
    # TODO: Support more spikes: in which case single-timesteps spikes would be trivial; but it is non-trivial to handle multi-length spikes
    """

    def __init__(self, fixed_config: dict = None, seed: int = None):
        super().__init__(seed=seed, fixed_config=fixed_config)

    def random_instance(self):
        datasets = ["traffic"]

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
            history_factor=self.random.randint(3, 7),
            random=self.random,
        )

        # Extract the history and future series
        history_series = window.iloc[: -metadata.prediction_length]
        future_series = window.iloc[-metadata.prediction_length :]

        if dataset_name == "traffic":
            # Sample a starting point in the first half of the history's index
            history_series.index = history_series.index.to_timestamp()
            spike_start_date = self.random.choice(
                history_series.index[
                    len(history_series) // 2 : -4
                ]  # Leave 3 points at the end: arbitrary
            )  # Arbitrary start point for now
            spike_start_point = history_series.index.get_loc(spike_start_date)
            spike_duration = self.random.choice(
                [1, 2, 3]
            )  # Arbitrarily picked from 1,2,3
            spike_type = self.random.choice([-1, 1])  # Upward spike or downward spike
            spike_magnitude = (
                self.random.choice([2, 3]) * history_series.max()
            )  # Arbitrarily set to twice or thrice the max value in the time series
            # Add spike to the data
            history_series.iloc[
                spike_start_point : spike_start_point + spike_duration
            ] = (spike_type * spike_magnitude)

            # Convert future index to timestamp for consistency
            future_series.index = future_series.index.to_timestamp()

            background = f"The sensor experienced an unexpected glitch resulting in a spike starting from {spike_start_date} for {spike_duration} timesteps. This should be disregarded in the forecast."

        else:
            raise NotImplementedError(f"Dataset {dataset_name} is not supported.")

        # Instantiate the class variables
        self.past_time = history_series.to_frame()
        self.future_time = future_series.to_frame()
        self.constraints = None
        self.background = background
        self.scenario = None


__TASKS__ = [
    SensorPeriodicMaintenanceTask,
    SensorTrendAccumulationTask,
    SensorSpikeTask,
]
