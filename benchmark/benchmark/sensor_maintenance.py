import logging
import numpy as np
import os
import glob
import pandas as pd

from tactis.gluon.dataset import get_dataset
from gluonts.dataset.util import to_pandas

from .base import UnivariateCRPSTask
from .utils import get_random_window_univar, datetime_to_str


from benchmark.data.pems import (
    load_traffic_series,
    get_traffic_prediction_length,
    get_traffic_history_factor,
)


class SensorPeriodicMaintenanceTask(UnivariateCRPSTask):
    """
    A task where the history contains misleading information due to periodic
    sensor maintenance. The maintenance periods should not be reflected in
    the forecast.

    """

    _context_sources = UnivariateCRPSTask._context_sources + ["c_cov", "c_h"]
    _skills = UnivariateCRPSTask._skills + ["instruction following"]
    __version__ = "0.0.1"  # Modification will trigger re-caching

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
            start_hour = self.random.randint(
                0, 24 - (duration + 1)
            )  # +1 so the drop doesn't come at the end of the history window
            maintenance_start_date = history_series.index[start_hour]
            maintenance_end_date = history_series.index[start_hour + duration]

            # Make the hour strings
            maintenance_start_hour = f"{maintenance_start_date.hour:02d}:00"
            maintenance_end_hour = f"{(maintenance_end_date.hour):02d}:00"

            # Add the maintenance period to the prediction window
            history_series.index = history_series.index.to_timestamp()
            history_series.loc[
                history_series.between_time(
                    maintenance_start_hour, maintenance_end_hour
                ).index
            ] = 0
            # Convert history index to timestamp for consistency
            future_series.index = future_series.index.to_timestamp()

            background = f"The sensor was offline for maintenance every day between {maintenance_start_hour} and {maintenance_end_hour}, which resulted in zero readings. Assume that the sensor will not be in maintenance in the future."
        else:
            raise NotImplementedError(f"Dataset {dataset_name} is not supported.")

        # Instantiate the class variables
        self.past_time = history_series.to_frame()
        self.future_time = future_series.to_frame()
        self.background = background

        # ROI parameters to add focus to the times where there would have been maintenance in the prediction region
        maintenance_hours_in_pred = [
            future_series.index.get_loc(x)
            for x in future_series.between_time(
                maintenance_start_hour, maintenance_end_hour
            ).index
        ]
        self.region_of_interest = maintenance_hours_in_pred


class SensorTrendAccumulationTask(UnivariateCRPSTask):
    """
    A task where the history contains misleading information due to the
    measurement sensor accumulating a trend over time due to a calibration
    issue. The trend should not be reflected in the forecast.

    """

    _context_sources = UnivariateCRPSTask._context_sources + ["c_cov", "c_h"]
    _skills = UnivariateCRPSTask._skills + ["instruction following", "reasoning: math"]
    __version__ = "0.0.2"  # Modification will trigger re-caching

    def get_series(
        self,
        dataset_name: str = "traffic",
        target=None,  #  'Speed (mph)' or 'Occupancy (%)'
    ):
        if dataset_name == "traffic":
            if target is None:
                target = "Occupancy (%)"
            series = load_traffic_series(target=target, random=self.random)
        else:
            raise NotImplementedError(f"Dataset {dataset_name} is not supported.")
        return series

    def get_prediction_length(self, dataset_name: str = "traffic"):
        if dataset_name == "traffic":
            return get_traffic_prediction_length()
        else:
            raise NotImplementedError(f"Dataset {dataset_name} is not supported.")

    def get_history_factor(self, dataset_name: str = "traffic"):
        if dataset_name == "traffic":
            return get_traffic_history_factor()
        else:
            raise NotImplementedError(f"Dataset {dataset_name} is not supported.")

    def random_instance(self):
        datasets = ["traffic"]

        dataset_name = self.random.choice(datasets)

        full_series = self.get_series(dataset_name=dataset_name)
        prediction_length = self.get_prediction_length(dataset_name=dataset_name)

        # Select a random window
        window = get_random_window_univar(
            full_series,
            prediction_length=prediction_length,
            history_factor=self.get_history_factor(dataset_name=dataset_name),
            random=self.random,
        )

        # Extract the history and future series
        history_series = window.iloc[:-prediction_length]
        future_series = window.iloc[-prediction_length:]

        if dataset_name == "traffic":
            # Sample a starting point in the first half of the history's index
            history_series.index = pd.to_datetime(history_series.index)
            start_point = self.random.choice(
                history_series.index[: len(history_series) // 2]
            )
            n_points_slope = len(history_series) - history_series.index.get_loc(
                start_point
            )

            # Design an artificial additive trend
            # XXX: We make sure that the maximum increase to any value in the series if
            #      of 1.25 to 2 times the absolute mean value of the series. This ensures a
            #      significant trend without making the series explode.
            mean = np.abs(history_series.mean())
            factor = (
                1.25 + self.random.rand() * 0.75
            )  # Random factor between 1.25 and 2
            # XXX: Assumes a constant frequency
            trend = np.linspace(0, factor * mean, n_points_slope + 1)[
                1:
            ]  # Start at non-zero value

            # Add trend to the series
            history_series.loc[start_point:] = history_series.loc[
                start_point:
            ] + np.float32(trend)

            # Convert future index to timestamp for consistency
            future_series.index = pd.to_datetime(future_series.index)

            background = (
                f"The sensor had a calibration problem starting from {datetime_to_str(start_point)} "
                + f"which resulted in an additive trend in the series that increases by {trend[1] - trend[0]:.4f} at every observed timestep. "
                + "Assume that the sensor will not have this calibration problem in the future."
            )

        else:
            raise NotImplementedError(f"Dataset {dataset_name} is not supported.")

        # Instantiate the class variables
        self.past_time = history_series.to_frame()
        self.future_time = future_series.to_frame()
        self.background = background

        # No RoI need to be defined as the full prediction window is important


class SensorSpikeTask(UnivariateCRPSTask):
    """
    A task where the history contains misleading information due to the
    measurement sensor having random spikes due to an unexpected glitch.
    This should not affect the forecast.
    # TODO: Support more spikes: in which case single-timesteps spikes would be trivial; but it is non-trivial to handle multi-length spikes
    """

    _context_sources = UnivariateCRPSTask._context_sources + ["c_cov", "c_h"]
    _skills = UnivariateCRPSTask._skills + ["instruction following"]
    __version__ = "0.0.2 "  # Modification will trigger re-caching

    def get_series(
        self,
        dataset_name: str = "traffic",
        target=None,  #  'Speed (mph)' or 'Occupancy (%)'
    ):
        if dataset_name == "traffic":
            if target is None:
                target = "Occupancy (%)"
            series = load_traffic_series(target=target, random=self.random)
        else:
            raise NotImplementedError(f"Dataset {dataset_name} is not supported.")
        return series

    def get_prediction_length(self, dataset_name: str = "traffic"):
        if dataset_name == "traffic":
            return get_traffic_prediction_length()
        else:
            raise NotImplementedError(f"Dataset {dataset_name} is not supported.")

    def get_history_factor(self, dataset_name: str = "traffic"):
        if dataset_name == "traffic":
            return get_traffic_history_factor()
        else:
            raise NotImplementedError(f"Dataset {dataset_name} is not supported.")

    def random_instance(self):
        datasets = ["traffic"]

        dataset_name = self.random.choice(datasets)

        full_series = self.get_series(dataset_name=dataset_name)
        prediction_length = self.get_prediction_length(dataset_name=dataset_name)

        # Select a random window
        window = get_random_window_univar(
            full_series,
            prediction_length=prediction_length,
            history_factor=self.get_history_factor(dataset_name=dataset_name),
            random=self.random,
        )

        # Extract the history and future series
        history_series = window.iloc[:-prediction_length]
        future_series = window.iloc[-prediction_length:]

        if dataset_name == "traffic":
            # Sample a starting point in the first half of the history's index
            history_series.index = pd.to_datetime(history_series.index)
            spike_duration = self.random.choice(
                [1, 2, 3]
            )  # Arbitrarily picked from 1,2,3
            # Arbitrary way to select a start date: sort the values of future_series (excluding the last spike_duration+1 points), pick it from the largest 5 values
            spike_start_point = self.random.choice(
                np.argsort(future_series.values[: -(spike_duration + 1)])[-5:][::-1]
            )
            spike_start_date = future_series.index[spike_start_point]
            spike_type = self.random.choice([-1, 1])  # Upward spike or downward spike
            spike_magnitude = (
                self.random.choice([2, 3]) * history_series.max()
            )  # Arbitrarily set to twice or thrice the max value in the time series
            # Add spike to the data
            history_series.iloc[
                spike_start_point : spike_start_point + spike_duration
            ] = (spike_type * spike_magnitude)

            # Convert future index to timestamp for consistency
            future_series.index = pd.to_datetime(future_series.index)

            background = f"The sensor experienced an unexpected glitch resulting in a spike starting from {datetime_to_str(spike_start_date)} for {spike_duration} {'hour' if spike_duration == 1 else 'hours'}. Assume that the sensor will not have this glitch in the future."

        else:
            raise NotImplementedError(f"Dataset {dataset_name} is not supported.")

        # Instantiate the class variables
        self.past_time = history_series.to_frame()
        self.future_time = future_series.to_frame()
        self.background = background

        # ROI metric parameters
        self.region_of_interest = slice(
            spike_start_point, spike_start_point + spike_duration
        )


class SensorMaintenanceInPredictionTask(UnivariateCRPSTask):
    """
    A task where the prediction part contains zero readings for a period due to maintenance.
    The maintenance periods should be reflected in the forecast.
    """

    _context_sources = UnivariateCRPSTask._context_sources + ["c_cov", "c_f"]
    _skills = UnivariateCRPSTask._skills + ["instruction following"]
    __version__ = "0.0.1"  # Modification will trigger re-caching

    def random_instance(self):
        # TODO: This task can use all datasets where the notion of a "sensor" is meaningful
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
            start_hour = self.random.randint(0, metadata.prediction_length - duration)
            maintenance_start_date = future_series.index[start_hour]
            maintenance_end_date = future_series.index[start_hour + duration]

            # Add the maintenance period to the prediction window
            future_series.index = future_series.index.to_timestamp()
            future_series.iloc[start_hour : start_hour + duration] = 0

            # Convert history index to timestamp for consistency
            history_series.index = history_series.index.to_timestamp()

            scenario = f"Consider that the sensor will be offline for maintenance between {datetime_to_str(maintenance_start_date)} and {datetime_to_str(maintenance_end_date)}, which results in zero readings."
        else:
            raise NotImplementedError(f"Dataset {dataset_name} is not supported.")

        # Instantiate the class variables
        self.past_time = history_series.to_frame()
        self.future_time = future_series.to_frame()
        self.scenario = scenario

        # ROI metric parameters
        self.region_of_interest = slice(start_hour, start_hour + duration)


__TASKS__ = [
    SensorMaintenanceInPredictionTask,
    SensorPeriodicMaintenanceTask,
    SensorTrendAccumulationTask,
    SensorSpikeTask,
]
