"""
Tasks where the given history is too short to accurately do the forecasting,
but context is given to allow the model to fill in the missing information.
"""

import datetime
import pandas as pd
from abc import abstractmethod

from gluonts.dataset.util import to_pandas
from gluonts.dataset.repository import get_dataset

from ..base import UnivariateCRPSTask
from ..config import DATA_STORAGE_PATH
from ..metrics.constraints import MinConstraint


class BaseHalfDaySolarForecastTask(UnivariateCRPSTask):
    """
    A task where the model is tasked to forecast the second half of the day,
    for a dataset which has a strong daily periodicity.
    The task comes with a textual description which can helps the model
    learns the daily shape of the signal.
    """

    __version__ = "0.0.2"  # Modification will trigger re-caching

    def random_instance(self):
        dataset = get_dataset(
            "solar_10_minutes", regenerate=False, path=DATA_STORAGE_PATH
        )

        # Select a random time series
        ts_index = self.random.choice(len(dataset.test))
        full_series = to_pandas(list(dataset.test)[ts_index])

        # The solar_10_minutes dataset is for the whole year of 2006.
        # Select a day for the forecast between 2006-07-01 (182nd day) and 2006-12-31 (365th day).
        day = self.random.randint(low=181, high=365)
        # Select the start of the forecast period, from 10:30 to 13:30.
        forecast_time = self.random.randint(low=10 * 6 + 3, high=13 * 6 + 3 + 1)
        # history = first 12 hours of the day, target = second 12 hours of the day
        full_history_series = full_series.iloc[: (day * 24 * 6)]
        history_series = full_series.iloc[
            (day * 24 * 6) : (day * 24 * 6) + forecast_time
        ]
        future_series = full_series.iloc[
            (day * 24 * 6) + forecast_time : (day * 24 * 6) + 24 * 6
        ]

        background = self.get_background(
            full_history_series, future_series.index[0].start_time
        )

        # Instantiate the class variables
        self.past_time = history_series.to_frame()
        self.future_time = future_series.to_frame()
        self.constraints = None
        self.background = background
        self.scenario = None

        # No RoI need to be defined as the full prediction window is important

    @abstractmethod
    def get_background(
        self, full_history_series: pd.Series, forecast_date: pd.Timestamp
    ) -> str:
        """
        Generate a textual hint for the model to know about the potential shape of the daily solar intensity.
        """
        pass

    @property
    def seasonal_period(self) -> int:
        """
        This returns the period which should be used by statistical models for this task.
        If negative, this means that the data either has no period, or the history is shorter than the period.
        """
        # Not enough history for a single period
        return -1


class MinimalInfoHalfDaySolarForecastTask(BaseHalfDaySolarForecastTask):
    """
    Version of the task where only the minimal background information is given.
    """

    _context_sources = ["c_i"]
    _skills = BaseHalfDaySolarForecastTask._skills + ["reasoning: deduction"]
    __version__ = "0.0.1"  # Modification will trigger re-caching

    def get_background(
        self, full_history_series: pd.Series, forecast_date: pd.Timestamp
    ) -> str:
        return (
            "This series contains the power production of a photovoltaic power plant."
        )


class LocaleInfoHalfDaySolarForecastTask(BaseHalfDaySolarForecastTask):
    """
    Version of the task where the state in which the data was collected is mentioned.
    """

    _context_sources = ["c_i"]
    _skills = BaseHalfDaySolarForecastTask._skills + ["reasoning: deduction"]
    __version__ = "0.0.1"  # Modification will trigger re-caching

    def get_background(
        self, full_history_series: pd.Series, forecast_date: pd.Timestamp
    ) -> str:
        return "This series contains the power production of a photovoltaic power plant in the state of Alabama."


class ZenithInfoHalfDaySolarForecastTask(BaseHalfDaySolarForecastTask):
    """
    Version of the task where the average time at which the daily maximum is reached is mentioned.
    """

    _context_sources = ["c_i", "c_h"]
    _skills = BaseHalfDaySolarForecastTask._skills + ["reasoning: deduction"]
    __version__ = "0.0.1"  # Modification will trigger re-caching

    def get_background(
        self, full_history_series: pd.Series, forecast_date: pd.Timestamp
    ) -> str:
        def func(s):
            p = s.idxmax()
            t = p.start_time.time()
            return 3600 * t.hour + 60 * t.minute + t.second + 1e-6 * t.microsecond

        # The resample is effectively a groupby based on the given frequency, so this gives us the mean
        # time of the day at which the series is the highest.
        past_90_days = full_history_series.iloc[-90 * 24 * 6 :]
        mean_zenith_seconds = round(past_90_days.resample("D").apply(func).mean())
        mean_zenith_formated = (
            datetime.datetime.fromtimestamp(mean_zenith_seconds)
            .time()
            .strftime("%H:%M:%S")
        )

        return (
            "This series contains the power production of a photovoltaic power plant in the state of Alabama.\n"
            f"Over the previous 90 days, the maximum power production happened on average at {mean_zenith_formated}."
        )


class SimilarLocationDaySolarForecastTask(BaseHalfDaySolarForecastTask):
    """
    Version of the task where the average time at which the daily maximum is reached is mentioned.
    """

    _context_sources = ["c_i"]
    _skills = BaseHalfDaySolarForecastTask._skills + ["reasoning: analogy"]
    __version__ = "0.0.1"  # Modification will trigger re-caching

    def random_instance(self):
        dataset = get_dataset("solar_10_minutes", regenerate=False)

        # Average over all time series
        dataset_list = list(dataset.test)
        df = pd.concat([to_pandas(d) for d in dataset_list], axis=1)
        full_series = df.mean(1)

        # The solar_10_minutes dataset is for the whole year of 2006.
        # Select a day for the forecast between 2006-07-01 (182nd day) and 2006-12-31 (365th day).
        day = self.random.randint(low=181, high=365)
        # Select the start of the forecast period, from 8:30 to 10:30.
        forecast_time = self.random.randint(low=8 * 6 + 3, high=10 * 6 + 3 + 1)
        # Predict rest of the day
        full_history_series = full_series.iloc[: (day * 24 * 6)]
        history_series = full_series.iloc[
            (day * 24 * 6) : (day * 24 * 6) + forecast_time
        ]
        future_series = full_series.iloc[
            (day * 24 * 6) + forecast_time : (day * 24 * 6) + 24 * 6
        ]

        background = self.get_background(
            full_history_series, future_series.index[0].start_time
        )

        # Instantiate the class variables
        self.past_time = history_series.to_frame()
        self.future_time = future_series.to_frame()
        self.constraints = None
        self.metric_constraint = MinConstraint(0)
        self.background = background
        self.scenario = None

    def get_background(
        self, full_history_series: pd.Series, forecast_date: pd.Timestamp
    ) -> str:

        return "This series estimates the power production for a given day of a new solar power plant located in the state of Georgia, which has a humid subtropical climate."


class ExplicitSimilarLocationDaySolarForecastTask(SimilarLocationDaySolarForecastTask):
    """
    Version of the task where the average time at which the daily maximum is reached is mentioned.
    """

    _context_sources = ["c_i"]
    _skills = SimilarLocationDaySolarForecastTask._skills
    __version__ = "0.0.1"  # Modification will trigger re-caching

    def get_background(
        self, full_history_series: pd.Series, forecast_date: pd.Timestamp
    ) -> str:

        return "This series estimates the power production for a given day of a new solar power plant located in the state of Georgia, which has a climate similar to Alabama's."


class SimilarLocationWithReferenceDaySolarForecastTask(
    SimilarLocationDaySolarForecastTask
):
    """
    Version of the task where the average time at which the daily maximum is reached is mentioned.
    """

    _context_sources = ["c_i"]
    _skills = BaseHalfDaySolarForecastTask._skills + [
        "reasoning: deduction",
    ]
    __version__ = "0.0.1"  # Modification will trigger re-caching

    def get_background(
        self, full_history_series: pd.Series, forecast_date: pd.Timestamp
    ) -> str:

        def get_max(s):
            p = s.idxmax()
            max_value = s.max()
            t = p.start_time.time()
            return (
                3600 * t.hour + 60 * t.minute + t.second + 1e-6 * t.microsecond,
                max_value,
            )

        # Use the longest day for reference
        day = 171
        reference_series = full_history_series.iloc[day * 24 * 6 : day * (24 + 1) * 6]
        zenith_seconds, max_power = get_max(reference_series)
        zenith_formated = (
            datetime.datetime.fromtimestamp(zenith_seconds).time().strftime("%H:%M:%S")
        )

        return f"This series estimates the power production for a given day of a new solar power plant located in the state of Georgia, which has a humid subtropical climate. As reference, the maximal power production in similar states on June 20th was of {max_power:.2f} at {zenith_formated}."


__TASKS__ = [
    MinimalInfoHalfDaySolarForecastTask,
    LocaleInfoHalfDaySolarForecastTask,
    ZenithInfoHalfDaySolarForecastTask,
    SimilarLocationDaySolarForecastTask,
    ExplicitSimilarLocationDaySolarForecastTask,
    SimilarLocationWithReferenceDaySolarForecastTask,
]
