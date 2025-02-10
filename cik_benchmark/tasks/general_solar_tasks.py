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
from .nsrdb_tasks import download_all_nsrdb_datasets
from ..memorization_mitigation import add_realistic_noise
from . import WeightCluster


class ClimatologyIrradianceTask(UnivariateCRPSTask):
    """
    A task where the model is tasked to forecast the second half of the day,
    for a dataset which has a strong daily periodicity.
    The task comes with a textual description which can helps the model
    learns the daily shape of the signal.
    Use the NSRDB data instead of the solar_10min dataset.
    Use the Direct Normal Irradiance, due to being more variable than the Global Horizontal Irradiance.
    """

    __version__ = "0.0.1"  # Modification will trigger re-caching

    def random_instance(self):
        # All lot of this work is repeated for all instances, so it wouldn't hurt to cache it.
        all_data = download_all_nsrdb_datasets(interval=10)

        # Select a random time series
        ts_index = self.random.choice(len(all_data))
        header = all_data[ts_index][0]
        full_series = all_data[ts_index][1]["GHI"]

        # The NSRDB dataset is for the whole year of 2022.
        # Select a day for the forecast between 2022-07-01 (182nd day) and 2022-12-31 (365th day).
        day = self.random.randint(low=181, high=365)
        # Select the start of the forecast period, from 10:30 to 13:30.
        forecast_time = self.random.randint(low=10 * 6 + 3, high=13 * 6 + 3 + 1)
        # history = first ~12 hours of the day, target = second ~12 hours of the day
        full_history_series = full_series.iloc[: (day * 24 * 6)]
        history_series = full_series.iloc[
            (day * 24 * 6) : (day * 24 * 6) + forecast_time
        ]
        future_series = full_series.iloc[
            (day * 24 * 6) + forecast_time : (day * 24 * 6) + 24 * 6
        ]

        state = header["State"][2:-1]
        country = header["Country"][2:-1]

        # Shift the dates by one day forward
        history_series.index = history_series.index + pd.Timedelta(days=1)
        future_series.index = future_series.index + pd.Timedelta(days=1)

        background = self.get_background(
            full_history_series,
            state,
            country,
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
        self,
        full_history_series: pd.Series,
        state: str,
        country: str,
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

    @property
    def max_directprompt_batch_size(self) -> int:
        """
        If set, only request that many samples at once when using a method using Direct Prompting.
        Mainly used to avoid crashing the Llama3-405b server.
        """
        return 5


class ClimatologyPowerProduction(UnivariateCRPSTask):
    """
    A task where the model is tasked to forecast the second half of the day,
    for a dataset which has a strong daily periodicity.
    The task comes with a textual description which can helps the model
    learns the daily shape of the signal.
    """

    __version__ = "0.0.1"  # Modification will trigger re-caching

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

        # Transform
        history_series = add_realistic_noise(
            history_series, self.random, noise_level=0.01, skip_zero_values=True
        )
        future_series = add_realistic_noise(
            future_series, self.random, noise_level=0.01, skip_zero_values=True
        )

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


    def make_transform(self, history_series, future_series):
        # Transform
        history_series = add_realistic_noise(
            history_series,
            self.random,
            noise_level=0.01,
            skip_zero_values=True,
            constraints=[MinConstraint(0)],
        )
        future_series = add_realistic_noise(
            future_series,
            self.random,
            noise_level=0.01,
            skip_zero_values=True,
            constraints=[MinConstraint(0)],
        )


    @property
    def seasonal_period(self) -> int:
        """
        This returns the period which should be used by statistical models for this task.
        If negative, this means that the data either has no period, or the history is shorter than the period.
        """
        # Not enough history for a single period
        return -1

    @property
    def max_directprompt_batch_size(self) -> int:
        """
        If set, only request that many samples at once when using a method using Direct Prompting.
        Mainly used to avoid crashing the Llama3-405b server.
        """
        return 5


__TASKS__ = [
    ClimatologyIrradianceTask,
    ClimatologyPowerProduction,
]

__CLUSTERS__ = [
    WeightCluster(
        weight=1,
        tasks=[
            ClimatologyIrradianceTask,
            ClimatologyPowerProduction,
        ],
    ),
]
