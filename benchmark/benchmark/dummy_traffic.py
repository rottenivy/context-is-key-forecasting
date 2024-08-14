import numpy as np

import pandas as pd
import json
import os
import glob

from benchmark.utils import get_random_window_univar

from benchmark.base import UnivariateCRPSTask

# TODO: add to package
from benchmark.data.pems import (
    download_traffic_files,
    TRAFFIC_SPLIT_PATH,
    TRAFFIC_METADATA_PATH,
)


class DummyTrafficTask(UnivariateCRPSTask):

    def __init__(
        self,
        fixed_config: dict = None,
        seed: int = None,
    ):
        self.init_data()
        self.traffic_split_path = TRAFFIC_SPLIT_PATH
        self.traffic_metadata_path = TRAFFIC_METADATA_PATH
        super().__init__(seed=seed, fixed_config=fixed_config)

    def init_data(self):
        """
        Check integrity of data files and download if needed.

        """
        if not os.path.exists(TRAFFIC_SPLIT_PATH) or not os.path.exists(
            TRAFFIC_METADATA_PATH
        ):
            download_traffic_files()

    def random_instance(self):
        # glob all sensor files
        sensor_files = glob.glob(os.path.join(self.traffic_split_path, "*.csv"))

        random_sensor_file = self.random.choice(sensor_files)

        dataset = pd.read_csv(random_sensor_file)

        dataset["date"] = pd.to_datetime(dataset["Hour"])
        dataset = dataset.set_index("date")

        self.prediction_length = 24

        series = dataset["Speed (mph)"]

        history_factor = 7

        window = get_random_window_univar(
            series,
            prediction_length=self.prediction_length,
            history_factor=history_factor,
            random=self.random,
        )

        # extract the history and future series
        history_series = window.iloc[: -self.prediction_length]
        future_series = window.iloc[-self.prediction_length :]

        self.past_time = history_series.to_frame()
        self.future_time = future_series.to_frame()
        self.constraints = None
        self.background = self.get_context(dataset)
        self.scenario = None

    def get_context(self, dataset):
        """
        Get the context of the task.
        """
        sensor_id = dataset["sensor_id"].iloc[0]
        freeway_dir = dataset["Fwy"].iloc[0]
        district = dataset["District"].iloc[0]
        county = dataset["County"].iloc[0]

        return f"Sensor ID: {sensor_id}, Freeway: {freeway_dir}, District: {district}, County: {county}"

    @property
    def seasonal_period(self) -> int:
        """
        This returns the period which should be used by statistical models for this task.
        If negative, this means that the data either has no period, or the history is shorter than the period.
        """
        return 7


__TASKS__ = [DummyTrafficTask]
