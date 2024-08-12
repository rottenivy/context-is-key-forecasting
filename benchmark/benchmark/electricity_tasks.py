from tactis.gluon.dataset import get_dataset
from gluonts.dataset.util import to_pandas
import numpy as np

from .base import UnivariateCRPSTask
from .utils import get_random_window_univar, datetime_to_str


class ElectricityIncreaseInPredictionTask(UnivariateCRPSTask):
    """
    A task where the consumption of electricity spikes in prediction part,
    due to a heat wave and people using a lot of air conditioning.
    The spikes should be deducted from the context and reflected in the forecast.
    TODO: A multivariate extension of this task, where weather is another time series

    """

    def __init__(self, fixed_config: dict = None, seed: int = None):
        super().__init__(seed=seed, fixed_config=fixed_config)
        self.context_flags["c_cov"] = True
        self.context_flags["c_f"] = True

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
            # Sample a starting point in the first half of the prediction
            future_series.index = future_series.index.to_timestamp()
            # Arbitrary way to select a start date: sort the values of future_series (excluding the last 4 points), pick it from the largest 5 values
            spike_start_point = self.random.choice(
                np.argsort(future_series.values[:-4])[-5:][::-1]
            )
            spike_start_date = future_series.index[spike_start_point]
            spike_duration = self.random.choice(
                [1, 2, 3]
            )  # Arbitrarily picked from 1,2,3
            spike_magnitude = self.random.choice(
                [3, 4, 5]
            )  # Arbitrarily set to twice or thrice the max value in the time series
            # Add spike to the data
            future_series.iloc[
                spike_start_point : spike_start_point + spike_duration
            ] = (spike_magnitude * future_series.iloc[spike_start_point])

            # Convert future index to timestamp for consistency
            history_series.index = history_series.index.to_timestamp()

            background = (
                f"This is the electricity consumption recorded in Kilowatt (kW)."
            )
            scenario = f"Suppose that there is a heat wave from {datetime_to_str(spike_start_date)} for {spike_duration} {'hour' if spike_duration == 1 else 'hours'}, leading to excessive use of air conditioning, and {spike_magnitude} times the usual electricity being consumed."

        else:
            raise NotImplementedError(f"Dataset {dataset_name} is not supported.")

        # Instantiate the class variables
        self.past_time = history_series.to_frame()
        self.future_time = future_series.to_frame()
        self.constraints = None
        self.background = None
        self.scenario = scenario


__TASKS__ = [ElectricityIncreaseInPredictionTask]
