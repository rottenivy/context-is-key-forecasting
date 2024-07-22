# from tactis.gluon.dataset import get_dataset
from gluonts.dataset.util import to_pandas

from tactis.gluon.dataset import get_dataset

from .base import UnivariateCRPSTask
from .utils import get_random_window_univar


class PredictableSpikesInPredTask(UnivariateCRPSTask):
    """
    Adds spikes to an arbitrary series.
    The presence of the spike is included in the context.
    Time series: agnostic
    Context: synthetic
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
            history_factor=self.random.randint(2, 5),
            random=self.random,
        )

        # Extract the history and future series
        history_series = window.iloc[: -metadata.prediction_length]
        future_series = window.iloc[-metadata.prediction_length :]

        if dataset_name == "electricity_hourly":
            start_hour = self.random.randint(0, 24 - 1)
            start_time = f"{start_hour:02d}:00"
            end_time = f"{(start_hour):02d}:00"

            history_series.index = history_series.index.to_timestamp()
            future_series.index = future_series.index.to_timestamp()
            ground_truth = future_series.copy()

            # Add the spike
            relative_impact = self.random.randint(1, 500)
            is_negative = self.random.choice([True, False])
            if is_negative:
                relative_impact = -relative_impact
            future_series.loc[
                future_series.between_time(start_time, end_time).index
            ] = future_series.loc[
                future_series.between_time(start_time, end_time).index
            ] + future_series.loc[
                future_series.between_time(start_time, end_time).index
            ] * (
                relative_impact / 100
            )

            background = f"An spike of {relative_impact}% is expected at exactly {start_time}, after which the series will return to normal."

        self.past_time = history_series
        self.future_time = future_series
        self.ground_truth = ground_truth
        self.constraints = None
        self.background = background
        self.scenario = None
