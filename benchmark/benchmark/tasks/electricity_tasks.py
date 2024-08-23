from tactis.gluon.dataset import get_dataset
from gluonts.dataset.util import to_pandas
import numpy as np

from ..base import UnivariateCRPSTask
from ..utils import get_random_window_univar, datetime_to_str


class ElectricityIncreaseInPredictionTask(UnivariateCRPSTask):
    """
    A task where the consumption of electricity spikes in prediction part,
    due to a heat wave and people using a lot of air conditioning.
    The spikes should be deducted from the context and reflected in the forecast.
    TODO: A multivariate extension of this task, where weather is another time series

    """

    _context_sources = UnivariateCRPSTask._context_sources + ["c_cov", "c_f"]
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
            scenario = self.get_scenario(
                spike_start_date, spike_duration, spike_magnitude
            )

        else:
            raise NotImplementedError(f"Dataset {dataset_name} is not supported.")

        # Instantiate the class variables
        self.past_time = history_series.to_frame()
        self.future_time = future_series.to_frame()
        self.constraints = None
        self.background = None
        self.scenario = scenario

        # ROI metric parameters
        self.region_of_interest = slice(
            spike_start_point, spike_start_point + spike_duration
        )

    def get_scenario(self, spike_start_date, spike_duration, spike_magnitude):
        return f"Suppose that there is a heat wave from {datetime_to_str(spike_start_date)} for {spike_duration} {'hour' if spike_duration == 1 else 'hours'}, leading to excessive use of air conditioning, and {spike_magnitude} times the usual electricity being consumed."


class ShortNewsElectricityIncreaseInPredictionTask(ElectricityIncreaseInPredictionTask):
    """
    A version of the ElectricityIncreaseInPredictionTask where the relevent
    information must be retrieved from within a short news article provided in context.

    """

    _context_sources = UnivariateCRPSTask._context_sources + ["c_cov", "c_f"]
    _skills = UnivariateCRPSTask._skills + [
        "instruction following",
        "retrieval: context",
    ]
    __version__ = "0.0.1"  # Modification will trigger re-caching

    def get_scenario(self, spike_start_date, spike_duration, spike_magnitude):
        # This news article was generated with the assistance of Claude
        scenario = f"A heatwave struck the city, which began on {datetime_to_str(spike_start_date)} and lasted for approximately {spike_duration} {'hour' if spike_duration == 1 else 'hours'}, saw temperatures soar to unprecedented levels. According to the city's electricity provider, power consumption during the peak of the heatwave reached approximately {spike_magnitude} times the typical usage for this time of year."
        return scenario


class MediumNewsElectricityIncreaseInPredictionTask(
    ElectricityIncreaseInPredictionTask
):
    """
    A version of the ElectricityIncreaseInPredictionTask where the relevent
    information must be retrieved from within a medium length news article provided in context.

    """

    _context_sources = UnivariateCRPSTask._context_sources + ["c_cov", "c_f"]
    _skills = UnivariateCRPSTask._skills + [
        "instruction following",
        "retrieval: context",
    ]
    __version__ = "0.0.1"  # Modification will trigger re-caching

    def get_scenario(self, spike_start_date, spike_duration, spike_magnitude):
        # This news article was generated with the assistance of Claude
        scenario = f"A sudden and intense heatwave struck the city, causing a dramatic surge in electricity consumption as residents sought refuge from the scorching temperatures. The extreme weather event, which began on {datetime_to_str(spike_start_date)} and lasted for approximately {spike_duration} {'hour' if spike_duration == 1 else 'hours'}, saw temperatures soar to unprecedented levels. In response, citizens across the metropolitan area turned to their air conditioning units en masse, leading to a significant strain on the local power grid. According to the city's electricity provider, power consumption during the peak of the heatwave reached approximately {spike_magnitude} times the typical usage for this time of year. \nFor now, citizens are encouraged to stay hydrated, check on vulnerable neighbors, and use air conditioning responsibly as the community works together to beat the heat."
        return scenario


class LongNewsElectricityIncreaseInPredictionTask(ElectricityIncreaseInPredictionTask):
    """
    A version of the ElectricityIncreaseInPredictionTask where the relevent
    information must be retrieved from within a long news article provided in context.

    """

    _context_sources = UnivariateCRPSTask._context_sources + ["c_cov", "c_f"]
    _skills = UnivariateCRPSTask._skills + [
        "instruction following",
        "retrieval: context",
    ]
    __version__ = "0.0.1"  # Modification will trigger re-caching

    def get_scenario(self, spike_start_date, spike_duration, spike_magnitude):
        # This news article was generated with the assistance of Claude
        scenario = f"A sudden and intense heatwave struck the city, causing a dramatic surge in electricity consumption as residents sought refuge from the scorching temperatures. The extreme weather event, which began on {datetime_to_str(spike_start_date)} and lasted for approximately {spike_duration} {'hour' if spike_duration == 1 else 'hours'}, saw temperatures soar to unprecedented levels. In response, citizens across the metropolitan area turned to their air conditioning units en masse, leading to a significant strain on the local power grid.According to the city's electricity provider, power consumption during the peak of the heatwave reached approximately {spike_magnitude} times the typical usage for this time of year. \"We've never seen anything quite like this,\" said Jane Smith, spokesperson for PowerCity Utilities. \"The sudden spike in demand pushed our systems to their limits.\" \nAs the city recovers from this unprecedented power surge, experts are already discussing long-term solutions to manage similar situations in the future. These may include upgrades to the power grid, incentives for energy-efficient appliances, and the development of more robust emergency response protocols. \nFor now, citizens are encouraged to stay hydrated, check on vulnerable neighbors, and use air conditioning responsibly as the community works together to beat the heat."
        return scenario


__TASKS__ = [
    ElectricityIncreaseInPredictionTask,
    ShortNewsElectricityIncreaseInPredictionTask,
    MediumNewsElectricityIncreaseInPredictionTask,
    LongNewsElectricityIncreaseInPredictionTask,
]
