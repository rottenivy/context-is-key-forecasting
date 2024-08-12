"""
Tasks based on the NN5 dataset (nn5_daily_without_missing), which is a dataset (with 111 series) of total number of cash withdrawals from 111 different Automated Teller
Machines (ATM) in the UK.
"""

from tactis.gluon.dataset import get_dataset
from gluonts.dataset.util import to_pandas

from .base import UnivariateCRPSTask
from .utils import get_random_window_univar, datetime_to_str


class CashDepletedinATMScenarioTask(UnivariateCRPSTask):
    """
    This task considers a scenario where cash is depleted at an ATM for a duration, and no withdrawals are possible during that duration.
    The depletion occurs in the prediction horizon, and should be deductable by a model from the text context.
    """

    def __init__(self, fixed_config: dict = None, seed: int = None):
        super().__init__(seed=seed, fixed_config=fixed_config)

    def random_instance(self):
        datasets = [
            "nn5_daily_without_missing"
        ]  # nn5_daily_without_missing has a prediction length of 56

        # Select a dataset
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

        if dataset_name == "nn5_daily_without_missing":
            drop_duration = self.random.choice(
                list(range(1, 6))
            )  # Arbitrarily picked from 1-5 hours
            future_series.index = future_series.index.to_timestamp()
            drop_start_date = self.random.choice(
                future_series.index[
                    :-7
                ]  # Starting point is anywhere from start of series to max(drop_duration) + 1 points before the series. +1 is so as to not have the drop not completely at the end of the pred.
            )
            # Introduce a zero period in the future series at that duration
            drop_start_point = future_series.index.get_loc(drop_start_date)
            future_series.iloc[drop_start_point : drop_start_point + drop_duration] = 0

            # Convert history index to timestamp for consistency
            history_series.index = history_series.index.to_timestamp()

        background = f"This is the number of cash withdrawals from an automated teller machine (ATM) in an arbitrary location in England, recorded at an hourly frequency."
        scenario = f"Suppose that cash will be depleted in the ATM from {datetime_to_str(drop_start_date)}, for {drop_duration} hours, leading to no withdrawals being possible during that period."  # TODO: May also specify drop end date instead of the drop duration.

        # Instantiate the class variables
        self.past_time = history_series.to_frame()
        self.future_time = future_series.to_frame()
        self.constraints = None
        self.background = background
        self.scenario = scenario


class ATMUnderPeriodicMaintenanceTask(UnivariateCRPSTask):
    """
    This task considers that an ATM is under periodic maintenance in the history repeatedly at certain intervals, which leads to misleading history. The context provides background information about this period.
    This period should be ignored by the forecasting algorithm in its forecasts.
    """

    def __init__(self, fixed_config: dict = None, seed: int = None):
        super().__init__(seed=seed, fixed_config=fixed_config)

    def random_instance(self):
        datasets = [
            "nn5_daily_without_missing"
        ]  # nn5_daily_without_missing has a prediction length of 56 (~~2 days)

        # Select a dataset
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

        history_factor = self.random.randint(7, 11)
        # Select a random window
        window = get_random_window_univar(
            full_series,
            prediction_length=metadata.prediction_length,
            history_factor=history_factor,
            random=self.random,
        )

        # Extract the history and future series
        history_series = window.iloc[: -metadata.prediction_length]
        future_series = window.iloc[-metadata.prediction_length :]

        history_series.index = history_series.index.to_timestamp()

        drop_duration = self.random.choice(
            list(range(4, 7))
        )  # Arbitrarily picked from 4-6 hours
        drop_spacing = 24 * self.random.choice(
            list(range(3, 7))
        )  # Arbitrarily picked from 3-6 days (hence multiplied by 24)
        drop_start_date = self.random.choice(
            history_series.index[
                : 24 * 1
            ]  # Starting point is anywhere from start of series to 1 day. This is done so we can get multiple such drops in the series going forward in the history.
        )
        drop_start_point = history_series.index.get_loc(drop_start_date)
        start_point = drop_start_point
        # Add drops to the data
        while start_point + drop_duration < len(history_series.index):
            history_series.iloc[start_point : start_point + drop_duration] = 0
            start_point += drop_spacing
        # Convert future index to timestamp for consistency
        future_series.index = future_series.index.to_timestamp()

        background = f"This is the number of cash withdrawals from an automated teller machine (ATM) in an arbitrary location in England, recorded at an hourly frequency."  # This is generic background information common to all NN5 tasks
        background += f" The ATM was under maintenance for {drop_duration} hours, periodically every {drop_spacing} days, starting from {datetime_to_str(drop_start_date)}. This should be disregarded in the forecast."

        # Instantiate the class variables
        self.past_time = history_series.to_frame()
        self.future_time = future_series.to_frame()
        self.constraints = None
        self.background = background
        self.scenario = None


class IncreasedWithdrawalScenario(UnivariateCRPSTask):
    """
    This task considers a scenario where the number of ATM withdrawals possible by a person are increased by the bank, resulting in more than usual withdrawals.
    This is a scenario that occurs in the prediction horizon, and should be deductable by a model from the text context.
    """

    def __init__(self, fixed_config: dict = None, seed: int = None):
        super().__init__(seed=seed, fixed_config=fixed_config)

    def random_instance(self):
        datasets = [
            "nn5_daily_without_missing"
        ]  # nn5_daily_without_missing has a prediction length of 56

        # Select a dataset
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

        limit_off_duration = self.random.choice(
            list(range(1, 25))
        )  # Arbitrarily picked from 1-24 hours
        future_series.index = future_series.index.to_timestamp()
        limit_off_start_date = self.random.choice(
            future_series.index[
                :-26
            ]  # Starting point is anywhere from start of series to max(drop_duration) + 1 points before the series. +1 is so as to have the drop not completely at the end of the pred.
        )  # Arbitrary start point for now
        # Introduce a zero period in the future series at a particular duration
        limit_off_start_point = future_series.index.get_loc(limit_off_start_date)
        # Determine increase in number of withdrawals
        increase_factor = self.random.choice(
            list(range(3, 6))
        )  # Randomly picked from 3 to 5. Arbitrary.
        future_series.iloc[
            limit_off_start_point : limit_off_start_point + limit_off_duration
        ] = (
            increase_factor
            * future_series.iloc[
                limit_off_start_point : limit_off_start_point + limit_off_duration
            ]
        )

        # Convert history index to timestamp for consistency
        history_series.index = history_series.index.to_timestamp()

        background = f"This is the number of cash withdrawals from an automated teller machine (ATM) in an arbitrary location in England, recorded at an hourly frequency."
        scenario = f"Suppose that there is a festival from {datetime_to_str(limit_off_start_date)}, for {limit_off_duration} hours leading to more people in the area, and {increase_factor} times the number of usual withdrawals during that period."
        # Covariate task: Event or not

        # Instantiate the class variables
        self.past_time = history_series.to_frame()
        self.future_time = future_series.to_frame()
        self.constraints = None
        self.background = background
        self.scenario = scenario


__TASKS__ = [
    CashDepletedinATMScenarioTask,
    ATMUnderPeriodicMaintenanceTask,
    IncreasedWithdrawalScenario,
]
