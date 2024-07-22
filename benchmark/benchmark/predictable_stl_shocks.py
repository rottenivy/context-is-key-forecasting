# from tactis.gluon.dataset import get_dataset
from gluonts.dataset.util import to_pandas

import random

from tactis.gluon.dataset import get_dataset

from .base import UnivariateCRPSTask
from .utils import get_random_window_univar

from statsmodels.tsa.seasonal import STL


class STLMutliplierTask(UnivariateCRPSTask):
    """
    A task where the series is first decomposed into trend, seasonality, and residuals
    using STL decomposition. The trend is then modified and the series is recomposed.
    Time series: agnostic
    Context: synthetic
    """

    def __init__(
        self,
        modified_component: str = None,
        fixed_config: dict = None,
        seed: int = None,
    ):
        assert (
            modified_component is not None
        ), "The modification parameter must be provided. 'trend', 'seasonal', or 'residual' are valid options."
        self.modified_component = modified_component
        super().__init__(seed=seed, fixed_config=fixed_config)

    def random_instance(self):
        # load dataset
        datasets = ["electricity_hourly"]
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
            start_hour = self.random.randint(0, 24 - 1)
            duration = self.random.randint(0, 24 - start_hour)
            start_time = f"{start_hour:02d}:00"
            end_time = f"{(start_hour + duration):02d}:00"

            window.index = window.index.to_timestamp()
            history_series.index = history_series.index.to_timestamp()
            future_series.index = future_series.index.to_timestamp()
            ground_truth = future_series.copy()
            # STL decomposition
            stl = STL(window, period=24)

            future_series_trend = stl.fit().trend[-metadata.prediction_length :]
            future_series_seasonal = stl.fit().seasonal[-metadata.prediction_length :]
            future_series_resid = stl.fit().resid[-metadata.prediction_length :]

            # modify the appropriate parameter

            if self.modified_component == "trend":
                # trend = stl.fit().trend
                trend_modification_min = -1
                trend_modification_max = 1
                # pick trend modification from uniform distribution between -1 and 1
                trend_modification = random.uniform(
                    trend_modification_min, trend_modification_max
                )

                modified_trend = future_series_trend.copy()

                modified_trend.loc[
                    future_series_trend.between_time(start_time, end_time).index
                ] = (
                    future_series_trend.loc[
                        future_series_trend.between_time(start_time, end_time).index
                    ]
                    * trend_modification
                )

                # Recompose the series, modifying the trend from start_time to end_time
                future_series = stl.fit().seasonal + modified_trend + stl.fit().resid

                background = f"The trend of the series will be multiplied by {trend_modification} between {start_time} and {end_time}."

            elif self.modified_component == "seasonal":
                # seasonal = stl.fit().seasonal
                seasonal_modification_min = -1
                seasonal_modification_max = 1
                # pick seasonal modification from uniform distribution between -1 and 1
                seasonal_modification = random.uniform(
                    seasonal_modification_min, seasonal_modification_max
                )

                modified_seasonal = future_series_seasonal.copy()

                modified_seasonal.loc[
                    future_series_seasonal.between_time(start_time, end_time).index
                ] = (
                    future_series_seasonal.loc[
                        future_series_seasonal.between_time(start_time, end_time).index
                    ]
                    * seasonal_modification
                )

                # Recompose the series, modifying the seasonal component from start_time to end_time
                future_series = stl.fit().trend + modified_seasonal + stl.fit().resid

                background = f"The seasonal component of the series will be multiplied by {seasonal_modification} between {start_time} and {end_time}."

            elif self.modified_component == "residual":
                # resid = stl.fit().resid
                resid_modification_min = -1
                resid_modification_max = 1
                # pick residual modification from uniform distribution between -1 and 1
                resid_modification = random.uniform(
                    resid_modification_min, resid_modification_max
                )

                modified_resid = future_series_resid.copy()

                modified_resid.loc[
                    future_series_resid.between_time(start_time, end_time).index
                ] = (
                    future_series_resid.loc[
                        future_series_resid.between_time(start_time, end_time).index
                    ]
                    * resid_modification
                )

                # Recompose the series, modifying the residual component from start_time to end_time
                future_series = stl.fit().trend + stl.fit().seasonal + modified_resid

                background = f"The residual component of the series will be multiplied by {resid_modification} between {start_time} and {end_time}."

            self.past_time = history_series
            self.future_time = future_series
            self.constraints = None
            self.background = background
            self.scenario = None
            self.ground_truth = ground_truth
            self.trend = stl.fit().trend
            self.seasonal = stl.fit().seasonal
            self.residual = stl.fit().resid


class STLTrendMultiplierTask(STLMutliplierTask):
    """
    A task where the trend component of the series is multiplied by a random factor.
    Time series: agnostic
    Context: synthetic
    """

    def __init__(self, fixed_config: dict = None, seed: int = None):
        super().__init__(
            modified_component="trend", seed=seed, fixed_config=fixed_config
        )


class STLSeasonalMultiplierTask(STLMutliplierTask):
    """
    A task where the seasonal component of the series is multiplied by a random factor.
    Time series: agnostic
    Context: synthetic
    """

    def __init__(self, fixed_config: dict = None, seed: int = None):
        super().__init__(
            modified_component="seasonal", seed=seed, fixed_config=fixed_config
        )


class STLResidualMultiplierTask(STLMutliplierTask):
    """
    A task where the residual component of the series is multiplied by a random factor.
    Time series: agnostic
    Context: synthetic
    """

    def __init__(self, fixed_config: dict = None, seed: int = None):
        super().__init__(
            modified_component="residual", seed=seed, fixed_config=fixed_config
        )
