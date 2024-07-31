import pandas as pd
import numpy as np
import statsmodels
import statsmodels.tsa.tsatools
import statsmodels.tsa.holtwinters
import statsmodels.tsa.exponential_smoothing.ets
from typing import Literal

from .base import Baseline
from ..base import BaseTask


def get_seasonal_periods(task_instance) -> int:
    """
    Return the season length for the given instance, based on its data frequency.
    This reuses the same method used internally by statsmodels when given a series with a datetime index.
    """
    freq = task_instance.past_time.index.freq
    if not freq:
        # TODO Only look at the first 3 timesteps for now, due to some task instances having holes
        # We can remove this hack once these instances are fixed.
        freq = pd.infer_freq(task_instance.past_time.index[:3])
    try:
        return statsmodels.tsa.tsatools.freq_to_period(freq)
    except ValueError:
        # Hard-coded exception for the solar_10_minutes dataset,
        # since its 10 minutes frequency is unknown in statsmodels.
        return 6


class ExponentialSmoothingForecaster(Baseline):
    def __init__(
        self,
        trend: Literal["add", "mul", None] = "add",
        seasonal: Literal["add", "mul", None] = "add",
    ):
        super().__init__()

        self.trend = trend
        self.seasonal = seasonal

    def __call__(self, task_instance: BaseTask, n_samples: int) -> np.ndarray:
        return self.forecast(
            past_time=task_instance.past_time,
            future_time=task_instance.future_time,
            seasonal_periods=get_seasonal_periods(task_instance),
            n_samples=n_samples,
        )

    def forecast(
        self,
        past_time: pd.DataFrame,
        future_time: pd.DataFrame,
        seasonal_periods: int,
        n_samples: int,
    ) -> np.ndarray:
        """
        This method allows a forecast to be done without requiring a complete BaseTask instance.
        This is primarly meant to be called inside a BaseTask constructor when doing rejection sampling or similar approaches.
        """
        simulations_samples = []
        for column in past_time.columns:
            model = statsmodels.tsa.holtwinters.ExponentialSmoothing(
                endog=past_time[column],
                trend=self.trend,
                seasonal=self.seasonal,
                seasonal_periods=seasonal_periods,
            )

            result = model.fit()

            simulations = result.simulate(
                nsimulations=future_time.shape[0], repetitions=n_samples
            )
            simulations_samples.append(simulations.to_numpy().transpose())

        return np.stack(simulations_samples, axis=-1)

    @property
    def cache_name(self) -> str:
        args_to_include = ["trend", "seasonal"]
        return f"{self.__class__.__name__}_" + "_".join(
            [f"{k}={getattr(self, k)}" for k in args_to_include]
        )


class ETSModelForecaster(Baseline):
    def __init__(
        self,
        trend: Literal["add", "mul", None] = "add",
        seasonal: Literal["add", "mul", None] = "add",
        error: Literal["add", "mul"] = "add",
    ):
        super().__init__()

        self.trend = trend
        self.seasonal = seasonal
        self.error = error

    def __call__(self, task_instance: BaseTask, n_samples: int) -> np.ndarray:
        return self.forecast(
            past_time=task_instance.past_time,
            future_time=task_instance.future_time,
            seasonal_periods=get_seasonal_periods(task_instance),
            n_samples=n_samples,
        )

    def forecast(
        self,
        past_time: pd.DataFrame,
        future_time: pd.DataFrame,
        seasonal_periods: int,
        n_samples: int,
    ) -> np.ndarray:
        """
        This method allows a forecast to be done without requiring a complete BaseTask instance.
        This is primarly meant to be called inside a BaseTask constructor when doing rejection sampling or similar approaches.
        """
        simulations_samples = []
        for column in past_time.columns:
            model = statsmodels.tsa.exponential_smoothing.ets.ETSModel(
                endog=past_time[column],
                trend=self.trend,
                seasonal=self.seasonal,
                error=self.error,
                seasonal_periods=seasonal_periods,
            )

            # Avoid L-BFGS-B output spam
            result = model.fit(disp=False)

            simulations = result.simulate(
                nsimulations=future_time.shape[0], repetitions=n_samples
            )
            simulations_samples.append(simulations.to_numpy().transpose())

        return np.stack(simulations_samples, axis=-1)

    @property
    def cache_name(self) -> str:
        args_to_include = ["trend", "seasonal", "error"]
        return f"{self.__class__.__name__}_" + "_".join(
            [f"{k}={getattr(self, k)}" for k in args_to_include]
        )


def additive_exponential_smoothing(task_instance, n_samples=50):
    """
    A baseline is just some callable that receives a task instance and returns a prediction.
    """
    simulations_samples = []
    for column in task_instance.past_time.columns:
        model = statsmodels.tsa.holtwinters.ExponentialSmoothing(
            endog=task_instance.past_time[column],
            trend="add",
            seasonal="add",
            seasonal_periods=get_seasonal_periods(task_instance),
        )

        result = model.fit()

        simulations = result.simulate(
            nsimulations=task_instance.future_time.shape[0], repetitions=n_samples
        )
        simulations_samples.append(simulations.to_numpy().transpose())

    return np.stack(simulations_samples, axis=-1)
