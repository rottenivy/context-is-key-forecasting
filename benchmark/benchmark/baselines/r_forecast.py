"""
This file contains various baselines which use the R package "forecast".

This requires R to be installed to run:
conda install r-essentials r-forecast
"""

import pandas as pd
import numpy as np
import time

_rpy2_initialized = False
try:
    import rpy2.robjects.packages as rpackages
    from rpy2.robjects import numpy2ri, pandas2ri

    _rpy2_initialized = True
except:
    pass

from .base import Baseline
from ..base import BaseTask


class R_ETS(Baseline):

    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(
        self,
        model: str = "ZZZ",
    ):
        """
        Get predictions from the ETS model.

        Parameters:
        -----------
        model: str
            Which model to use, a 3 letters code for error, trend, and seasonality.
            The letters can be Z (automatic), A (additive), M (multiplicative) or N (none)
        """
        if not _rpy2_initialized:
            raise RuntimeError("The rpy2 package has not been successfully imported.")

        super().__init__()

        self.model = model

        # Required R packages:
        self._stats_pkg = None  # "stats"
        self._forecast_pkg = None  # "forecast"

    def __call__(self, task_instance: BaseTask, n_samples: int) -> np.ndarray:
        starting_time = time.time()
        samples = self.forecast(
            past_time=task_instance.past_time,
            future_time=task_instance.future_time,
            seasonal_periods=task_instance.seasonal_period,
            n_samples=n_samples,
        )
        extra_info = {
            "total_time": time.time() - starting_time,
        }
        return samples, extra_info

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

        Note: If seasonal_periods is <= 0, then we set the period to 1, which skips it if the model uses "Z" as its seasonal component.
              If the model uses "A" or "M", then it will fail.
        """
        # Lazy initialization, since otherwise the parallel dispatching using pickle will not work
        if self._stats_pkg is None:
            self._stats_pkg = rpackages.importr("stats")
        if self._forecast_pkg is None:
            self._forecast_pkg = rpackages.importr("forecast")

        history = pandas2ri.py2rpy(past_time[past_time.columns[-1]])
        ts = self._stats_pkg.ts(history, frequency=max(1, seasonal_periods))
        fit = self._forecast_pkg.ets(ts, model=self.model)

        samples = np.stack(
            [
                numpy2ri.rpy2py(
                    self._forecast_pkg.simulate_ets(fit, nsim=len(future_time))
                )
                for _ in range(n_samples)
            ]
        )
        samples = samples[:, :, None]

        return samples

    @property
    def cache_name(self) -> str:
        args_to_include = ["model"]
        return f"{self.__class__.__name__}_" + "_".join(
            [f"{k}={getattr(self, k)}" for k in args_to_include]
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        # Don't pickle the R packages
        state["_stats_pkg"] = None
        state["_forecast_pkg"] = None
        return state
