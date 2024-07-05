"""
Tasks where the output forecast must be censored by some constraint.
Since the exact output distribution can be known, we use the Wasserstein
distance of the forecast at each time point as the metric,
normalized by what you would obtained ignoring the contraint.
"""

import scipy
import numpy as np
import pandas as pd

from .base import BaseTask


class ConstrainedRandomWalk(BaseTask):
    def __init__(
        self,
        seed: int = None,
        variance: float = 1.0,
        trend: float = 0.0,
        start_value: float = 0.0,
        constraint_less_than: bool = False,
        constraint_value: float = 0.0,
        num_hist_values: int = 20,
        num_pred_values: int = 10,
        num_samples: int = 1000,
    ):
        self.variance = variance
        self.trend = trend
        self.start_value = start_value  # Value of the last point in the history
        self.constraint_less_than = constraint_less_than
        self.constraint_value = constraint_value
        self.num_hist_values = num_hist_values
        self.num_pred_values = num_pred_values
        self.num_samples = num_samples

        super().__init__(seed=seed, fixed_config=None)

    def random_instance(self):
        """
        Generate a random instance of the task and instantiate its data
        """
        hist_steps = (
            np.sqrt(self.variance) * self.random.randn(self.num_hist_values)
            + self.trend
        )
        hist_values = np.cumsum(hist_steps)
        hist_values = hist_values - hist_values[-1] + self.start_value

        pred_steps = (
            np.sqrt(self.variance)
            * self.random.randn(self.num_samples, self.num_pred_values)
            + self.trend
        )
        pred_values = np.cumsum(pred_steps, axis=1) + self.start_value

        if self.constraint_less_than:
            const_hist_values = hist_values.clip(max=self.constraint_value)
            const_pred_values = pred_values.clip(max=self.constraint_value)
            constraints = f"Value <= {self.constraint_value}"
        else:
            const_hist_values = hist_values.clip(min=self.constraint_value)
            const_pred_values = pred_values.clip(min=self.constraint_value)
            constraints = f"Value >= {self.constraint_value}"

        # Convert the constrained series to Pandas series, for compatibility with what is in misleading_history
        # TODO: Randomly select frequency + starting date
        # inclusive="left" drops the last value from the index
        history_series = pd.Series(
            data=const_hist_values,
            index=pd.date_range(
                end="2010-01-01",
                freq="D",
                periods=self.num_hist_values + 1,
                inclusive="left",
            ),
        )
        future_series = pd.Series(
            data=const_pred_values[0, :],
            index=pd.date_range(
                start="2010-01-01",
                freq="D",
                periods=self.num_pred_values,
                inclusive="both",
            ),
        )

        # Values required for the evaluation,
        # regenerating the non_const forecast to allow measuring the distance between both
        self.perfect_const_forecast = const_pred_values
        pred_steps = (
            np.sqrt(self.variance)
            * self.random.randn(self.num_samples, self.num_pred_values)
            + self.trend
        )
        pred_values = np.cumsum(pred_steps, axis=1) + self.start_value
        self.perfect_non_const_forecast = pred_values

        # Instantiate the class variables
        self.past_time = history_series
        self.future_time = future_series
        self.constraints = constraints
        self.background = None
        self.scenario = None

    def evaluate(self, samples: np.ndarray):
        """
        Evaluate success based on samples from the inferred distribution

        Parameters:
        -----------
        samples: np.ndarray, shape (n_samples, n_time)
            Samples from the inferred distribution

        Returns:
        --------
        metric: float
            Metric value

        """
        cum_metric = 0.0
        cum_naive = 0.0

        for t_idx in range(self.num_pred_values):
            cum_metric += scipy.stats.wasserstein_distance(
                samples[:, t_idx], self.perfect_const_forecast[:, t_idx]
            )
            cum_naive += scipy.stats.wasserstein_distance(
                self.perfect_non_const_forecast[:, t_idx],
                self.perfect_const_forecast[:, t_idx],
            )

        print(cum_metric / self.num_pred_values, cum_naive / self.num_pred_values)
        return (cum_metric - cum_naive) / self.num_pred_values
