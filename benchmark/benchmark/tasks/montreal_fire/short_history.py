import numpy as np

from calendar import month_name
from scipy import stats

from .utils import calculate_yearly_sum_stats_for_months
from ...base import UnivariateCRPSTask
from ...data.montreal_fire.load import (
    get_incident_log,
    get_time_count_series_by_borough,
)


class MontrealFireShortHistoryTask(UnivariateCRPSTask):
    """
    A task that requires forecasting the number of incidents of a specific type in Montreal based on a short history of past incidents.

    Parameters:
    -----------
    seed : int, optional
        The random seed to use.
    fixed_config : dict, optional
        A dictionary of fixed configuration parameters.
    series : str, default 'Field fire'
        The type of incident to forecast.
    history_start_month : int, default 12
        The month in which the historical data starts.
    history_length : int, default 6
        The number of months of historical data to use.
    include_max_month : bool, default True
        Whether to include information about the month with the most incidents in the past.

    """

    _context_sources = UnivariateCRPSTask._context_sources + ["c_i"]
    _skills = UnivariateCRPSTask._skills + ["reasoning: deduction"]
    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(
        self,
        seed=None,
        fixed_config=None,
        series="Field fire",
        history_start_month=12,
        history_length=6,
        include_max_month=True,
    ):
        self.series = series
        self.history_start_month = history_start_month
        self.history_length = history_length
        self.include_max_month = include_max_month
        super().__init__(seed=seed, fixed_config=fixed_config)

    @property
    def seasonal_period(self):
        return -1

    def _get_data(self):
        incidents = get_incident_log()
        return get_time_count_series_by_borough(incidents, self.series, frequency="M")

    def random_instance(self):
        series = self._get_data()["total"]

        valid_start_dates = [
            date
            for date in series.loc[series.index.month == self.history_start_month].index
            if len(series[date:]) >= 12
        ]

        history_start_date = self.random.choice(valid_start_dates)
        history_end_date = history_start_date + self.history_length - 1
        forecast_start_date = history_end_date + 1
        forecast_end_date = forecast_start_date + 12 - self.history_length

        self.past_time = series[history_start_date:history_end_date].to_frame()
        self.future_time = series[forecast_start_date:forecast_end_date].to_frame()

        self.background = f'This series contains the number of "{self.series}" incidents responded to by the Montreal Fire Department.'

        # Get the other windows to calculate stats
        other = list(valid_start_dates)
        other.remove(history_start_date)

        # Calculate the total number of incidents
        count_per_year = calculate_yearly_sum_stats_for_months(
            series, list(range(13)), cutoff_year=2024
        )["values"]
        del count_per_year[history_start_date.year]
        del count_per_year[forecast_end_date.year]
        self.background += f" In other years, the average number of incidents was {np.mean(list(count_per_year.values())):.0f}"

        if self.include_max_month:
            max_month = stats.mode(
                [series[o : o + 12].idxmax().month for o in other]
            ).mode
            self.background += (
                f" and the month with the most incidents was {month_name[max_month]}."
            )
        else:
            self.background += "."


class MontrealFireFieldFireExplicitShortHistoryTask(MontrealFireShortHistoryTask):
    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(self, seed=None, fixed_config=None):
        super().__init__(
            seed=seed,
            fixed_config=fixed_config,
            series="Field fire",
            history_start_month=12,
            history_length=6,
            include_max_month=True,
        )


class MontrealFireFieldFireImplicitShortHistoryTask(MontrealFireShortHistoryTask):
    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(self, seed=None, fixed_config=None):
        super().__init__(
            seed=seed,
            fixed_config=fixed_config,
            series="Field fire",
            history_start_month=12,
            history_length=6,
            include_max_month=False,
        )


class MontrealFireTrashFireExplicitShortHistoryTask(MontrealFireShortHistoryTask):
    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(self, seed=None, fixed_config=None):
        super().__init__(
            seed=seed,
            fixed_config=fixed_config,
            series="Trash fire",
            history_start_month=12,
            history_length=6,
            include_max_month=True,
        )


class MontrealFireTrashFireImplicitShortHistoryTask(MontrealFireShortHistoryTask):
    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(self, seed=None, fixed_config=None):
        super().__init__(
            seed=seed,
            fixed_config=fixed_config,
            series="Trash fire",
            history_start_month=12,
            history_length=6,
            include_max_month=False,
        )


class MontrealFireNauticalRescueExplicitShortHistoryTask(MontrealFireShortHistoryTask):
    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(self, seed=None, fixed_config=None):
        super().__init__(
            seed=seed,
            fixed_config=fixed_config,
            series="Nautical rescue",
            history_start_month=12,
            history_length=6,
            include_max_month=True,
        )


class MontrealFireNauticalRescueImplicitShortHistoryTask(MontrealFireShortHistoryTask):
    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(self, seed=None, fixed_config=None):
        super().__init__(
            seed=seed,
            fixed_config=fixed_config,
            series="Nautical rescue",
            history_start_month=12,
            history_length=6,
            include_max_month=False,
        )


class MontrealFireIceRescueExplicitShortHistoryTask(MontrealFireShortHistoryTask):
    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(self, seed=None, fixed_config=None):
        super().__init__(
            seed=seed,
            fixed_config=fixed_config,
            series="Ice rescue",
            history_start_month=7,
            history_length=6,
            include_max_month=True,
        )


class MontrealFireIceRescueImplicitShortHistoryTask(MontrealFireShortHistoryTask):
    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(self, seed=None, fixed_config=None):
        super().__init__(
            seed=seed,
            fixed_config=fixed_config,
            series="Ice rescue",
            history_start_month=7,
            history_length=6,
            include_max_month=False,
        )


__TASKS__ = [
    MontrealFireFieldFireExplicitShortHistoryTask,
    MontrealFireFieldFireImplicitShortHistoryTask,
    MontrealFireTrashFireExplicitShortHistoryTask,
    MontrealFireTrashFireImplicitShortHistoryTask,
    MontrealFireNauticalRescueExplicitShortHistoryTask,
    MontrealFireNauticalRescueImplicitShortHistoryTask,
    MontrealFireIceRescueExplicitShortHistoryTask,
    MontrealFireIceRescueImplicitShortHistoryTask,
]
