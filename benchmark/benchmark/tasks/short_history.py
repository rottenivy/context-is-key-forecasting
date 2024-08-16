"""
Tasks where the given history is too short to accurately do the forecasting,
but context is given to allow the model to fill in the missing information.
"""

import datetime
import pandas as pd
from typing import Optional
from abc import abstractmethod

from gluonts.dataset.util import to_pandas
from gluonts.dataset.repository import get_dataset

from ..base import UnivariateCRPSTask


class BaseHalfDaySolarForecastTask(UnivariateCRPSTask):
    """
    A task where the model is tasked to forecast the second half of the day,
    for a dataset which has a strong daily periodicity.
    The task comes with a textual description which can helps the model
    learns the daily shape of the signal.
    """

    def random_instance(self):
        dataset = get_dataset("solar_10_minutes", regenerate=False)

        # Select a random time series
        ts_index = self.random.choice(len(dataset.test))
        full_series = to_pandas(list(dataset.test)[ts_index])

        # The solar_10_minutes dataset is for the whole year of 2006.
        # Select a day for the forecast between 2006-07-01 (182nd day) and 2006-12-31 (365th day).
        day = self.random.randint(low=181, high=366)
        # Select the start of the forecast period, from 10:30 to 13:30.
        forecast_time = self.random.randint(low=10 * 6 + 3, high=13 * 6 + 3 + 1)
        # history = first 12 hours of the day, target = second 12 hours of the day
        full_history_series = full_series.iloc[: (day * 24 * 6)]
        history_series = full_series.iloc[
            (day * 24 * 6) : (day * 24 * 6) + forecast_time
        ]
        future_series = full_series.iloc[
            (day * 24 * 6) + forecast_time : (day * 24 * 6) + 24 * 6
        ]

        background = self.get_background(
            full_history_series, future_series.index[0].start_time
        )

        # Instantiate the class variables
        self.past_time = history_series.to_frame()
        self.future_time = future_series.to_frame()
        self.constraints = None
        self.background = background
        self.scenario = None

        # No RoI need to be defined as the full prediction window is important

    @abstractmethod
    def get_background(
        self, full_history_series: pd.Series, forecast_date: pd.Timestamp
    ) -> str:
        """
        Generate a textual hint for the model to know about the potential shape of the daily solar intensity.
        """
        pass

    @property
    def seasonal_period(self) -> int:
        """
        This returns the period which should be used by statistical models for this task.
        If negative, this means that the data either has no period, or the history is shorter than the period.
        """
        # Not enough history for a single period
        return -1


class MinimalInfoHalfDaySolarForecastTask(BaseHalfDaySolarForecastTask):
    """
    Version of the task where only the minimal background information is given.
    """

    _context_sources = ["c_i"]
    _skills = BaseHalfDaySolarForecastTask._skills + ["reasoning: deduction"]

    def get_background(
        self, full_history_series: pd.Series, forecast_date: pd.Timestamp
    ) -> str:
        return (
            "This series contains the power production of a photovoltaic power plant."
        )


class LocaleInfoHalfDaySolarForecastTask(BaseHalfDaySolarForecastTask):
    """
    Version of the task where the state in which the data was collected is mentioned.
    """

    _context_sources = ["c_i"]
    _skills = BaseHalfDaySolarForecastTask._skills + ["reasoning: deduction"]

    def get_background(
        self, full_history_series: pd.Series, forecast_date: pd.Timestamp
    ) -> str:
        return "This series contains the power production of a photovoltaic power plant in the state of Alabama."


class ZenithInfoHalfDaySolarForecastTask(BaseHalfDaySolarForecastTask):
    """
    Version of the task where the average time at which the daily maximum is reached is mentioned.
    """

    _context_sources = ["c_i", "c_h"]
    _skills = BaseHalfDaySolarForecastTask._skills + ["reasoning: deduction"]

    def get_background(
        self, full_history_series: pd.Series, forecast_date: pd.Timestamp
    ) -> str:
        def func(s):
            p = s.idxmax()
            t = p.start_time.time()
            return 3600 * t.hour + 60 * t.minute + t.second + 1e-6 * t.microsecond

        # The resample is effectively a groupby based on the given frequency, so this gives us the mean
        # time of the day at which the series is the highest.
        past_90_days = full_history_series.iloc[-90 * 24 * 6 :]
        mean_zenith_seconds = round(past_90_days.resample("D").apply(func).mean())
        mean_zenith_formated = (
            datetime.datetime.fromtimestamp(mean_zenith_seconds)
            .time()
            .strftime("%H:%M:%S")
        )

        return (
            "This series contains the power production of a photovoltaic power plant in the state of Alabama.\n"
            f"Over the previous 90 days, the maximum power production happened on average at {mean_zenith_formated}."
        )


class TrafficForecastTaskwithHolidaysInPredictionWindow(UnivariateCRPSTask):
    """
    Forecasting task based on the Monash traffic dataset, with frequency hourly, where the history is 7 days (168 timesteps) and prediction is 3 days (72 days), and windows are chosen such that there is a holiday in the middle of the prediction window.
    """

    def __init__(self, seed: int = None, fixed_config: Optional[dict] = None):
        # Holidays with observable differences from the preceeding 7 days
        self.holidays = [
            (datetime.date(2015, 5, 25), "Memorial Day"),
            (datetime.date(2015, 7, 4), "Independence Day"),
            (datetime.date(2015, 9, 7), "Labor Day"),
            (datetime.date(2015, 11, 11), "Veterans Day"),
            (datetime.date(2015, 11, 26), "Thanksgiving"),
            (datetime.date(2015, 12, 25), "Christmas Day"),
            (datetime.date(2016, 5, 30), "Memorial Day"),
            (datetime.date(2016, 7, 4), "Independence Day"),
            (datetime.date(2016, 9, 5), "Labor Day"),
            (datetime.date(2016, 11, 11), "Veterans Day"),
            (datetime.date(2016, 11, 24), "Thanksgiving"),
            (datetime.date(2016, 12, 25), "Christmas Day"),
        ]

        super().__init__(seed=seed, fixed_config=fixed_config)

    def random_instance(self):
        dataset = get_dataset("traffic", regenerate=False)

        # Select a random time series
        ts_index = self.random.choice(len(dataset.test))
        full_series = to_pandas(list(dataset.test)[ts_index])

        # The traffic dataset is for the whole years of 2015 and 2016, and is hourly.
        # Get rid of the first (potentially) incomplete week, to always start on Monday.
        first_day_of_week = full_series.index[0].day_of_week
        full_series = full_series.iloc[(7 - first_day_of_week) * 24 :]

        # Select a random holiday
        holiday_date, holiday_name = self.holidays[
            self.random.choice(len(self.holidays))
        ]
        holiday_datetime = pd.to_datetime(holiday_date)
        holiday_index = full_series.index.get_loc(holiday_datetime)

        # Prediction could start with this or end with this (both of which have length 2), or contain this in the middle (length 3)
        # Here I implement starting with
        # History could be as long as possible
        history_series = full_series.iloc[
            holiday_index - (24 * 8) : holiday_index - (24 * 1)
        ]
        future_series = full_series.iloc[
            holiday_index - (24 * 1) : holiday_index + (24 * 2)
        ]

        background = self.get_background(future_series, holiday_date, holiday_name)

        # Instantiate the class variables
        self.past_time = history_series.to_frame()
        self.future_time = future_series.to_frame()
        self.constraints = None
        self.background = background
        self.scenario = None

        # RoI is the holiday
        self.region_of_interest = slice(holiday_index, holiday_index + 24)

    @abstractmethod
    def get_background(self, future_series, holiday_date, holiday_name):
        pass


class ImplicitTrafficForecastTaskwithHolidaysInPredictionWindow(
    TrafficForecastTaskwithHolidaysInPredictionWindow
):
    """
    TrafficForecastTaskwithHolidaysInPredictionWindow with only intemporal information given as context
    """

    _context_sources = UnivariateCRPSTask._context_sources + ["c_i"]
    _skills = UnivariateCRPSTask._skills + ["reasoning: deduction"]

    def get_background(self, future_series, holiday_date, holiday_name):
        background = "This series contains the road occupancy rates on a freeway in the San Francisco Bay area."
        return background


class ExplicitTrafficForecastTaskwithHolidaysInPredictionWindow(
    TrafficForecastTaskwithHolidaysInPredictionWindow
):
    """
    TrafficForecastTaskwithHolidaysInPredictionWindow with intemporal information and holiday date and name given as context
    """

    _context_sources = UnivariateCRPSTask._context_sources + ["c_i", "c_f"]
    _skills = UnivariateCRPSTask._skills + ["reasoning: deduction"]

    def get_background(self, future_series, holiday_date, holiday_name):
        background = "This series contains the road occupancy rates on a freeway in the San Francisco Bay area."
        background += f" Note that {holiday_date} is a holiday due to {holiday_name}."
        return background


class ExplicitWithDaysTrafficForecastTaskwithHolidaysInPredictionWindow(
    TrafficForecastTaskwithHolidaysInPredictionWindow
):
    """
    TrafficForecastTaskwithHolidaysInPredictionWindow with intemporal information, holiday date and name, and days for which forecast has to be made given as context
    """

    _context_sources = UnivariateCRPSTask._context_sources + ["c_i", "c_f"]
    _skills = UnivariateCRPSTask._skills + ["reasoning: deduction"]

    def get_background(self, future_series, holiday_date, holiday_name):
        background = "This series contains the road occupancy rates on a freeway in the San Francisco Bay area."
        background += " The days for which the forecast is required are "
        idx = 0
        while idx < len(future_series):
            day = future_series.index[idx]
            day_name = day.to_timestamp().day_name()
            background += f"{day_name}, "
            idx += 24
        background = background[:-2] + " respectively."
        background += f" Note that {holiday_date} is a holiday due to {holiday_name}."
        return background


class ExplicitWithDatesAndDaysTrafficForecastTaskwithHolidaysInPredictionWindow(
    TrafficForecastTaskwithHolidaysInPredictionWindow
):
    """
    TrafficForecastTaskwithHolidaysInPredictionWindow with intemporal information, holiday date and name, days referencing their dates for which forecast has to be made given as context
    """

    _context_sources = UnivariateCRPSTask._context_sources + ["c_i", "c_f"]
    _skills = UnivariateCRPSTask._skills + ["reasoning: deduction"]

    def get_background(self, future_series, holiday_date, holiday_name):
        background = "This series contains the road occupancy rates on a freeway in the San Francisco Bay area."
        background += " The days for which the forecast is required are "
        idx = 0
        while idx < len(future_series):
            day = future_series.index[idx]
            date = day.to_timestamp().date()
            day_name = day.to_timestamp().day_name()
            background += f"{date} ({day_name}), "
            idx += 24
        background = background[:-2] + " respectively."
        background += f" Note that {holiday_date} is a holiday due to {holiday_name}."
        return background


class TrafficForecastTaskwithHolidaysInPredictionWindowAndPastYearAnalogy(
    UnivariateCRPSTask
):
    """
    Forecasting task based on the Monash traffic dataset, with frequency hourly, where the history is 7 days (168 timesteps) and prediction is 3 days (72 days), and windows are chosen such that there is a holiday in the start of the prediction window. Only 2016 holidays (6 holidays) are put in the prediction window, and an analogy to 2015 is also given. The analogy has the mean and max difference between the preceeding 7 days and the holiday in 2015.
    NOTE: The assumption that the difference in traffic pattern between the preceeding 7 days and the holiday would be the same between 2015 and 2016 is made here.
    """

    def __init__(self, seed: int = None, fixed_config: Optional[dict] = None):
        # Holidays with observable differences from the preceeding 7 days
        self.holidays_2015 = [
            (datetime.date(2015, 5, 25), "Memorial Day"),
            (datetime.date(2015, 7, 4), "Independence Day"),
            (datetime.date(2015, 9, 7), "Labor Day"),
            (datetime.date(2015, 11, 11), "Veterans Day"),
            (datetime.date(2015, 11, 26), "Thanksgiving"),
            (datetime.date(2015, 12, 25), "Christmas Day"),
        ]
        self.holidays_2016 = [
            (datetime.date(2016, 5, 30), "Memorial Day"),
            (datetime.date(2016, 7, 4), "Independence Day"),
            (datetime.date(2016, 9, 5), "Labor Day"),
            (datetime.date(2016, 11, 11), "Veterans Day"),
            (datetime.date(2016, 11, 24), "Thanksgiving"),
            (datetime.date(2016, 12, 25), "Christmas Day"),
        ]

        super().__init__(seed=seed, fixed_config=fixed_config)

    def random_instance(self):
        dataset = get_dataset("traffic", regenerate=False)

        # Select a random time series
        ts_index = self.random.choice(len(dataset.test))
        full_series = to_pandas(list(dataset.test)[ts_index])

        # The traffic dataset is for the whole years of 2015 and 2016, and is hourly.
        # Get rid of the first (potentially) incomplete week, to always start on Monday.
        first_day_of_week = full_series.index[0].day_of_week
        full_series = full_series.iloc[(7 - first_day_of_week) * 24 :]

        # Select a random holiday
        holiday_choice = self.random.choice(len(self.holidays_2016))
        holiday_date, holiday_name = self.holidays_2016[holiday_choice]
        holiday_datetime = pd.to_datetime(holiday_date)
        holiday_index = full_series.index.get_loc(holiday_datetime)

        # Prediction could start with this or end with this (both of which have length 2), or contain this in the middle (length 3)
        # Here I implement starting with
        # History could be as long as possible
        history_series = full_series.iloc[holiday_index - (24 * 7) : holiday_index]
        future_series = full_series.iloc[holiday_index : holiday_index + (24 * 3)]

        # Get past year data
        past_year_holiday_date, _ = self.holidays_2015[holiday_choice]
        past_year_holiday_datetime = pd.to_datetime(past_year_holiday_date)
        past_year_holiday_index = full_series.index.get_loc(past_year_holiday_datetime)
        past_year_history_series = full_series.iloc[
            past_year_holiday_index - (24 * 7) : past_year_holiday_index
        ]
        past_year_future_series = full_series.iloc[
            past_year_holiday_index : past_year_holiday_index + (24 * 3)
        ]

        background = self.get_background(
            future_series, holiday_date, holiday_name
        ) + self.get_analogy_to_past_year(
            past_year_history_series, past_year_future_series
        )

        # Instantiate the class variables
        self.past_time = history_series.to_frame()
        self.future_time = future_series.to_frame()
        self.constraints = None
        self.background = background
        self.scenario = None

        # RoI is the holiday
        self.region_of_interest = slice(holiday_index, holiday_index + 24)

    @abstractmethod
    def get_background(self, future_series, holiday_date, holiday_name):
        pass

    def get_analogy_to_past_year(
        self, past_year_history_series, past_year_future_series
    ):
        # POTENTIAL IMPROVEMENT 1: Instead of using all 7 days, the weekdays out of the 7 days can be used
        # POTENTIAL IMPROVEMENT 2: The difference between the holiday and 7th day before the holiday can be provided (same day of the week, 1 week before)

        mean_history_traffic_last_year = past_year_history_series.mean()
        mean_holiday_traffic_last_year = past_year_future_series[:24].mean()
        mean_traffic_percentage_difference = (
            mean_holiday_traffic_last_year / mean_history_traffic_last_year
        )
        max_history_traffic_last_year = past_year_history_series.max()
        max_holiday_traffic_last_year = past_year_future_series[:24].max()
        max_traffic_percentage_difference = (
            max_holiday_traffic_last_year / max_history_traffic_last_year
        )
        analogy_background = f" In 2015, the mean traffic on the holiday was {mean_traffic_percentage_difference:.4f} times the mean traffic in the preceeding 7 days before the holiday, and traffic in the busiest hour on the holiday was {max_traffic_percentage_difference:.4f} times the traffic in the preceeding 7 days before the holiday."
        return analogy_background


class ExplicitTrafficForecastTaskwithHolidaysInPredictionWindowAndPastYearAnalogy(
    TrafficForecastTaskwithHolidaysInPredictionWindowAndPastYearAnalogy
):
    """
    ExplicitTrafficForecastTaskwithHolidaysInPredictionWindowAndPastYearAnalogy with intemporal information and holiday date and name given as context
    """

    _context_sources = UnivariateCRPSTask._context_sources + ["c_i", "c_f"]
    _skills = UnivariateCRPSTask._skills + [
        "reasoning: deduction",
        "reasoning: analogy",
    ]

    def get_background(self, future_series, holiday_date, holiday_name):
        background = "This series contains the road occupancy rates on a freeway in the San Francisco Bay area."
        background += f" Note that {holiday_date} is a holiday due to {holiday_name}."

        return background


class ExplicitWithDaysTrafficForecastTaskwithHolidaysInPredictionWindowAndPastYearAnalogy(
    TrafficForecastTaskwithHolidaysInPredictionWindowAndPastYearAnalogy
):
    """
    TrafficForecastTaskwithHolidaysInPredictionWindowAndPastYearAnalogy with intemporal information, holiday date and name, and days for which forecast has to be made given as context=
    """

    _context_sources = UnivariateCRPSTask._context_sources + ["c_i", "c_f"]
    _skills = UnivariateCRPSTask._skills + [
        "reasoning: deduction",
        "reasoning: analogy",
    ]

    def get_background(self, future_series, holiday_date, holiday_name):
        background = "This series contains the road occupancy rates on a freeway in the San Francisco Bay area."
        background += " The days for which the forecast is required are "
        idx = 0
        while idx < len(future_series):
            day = future_series.index[idx]
            day_name = day.to_timestamp().day_name()
            background += f"{day_name}, "
            idx += 24
        background = background[:-2] + " respectively."
        background += f" Note that {holiday_date} is a holiday due to {holiday_name}."
        return background


class ExplicitWithDatesAndDaysTrafficForecastTaskwithHolidaysInPredictionWindowAndPastYearAnalogy(
    TrafficForecastTaskwithHolidaysInPredictionWindowAndPastYearAnalogy
):
    """
    TrafficForecastTaskwithHolidaysInPredictionWindowAndPastYearAnalogy with intemporal information, holiday date and name, days referencing their dates for which forecast has to be made given as context
    """

    _context_sources = UnivariateCRPSTask._context_sources + ["c_i", "c_f"]
    _skills = UnivariateCRPSTask._skills + [
        "reasoning: deduction",
        "reasoning: analogy",
    ]

    def get_background(self, future_series, holiday_date, holiday_name):
        background = "This series contains the road occupancy rates on a freeway in the San Francisco Bay area."
        background += " The days for which the forecast is required are "
        idx = 0
        while idx < len(future_series):
            day = future_series.index[idx]
            date = day.to_timestamp().date()
            day_name = day.to_timestamp().day_name()
            background += f"{date} ({day_name}), "
            idx += 24
        background = background[:-2] + " respectively."
        background += f" Note that {holiday_date} is a holiday due to {holiday_name}."
        return background


__TASKS__ = [
    MinimalInfoHalfDaySolarForecastTask,
    LocaleInfoHalfDaySolarForecastTask,
    ZenithInfoHalfDaySolarForecastTask,
    ImplicitTrafficForecastTaskwithHolidaysInPredictionWindow,
    ExplicitTrafficForecastTaskwithHolidaysInPredictionWindow,
    ExplicitWithDaysTrafficForecastTaskwithHolidaysInPredictionWindow,
    ExplicitWithDatesAndDaysTrafficForecastTaskwithHolidaysInPredictionWindow,
    ExplicitTrafficForecastTaskwithHolidaysInPredictionWindowAndPastYearAnalogy,
    ExplicitWithDaysTrafficForecastTaskwithHolidaysInPredictionWindowAndPastYearAnalogy,
    ExplicitWithDatesAndDaysTrafficForecastTaskwithHolidaysInPredictionWindowAndPastYearAnalogy,
]
