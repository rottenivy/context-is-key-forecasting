import numpy as np
import ast

import pandas as pd
import os

from benchmark.base import UnivariateCRPSTask

from benchmark.data.pems import (
    # download_lane_closure_files,
    # LANE_CLOSURE_SPLIT_PATH,
    # LANE_CLOSURE_SENSOR_PATH,
    # SLOW_FWYS_FILEPATH,
    # UNINTERESTING_FILES_PATH,
    download_instances,
    INSTANCES_DIR,
)

# from ..window_selection import (
#     # difference_in_means_is_large,
#     intersection_over_union_is_low,
#     quartile_intersection_over_union_is_low,
#     median_absolute_deviation_intersection_is_low,
# )


def has_structural_break(before_data, during_data, after_data, threshold=0.2):
    """
    Checks for structural breaks by comparing the EMA of Lane 1 Speed (mph) before, during, and after the interval.
    Returns True if there's a significant difference, indicating a structural break.
    """
    # Ensure there's enough data to compute the EMA
    if len(before_data) < 2 or len(during_data) < 2 or len(after_data) < 2:
        return False

    min_before = before_data.min()
    min_during = during_data.min()
    min_after = after_data.min()

    ema_before = before_data.ewm(span=12, adjust=False).mean().iloc[-1]
    ema_during = during_data.ewm(span=12, adjust=False).mean().iloc[-1]
    ema_after = after_data.ewm(span=12, adjust=False).mean().iloc[-1]

    # Check if the EMA difference exceeds the threshold
    conditions = [
        # ema_before - ema_during > threshold * ema_during,
        # ema_after - ema_during > threshold * ema_during,
        (1 - threshold) * min_before > min_during,
        (1 - threshold) * min_after > min_during,
    ]
    if all(conditions):
        return True
    return False


class DefaultLaneClosureTrafficTask(UnivariateCRPSTask):
    """
    Default traffic task, randomly sampled lane closure with all windows available.
    """

    _context_sources = UnivariateCRPSTask._context_sources + ["c_i"]
    _skills = UnivariateCRPSTask._skills + ["reasoning: deduction"]
    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(
        self,
        fixed_config: dict = None,
        seed: int = None,
        target: str = "Speed (mph)",  # or 'Occupancy (%)'
    ):
        self.init_data()
        self.target = target
        self.seed = seed
        super().__init__(seed=seed, fixed_config=fixed_config)

    def init_data(self):
        """
        Check integrity of data files and download if needed.

        """
        if not os.path.exists(INSTANCES_DIR):
            download_instances()

    def random_instance(self):
        # glob all sensor files

        lane_closure, window_df = self.get_instance()
        window = window_df[self.target]

        # Set the start and end of lane closure (round to the start of the day)
        self.lane_closure_start = pd.to_datetime(lane_closure["Start Date"]).normalize()
        lane_closure_duration = pd.to_timedelta(
            lane_closure["Reported Duration"], unit="m"
        )
        self.lane_closure_end = (
            self.lane_closure_start + lane_closure_duration
        ).normalize()

        # Calculate extended end of closure day (+2 days)
        end_of_closure_day_extended = self.lane_closure_end.normalize() + pd.DateOffset(
            days=2
        )

        # Define the start of the history window (7 days before the closure start day)
        history_start_day = self.lane_closure_start - pd.DateOffset(days=7)

        # Convert window index to datetime for filtering
        window_index = pd.to_datetime(window.index)

        # History series: From 7 days before closure start to just before the closure start day
        history_series = window[
            (window_index >= history_start_day)
            & (window_index < self.lane_closure_start)
        ]

        # Future series: From the start of the closure day to the extended end of the closure day (+2 days after closure end)
        future_series = window[
            (window_index >= self.lane_closure_start)
            & (window_index <= end_of_closure_day_extended)
        ]

        self.past_time = history_series.to_frame()
        self.future_time = future_series.to_frame()
        self.constraints = None
        self.background = self.get_context(window_df, lane_closure)
        self.scenario = None

    def get_history_factor(self):
        return 7

    def get_prediction_length(self, window):
        # Calculate the number of hours between the end of the window and the start of the lane closure day
        lane_closure_start = self.lane_closure_start
        lane_closure_end = self.lane_closure_end
        start_of_closure_day = pd.to_datetime(lane_closure_start).normalize()
        end_of_closure_day = pd.to_datetime(lane_closure_end).normalize()

        end_of_closure_day_extended = pd.to_datetime(
            end_of_closure_day
        ).normalize() + pd.DateOffset(days=2)

        # The prediction length is the number of hours from the start of the closure day to the end of the window
        prediction_length = (
            window.index[-1] - end_of_closure_day_extended
        ).total_seconds() / 3600

        return int(prediction_length)

    def get_instance(self):

        # Load the lane closure data
        instance_files = [f for f in os.listdir(INSTANCES_DIR) if "lane_closure" in f]
        abs_pms = [f.split("_abs_pm_")[1].split("_")[0] for f in instance_files]

        # sort the abs_pms
        abs_pms = sorted(abs_pms)
        selected_abs_pm = abs_pms[self.seed % len(abs_pms)]

        # Load the lane closure data
        lane_closure_file = [
            f for f in instance_files if f"abs_pm_{selected_abs_pm}" in f
        ][0]

        lane_closure = pd.read_csv(os.path.join(INSTANCES_DIR, lane_closure_file))

        # Load the sensor data
        sensor_file = lane_closure_file.replace("lane_closure", "sensor_window")
        sensor_data = pd.read_csv(os.path.join(INSTANCES_DIR, sensor_file))
        sensor_data["date"] = pd.to_datetime(sensor_data["date"])
        sensor_data.set_index("date", inplace=True)

        return lane_closure.iloc[0], sensor_data

    def is_interesting_window(self, window, lane_closure):
        """
        Check if the window is interesting for the task.
        """
        lane_closure_start = pd.to_datetime(lane_closure["Start Date"])
        lane_closure_end = pd.to_datetime(lane_closure["End Date"])
        window_target = window[self.target]

        if window_target.empty:
            return False

        # Extract the lane closure series
        lane_closure_series = window_target[
            (window_target.index >= lane_closure_start)
            & (window_target.index <= lane_closure_end)
        ]

        before_lane_closure_series = window_target[
            window_target.index < lane_closure_start
        ]

        after_lane_closure_series = window_target[
            window_target.index > lane_closure_end
        ]

        if has_structural_break(
            before_lane_closure_series, lane_closure_series, after_lane_closure_series
        ):
            return True

        return False

    def get_context(self, sensor_data, lane_closure):
        """
        Get the context of the task.
        """
        freeway_dir = sensor_data["Fwy"].iloc[0]
        district = sensor_data["District"].iloc[0]
        county = sensor_data["County"].iloc[0]
        abs_pm = sensor_data["Abs PM"].iloc[0]

        expected_start_date = pd.to_datetime(lane_closure["Start Date"])
        expected_end_date = pd.to_datetime(lane_closure["End Date"])
        expected_duration = str(lane_closure["Planned Duration"]) + " minutes"

        closure_lanes = lane_closure["Closure Lanes"]
        total_lanes = lane_closure["Total Lanes"]

        return (
            f"A lane closure is planned on freeway {freeway_dir} in district {district} "
            f"in county {county} at absolute postmile marker {abs_pm}. "
            f"The lane closure is expected to start on {expected_start_date} and end on "
            f"{expected_end_date} with a planned duration of {expected_duration}. "
            f"The number of lanes closed is {closure_lanes} out of {total_lanes} total lanes."
        )

    @property
    def seasonal_period(self) -> int:
        """
        This returns the period which should be used by statistical models for this task.
        If negative, this means that the data either has no period, or the history is shorter than the period.
        """
        return 7


__TASKS__ = [DefaultLaneClosureTrafficTask]
