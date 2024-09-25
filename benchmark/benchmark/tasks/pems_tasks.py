import pandas as pd
import os

from benchmark.base import UnivariateCRPSTask

from benchmark.data.pems import (
    download_instances,
    INSTANCES_DIR,
)


freeway_dir_context = "The freeway is {freeway_dir}."
district_context = "The district is {district}."
county_context = "The county is {county}."
abs_pm_context = "The absolute postmile marker is {abs_pm}."

short_background = freeway_dir_context
medium_background = f"{freeway_dir_context} {district_context} {county_context}"
long_background = (
    f"{freeway_dir_context} {district_context} {county_context} {abs_pm_context}"
)

lane_closure_placements = ["before", "during", "after"]

# targets = ["Speed (mph)", "Occupancy (%)"]

history_factors = {
    "short": 1,
    "default": 7,
}


class AbstractLaneClosureTask(UnivariateCRPSTask):
    """
    Abstract traffic task, randomly sampled lane closure with all windows available.
    """

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

    def get_instance(self):

        # Load the lane closure data
        instance_files = [f for f in os.listdir(INSTANCES_DIR) if "lane_closure" in f]
        abs_pms = [f.split("_abs_pm_")[1].split("_")[0] for f in instance_files]

        # sort the abs_pms
        abs_pms = sorted(abs_pms)
        if self.seed is None:
            self.seed = self.random.randint(0, len(abs_pms))
        selected_abs_pm = abs_pms[self.seed % len(abs_pms)]

        # Load the lane closure data
        lane_closure_file = [
            f for f in instance_files if f"abs_pm_{selected_abs_pm}" in f
        ][
            0
        ]  # uses abs_pm as unique identifier

        lane_closure = pd.read_csv(os.path.join(INSTANCES_DIR, lane_closure_file))
        lane_closure = lane_closure.iloc[0]

        # Load the sensor data
        sensor_file = lane_closure_file.replace("lane_closure", "sensor_window")
        sensor_data = pd.read_csv(os.path.join(INSTANCES_DIR, sensor_file))
        sensor_data["date"] = pd.to_datetime(sensor_data["date"])
        sensor_data.set_index("date", inplace=True)
        sensor_data.index = pd.to_datetime(sensor_data.index)

        self.lane_closure_start = pd.to_datetime(lane_closure["Start Date"])
        self.lane_closure_end = self.lane_closure_start + pd.to_timedelta(
            lane_closure["Reported Duration"], unit="m"
        )

        return lane_closure, sensor_data

    def random_instance(self):
        lane_closure, window_df = self.get_instance()

        history_series, future_series = self.get_history_future(
            lane_closure, window_df[self.target]
        )

        self.past_time = history_series.to_frame()
        self.future_time = future_series.to_frame()
        self.constraints = None
        self.background = self.get_background(window_df)
        self.scenario = self.get_scenario(lane_closure)

    def get_history_factor(self):
        # if owning object class name has short, return 1, else return 7
        if "ShortHistory" in self.__class__.__name__:
            return history_factors["short"]
        return history_factors["default"]

    def get_history_future(self, lane_closure, window):

        lane_closure_start_day = self.lane_closure_start.normalize()
        lane_closure_end_day = self.lane_closure_end.normalize()

        if "LaneClosureBefore" in self.__class__.__name__:
            history_end_day = lane_closure_end_day + pd.DateOffset(days=1)
            history_start_day = lane_closure_start_day.normalize() - pd.DateOffset(
                days=self.get_history_factor()
            )
            history_series = window[
                (window.index >= history_start_day) & (window.index < history_end_day)
            ]
            future_series = window[
                (window.index >= history_end_day)
                & (window.index < history_end_day + pd.DateOffset(days=7))
            ]
        elif "LaneClosureDuring" in self.__class__.__name__:
            # get midpoint between start and end of lane closure
            midpoint_duration = (self.lane_closure_end - self.lane_closure_start) / 2
            lane_closure_midpoint = self.lane_closure_start + midpoint_duration
            lane_closure_midpoint = lane_closure_midpoint.round("H")
            history_start_day = (
                lane_closure_midpoint - pd.DateOffset(days=self.get_history_factor())
            ).normalize()

            history_end_day = lane_closure_midpoint
            history_series = window[
                (window.index >= history_start_day) & (window.index < history_end_day)
            ]
            future_series = window[
                (window.index >= history_end_day)
                & (window.index < (history_end_day + pd.DateOffset(days=7)).normalize())
            ]
        elif "LaneClosureAfter" in self.__class__.__name__:
            history_end_day = lane_closure_start_day - pd.DateOffset(days=1)
            history_start_day = history_end_day - pd.DateOffset(
                days=self.get_history_factor()
            )
            history_series = window[
                (window.index >= history_start_day) & (window.index < history_end_day)
            ]
            future_series = window[
                (window.index >= history_end_day)
                & (window.index < history_end_day + pd.DateOffset(days=7))
            ]

        else:
            raise ValueError("Invalid task class name")

        return history_series, future_series

    # def get_context(self, sensor_data, lane_closure):
    #     """
    #     Get the context of the task.
    #     """
    #     background = self.get_background(sensor_data)
    #     scenario = self.get_scenario(lane_closure)
    #     return f"{background} {scenario}"

    def get_background(self, sensor_data):
        freeway_dir = sensor_data["Fwy"].iloc[0]
        district = sensor_data["District"].iloc[0]
        county = sensor_data["County"].iloc[0]
        abs_pm = sensor_data["Abs PM"].iloc[0]

        if "ShortBackground" in self.__class__.__name__:
            return short_background.format(freeway_dir=freeway_dir)
        elif "MediumBackground" in self.__class__.__name__:
            return medium_background.format(
                freeway_dir=freeway_dir, district=district, county=county
            )
        elif "LongBackground" in self.__class__.__name__:
            return long_background.format(
                freeway_dir=freeway_dir, district=district, county=county, abs_pm=abs_pm
            )

    def get_scenario(self, lane_closure):
        """
        Get the scenario of the task.
        """
        expected_start_date = pd.to_datetime(lane_closure["Start Date"])
        expected_end_date = pd.to_datetime(lane_closure["End Date"])
        expected_duration = str(lane_closure["Planned Duration"]) + " minutes"

        closure_lanes = lane_closure["Closure Lanes"]
        total_lanes = lane_closure["Total Lanes"]

        return (
            f"Lane closure from {expected_start_date} to {expected_end_date} with a planned duration of {expected_duration}."
            f"The number of lanes closed is {closure_lanes} out of {total_lanes} total lanes."
        )

    def get_prediction_length(self, window):
        # Calculate the number of hours between the end of the window and the start of the lane closure day
        lane_closure_end = self.lane_closure_end
        end_of_closure_day = pd.to_datetime(lane_closure_end).normalize()

        end_of_closure_day_extended = pd.to_datetime(
            end_of_closure_day
        ).normalize() + pd.DateOffset(days=2)

        # The prediction length is the number of hours from the start of the closure day to the end of the window
        prediction_length = (
            window.index[-1] - end_of_closure_day_extended
        ).total_seconds() / 3600

        return int(prediction_length)

    @property
    def seasonal_period(self) -> int:
        """
        This returns the period which should be used by statistical models for this task.
        If negative, this means that the data either has no period, or the history is shorter than the period.
        """
        return 24


class LaneClosureBeforeShortBackgroundTask(AbstractLaneClosureTask):
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
        super().__init__(seed=seed, fixed_config=fixed_config, target=target)


class LaneClosureBeforeMediumBackgroundTask(AbstractLaneClosureTask):
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
        super().__init__(seed=seed, fixed_config=fixed_config, target=target)


class LaneClosureBeforeLongBackgroundTask(AbstractLaneClosureTask):
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
        super().__init__(seed=seed, fixed_config=fixed_config, target=target)


class LaneClosureDuringShortBackgroundTask(AbstractLaneClosureTask):
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
        super().__init__(seed=seed, fixed_config=fixed_config, target=target)


class LaneClosureDuringMediumBackgroundTask(AbstractLaneClosureTask):
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
        super().__init__(seed=seed, fixed_config=fixed_config, target=target)


class LaneClosureDuringLongBackgroundTask(AbstractLaneClosureTask):
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
        super().__init__(seed=seed, fixed_config=fixed_config, target=target)


class LaneClosureAfterShortBackgroundTask(AbstractLaneClosureTask):
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
        super().__init__(seed=seed, fixed_config=fixed_config, target=target)


class LaneClosureAfterMediumBackgroundTask(AbstractLaneClosureTask):
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
        super().__init__(seed=seed, fixed_config=fixed_config, target=target)


class LaneClosureAfterLongBackgroundTask(AbstractLaneClosureTask):
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
        super().__init__(seed=seed, fixed_config=fixed_config, target=target)


class LaneClosureBeforeShortHistoryShortBackgroundTask(AbstractLaneClosureTask):
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
        super().__init__(seed=seed, fixed_config=fixed_config, target=target)


class LaneClosureBeforeShortHistoryMediumBackgroundTask(AbstractLaneClosureTask):
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
        super().__init__(seed=seed, fixed_config=fixed_config, target=target)


class LaneClosureBeforeShortHistoryLongBackgroundTask(AbstractLaneClosureTask):
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
        super().__init__(seed=seed, fixed_config=fixed_config, target=target)


class LaneClosureDuringShortHistoryShortBackgroundTask(AbstractLaneClosureTask):
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
        super().__init__(seed=seed, fixed_config=fixed_config, target=target)


class LaneClosureDuringShortHistoryMediumBackgroundTask(AbstractLaneClosureTask):
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
        super().__init__(seed=seed, fixed_config=fixed_config, target=target)


class LaneClosureDuringShortHistoryLongBackgroundTask(AbstractLaneClosureTask):
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
        super().__init__(seed=seed, fixed_config=fixed_config, target=target)


class LaneClosureAfterShortHistoryShortBackgroundTask(AbstractLaneClosureTask):
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
        super().__init__(seed=seed, fixed_config=fixed_config, target=target)


class LaneClosureAfterShortHistoryMediumBackgroundTask(AbstractLaneClosureTask):
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
        super().__init__(seed=seed, fixed_config=fixed_config, target=target)


class LaneClosureAfterShortHistoryLongBackgroundTask(AbstractLaneClosureTask):
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
        super().__init__(seed=seed, fixed_config=fixed_config, target=target)


__TASKS__ = [
    LaneClosureBeforeShortBackgroundTask,
    LaneClosureBeforeMediumBackgroundTask,
    LaneClosureBeforeLongBackgroundTask,
    LaneClosureDuringShortBackgroundTask,
    LaneClosureDuringMediumBackgroundTask,
    LaneClosureDuringLongBackgroundTask,
    LaneClosureAfterShortBackgroundTask,
    LaneClosureAfterMediumBackgroundTask,
    LaneClosureAfterLongBackgroundTask,
    LaneClosureBeforeShortHistoryShortBackgroundTask,
    LaneClosureBeforeShortHistoryMediumBackgroundTask,
    LaneClosureBeforeShortHistoryLongBackgroundTask,
    LaneClosureDuringShortHistoryShortBackgroundTask,
    LaneClosureDuringShortHistoryMediumBackgroundTask,
    LaneClosureDuringShortHistoryLongBackgroundTask,
    LaneClosureAfterShortHistoryShortBackgroundTask,
    LaneClosureAfterShortHistoryMediumBackgroundTask,
    LaneClosureAfterShortHistoryLongBackgroundTask,
]
