import numpy as np
import pandas as pd

from causalchamber.datasets import Dataset
from collections import namedtuple

from ..base import UnivariateCRPSTask

Window = namedtuple("Window", ["seed", "history_start", "future_start", "time_end"])


class WindTunnelTask(UnivariateCRPSTask):

    def __init__(
        self,
        target_name: str,
        seed: int = None,
        fixed_config: dict = None,
        dataset_name: str = "wt_changepoints_v1",
        datadir: str = "/mnt/starcaster/data/causal_chambers/",
    ):

        self.dataset = Dataset(dataset_name, root=datadir, download=True)
        self.seed = seed
        self.covariate_name = "load_in"
        self.target_name = target_name

        super().__init__(seed=seed, fixed_config=fixed_config)
    
    def _get_number_instances(self):
        """ Returns number of different instances/windows this task comprises"""
        return len(self.possible_windows)

    def _get_instance_by_idx(self, idx: int, downsample : str = None):
        """ Returns instance corresponding to specified index, downsampled if required.

        Args:
            idx (int): Instance index.
            downsample (str, optional): Downsampling rule (see https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.resample.html). Defaults to None.
        """

        window = self.possible_windows[idx]

        experiment = self.dataset.get_experiment(name=f"load_in_seed_{window.seed}")
        observations = experiment.as_pandas_dataframe()

        selected_variables = observations[
            [self.covariate_name, self.target_name]
        ].copy()
        selected_variables.index = pd.to_datetime(observations.timestamp, unit="s")

        # select past and future
        past_time = selected_variables[window.history_start : window.future_start]
        future_time = selected_variables[window.future_start : window.time_end]

        # get verbalized covariates
        text_covariates = self.verbalize_covariate(
            observations.iloc[window.history_start : window.time_end]
        )

        if downsample is not None:
            # downsample numerical variates, not averaging to avoid introducing new values
            past_time = past_time.resample(downsample).min()
            future_time = future_time.resample(downsample).min()
        
        return window, past_time, future_time, text_covariates


    def random_instance(self, downsample: str = "1s"):
        """ Sets random downsampled instance/window as task instance.

        Args:
            downsample (str, optional): Downsampling rule (see https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.resample.html). Defaults to "1s".
        """

        window_idx = self.random.choice(self._get_number_instances())
        self.window, self.past_time, self.future_time, self.covariates = self._get_instance_by_idx(window_idx, downsample)
        

    def verbalize_covariate(self, observations: pd.DataFrame, round_freq: str = 's'):

        covariate = observations[self.covariate_name]
        # intervention column = 0 when load is constant and = 1 when it changes wrt previous timestep
        change_points = list(observations[observations.intervention == 1].index)

        # timestep 0 is always tagged as change point, but we don't need it
        # timestep 0 is not present when window starts at a later timestep
        if 0 in change_points:
            change_points.remove(0)

        # get datetimes from unix times, drop date as it is constant
        timestamps = pd.to_datetime(observations.timestamp, unit="s").dt.round(freq=round_freq).dt.time

        ans = f"The load is set to: {covariate.iloc[0]:.1f}"
        for c in change_points:
            ans += f" until {timestamps[c]}, {covariate[c]:.1f} from {timestamps[c]}"

        ans += f" until {timestamps.iloc[-1]}."

        return ans


class SpeedFromLoad(WindTunnelTask):

    def __init__(
        self,
        seed: int = None,
        fixed_config: dict = None,
        datadir: str = "/mnt/starcaster/data/causal_chambers/",
    ):

        self.possible_windows = [
            Window(4, 0, 952, 1100),
            Window(3, 300, 807, 1420),
            Window(7, 0, 613, 1000),
            Window(4, 0, 1886, 2000),
            Window(5, 0, 502, 600),
            Window(6, 0, 686, 880),
            Window(2, 0, 440, 700),
            Window(0, 0, 1159, 1300),
            Window(1, 0, 779, 900),
            Window(1, 0, 779, 1400)
        ]

        self.background = "The wind tunnel is a chamber with one controllable fan that pushes air through it. We can control the load of the fan (corresponding to the duty cycle of the pulse-width-modulation signal) and measure its speed (in revolutions per minute). The fan is designed so its steady-state speed scales broadly linearly with the load. Unless completely powered off, the fan never operates below a certain speed, corresponding to a minimum effective load between 0.1 and 0.2."
        self.constraints = "The load is between 0 and 1. At full load (=1), the fan turns at a maximum speed of 3000 rpm."

        super().__init__("rpm_in", seed, fixed_config, "wt_changepoints_v1", datadir)
