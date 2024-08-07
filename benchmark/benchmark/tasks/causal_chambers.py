import numpy as np
import pandas as pd

from causalchamber.datasets import Dataset
from collections import namedtuple

from .base import UnivariateCRPSTask

Window = namedtuple('Window', ['seed', 'history_start', 'future_start', 'time_end'])

class WindTunnelTask(UnivariateCRPSTask):

    def __init__(self, target_name: str, seed: int = None, fixed_config: dict = None, dataset_name: str = 'wt_changepoints_v1', datadir: str = '/mnt/starcaster/data/causal_chambers/'):
        
        super().__init__(seed=seed, fixed_config=fixed_config)

        # Download the dataset and store it in starcaster disk
        self.dataset = Dataset(dataset_name, root=datadir, download=True)
        self.seed = seed
        self.covariate_name = "load_in"
        self.target_name = target_name 

    def random_instance(self):

        window = np.random.choice(self.possible_windows)
        
        experiment = self.dataset.get_experiment(name=f'load_in_seed_{window.seed}')
        observations = experiment.as_pandas_dataframe()

        selected_variables = observations[[self.covariate_name, self.target_name]].copy()
        selected_variables.index = selected_variables.index.to_timestamp()

        # Instantiate the class variables
        self.past_time = selected_variables[window.history_start:window.future_start]
        self.future_time = selected_variables[window.future_start:window.time_end]
        

class SpeedFromLoad(WindTunnelTask):

    def __init__(self, target_name: str = "rpm_in", seed: int = None, fixed_config: dict = None, dataset_name: str = 'wt_changepoints_v1', datadir: str = '/mnt/starcaster/data/causal_chambers/'):
        
        self.possible_windows = [
            Window(4, 0, 952, 1180),
            Window(3, 300, 807, 1420),
        ]
         
        self.background = "The wind tunnel is a chamber with one controllable fan that pushes air through it. We can control the load of the fan (corresponding to the duty cycle of the pulse-width-modulation signal) and measure its speed (in revolutions per minute). The fan is designed so its steady-state speed scales broadly linearly with the load. Unless completely powered off, the fan never operates below a certain speed, corresponding to a minimum effective load, which is between 0.1 and 0.2."
        self.constraints = "The load is between 0 and 1. At full load (=1), the fan turns at a maximum speed of 3000 rpm."