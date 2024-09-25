__version__ = "0.0.1"

from .tasks.electricity_tasks import __TASKS__ as ELECTRICITY_TASKS
from .tasks.nn5_tasks import __TASKS__ as NN5_TASKS
from .tasks.pred_change_tasks import __TASKS__ as PRED_CHANGE_TASKS
from .tasks.predictable_constraints_real_data import (
    __TASKS__ as PREDICTABLE_CONSTRAINT_TASKS,
)
from .predictable_grocer_shocks import __TASKS__ as PREDICTABLE_GROCER_SHOCKS_TASKS
from .tasks.predictable_spikes_in_pred import (
    __TASKS__ as PREDICTABLE_SPIKES_IN_PRED_TASKS,
)
from .tasks.predictable_stl_shocks import __TASKS__ as PREDICTABLE_STL_SHOCKS_TASKS
from .tasks.sensor_maintenance import __TASKS__ as SENSOR_MAINTENANCE_TASKS
from .tasks.causal_chambers import __TASKS__ as CAUSAL_CHAMBERS_TASKS
from .tasks.bivariate_categorical_causal import __TASKS__ as CATEGORICAL_CAUSAL_TASKS
from .tasks.solar_tasks import __TASKS__ as SOLAR_TASKS
from .tasks.traffic_tasks import __TASKS__ as TRAFFIC_TASKS
from .tasks.nsrdb_tasks import __TASKS__ as NSRDB_TASKS
from .tasks.montreal_fire import __TASKS__ as MONTREAL_FIRE_TASKS
from .tasks.fred_county_tasks import __TASKS__ as FRED_COUNTY_TASKS
from .tasks.pems_tasks import __TASKS__ as PEMS_TASKS

# All tasks that are officially included in the benchmark
ALL_TASKS = (
    ELECTRICITY_TASKS
    + NN5_TASKS
    + PRED_CHANGE_TASKS
    + PREDICTABLE_CONSTRAINT_TASKS
    + PREDICTABLE_GROCER_SHOCKS_TASKS
    # + PREDICTABLE_SPIKES_IN_PRED_TASKS
    # + PREDICTABLE_STL_SHOCKS_TASKS
    + SENSOR_MAINTENANCE_TASKS
    + CAUSAL_CHAMBERS_TASKS
    + CATEGORICAL_CAUSAL_TASKS
    + SOLAR_TASKS
    + TRAFFIC_TASKS
    + NSRDB_TASKS
    + MONTREAL_FIRE_TASKS
    + FRED_COUNTY_TASKS
    + PEMS_TASKS
)
