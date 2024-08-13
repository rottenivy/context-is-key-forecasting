__version__ = "0.0.1"

from .tasks.electricity_tasks import __TASKS__ as ELECTRICITY_TASKS
from .tasks.nn5_tasks import __TASKS__ as NN5_TASKS
from .pred_change_tasks import __TASKS__ as PRED_CHANGE_TASKS
from .predictable_constraints_real_data import __TASKS__ as PREDICTABLE_CONSTRAINT_TASKS
from .predictable_grocer_shocks import __TASKS__ as PREDICTABLE_GROCER_SHOCKS_TASKS
from .predictable_spikes_in_pred import __TASKS__ as PREDICTABLE_SPIKES_IN_PRED_TASKS
from .predictable_stl_shocks import __TASKS__ as PREDICTABLE_STL_SHOCKS_TASKS
from .sensor_maintenance import __TASKS__ as SENSOR_MAINTENANCE_TASKS
from .tasks.causal_chambers import __TASKS__ as CAUSAL_CHAMBERS_TASKS
from .tasks.short_history import __TASKS__ as SHORT_HISTORY_TASKS

# All tasks that are officially included in the benchmark
ALL_TASKS = (
    ELECTRICITY_TASKS
    + NN5_TASKS
    + PRED_CHANGE_TASKS
    + PREDICTABLE_CONSTRAINT_TASKS
    + PREDICTABLE_GROCER_SHOCKS_TASKS
    + PREDICTABLE_SPIKES_IN_PRED_TASKS
    + PREDICTABLE_STL_SHOCKS_TASKS
    + SENSOR_MAINTENANCE_TASKS
    + SHORT_HISTORY_TASKS
    + PREDICTABLE_STL_SHOCKS_TASKS
    + CAUSAL_CHAMBERS_TASKS
)
