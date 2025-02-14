__version__ = "0.0.1"

from fractions import Fraction

from .base import BaseTask


from .tasks.general_traffic_tasks import (
        __TASKS__ as TRAFFIC_TASKS,
        __CLUSTERS__ as TRAFFIC_CLUSTERS
)
from .tasks.general_solar_tasks import (
        __TASKS__ as SOLAR_TASKS,
        __CLUSTERS__ as SOLAR_CLUSTERS
)
from .tasks.general_electricity_tasks import (
        __TASKS__ as ELECTRICITY_TASKS,
        __CLUSTERS__ as ELECTRICITY_CLUSTERS
)
from .tasks.general_fred_tasks import (
        __TASKS__ as FRED_TASKS,
        __CLUSTERS__ as FRED_CLUSTERS
)
from .tasks.general_retail_tasks import (
        __TASKS__ as RETAIL_TASKS,
        __CLUSTERS__ as RETAIL_CLUSTERS
)

# All tasks that are officially included in the benchmark
ALL_TASKS = (
    # TRAFFIC_TASKS,
    # SOLAR_TASKS,
    # ELECTRICITY_TASKS,
    # FRED_TASKS,
    RETAIL_TASKS
)

WEIGHT_CLUSTERS = (
    # TRAFFIC_CLUSTERS,
    # SOLAR_CLUSTERS,
    # ELECTRICITY_CLUSTERS,
    # FRED_CLUSTERS,
    RETAIL_CLUSTERS
)


def get_task_weight(task: BaseTask) -> Fraction:
    for cluster in WEIGHT_CLUSTERS:
        if task in cluster.tasks:
            return Fraction(cluster.weight) / len(cluster.tasks)


TASK_NAME_TO_WEIGHT = {task.__name__: get_task_weight(task) for task in ALL_TASKS}
