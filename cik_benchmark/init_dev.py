__version__ = "0.0.1"

from fractions import Fraction

from .base import BaseTask


from.tasks.general_traffic_tasks import (
        __TASKS__ as TRAFFIC_TASKS,
        __CLUSTERS__ as TRAFFIC_CLUSTERS
)

# All tasks that are officially included in the benchmark
ALL_TASKS = (
    TRAFFIC_TASKS
)

WEIGHT_CLUSTERS = (
    TRAFFIC_CLUSTERS

)


def get_task_weight(task: BaseTask) -> Fraction:
    for cluster in WEIGHT_CLUSTERS:
        if task in cluster.tasks:
            return Fraction(cluster.weight) / len(cluster.tasks)


TASK_NAME_TO_WEIGHT = {task.__name__: get_task_weight(task) for task in ALL_TASKS}
