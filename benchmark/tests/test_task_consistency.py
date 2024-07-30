"""
Unit tests that check if tasks are consistent with our standards

"""

import pandas as pd
import pytest

from benchmark import ALL_TASKS
from benchmark.base import BaseTask

# TODO: Remove this when dominick data loader is merged
from benchmark.predictable_grocer_shocks import __TASKS__ as DOMINICK_TASKS

ALL_TASKS = list(set(ALL_TASKS) - set(DOMINICK_TASKS))


@pytest.mark.parametrize("task", ALL_TASKS)
def test_inherits_base(task):
    """
    Test that each task inherits the base class

    """
    assert issubclass(task, BaseTask)


@pytest.mark.parametrize("task", ALL_TASKS)
def test_time_data_is_dataframe(task):
    """
    Test that the temporal data is given as a Pandas dataframe

    """
    task_instance = task()
    assert isinstance(task_instance.past_time, pd.DataFrame)
    assert isinstance(task_instance.future_time, pd.DataFrame)


@pytest.mark.parametrize("task", ALL_TASKS)
def test_some_context_exists(task):
    """
    Test that at least part of the context is non-empty

    """
    task_instance = task()
    assert (
        task_instance.background or task_instance.constraints or task_instance.scenario
    )
