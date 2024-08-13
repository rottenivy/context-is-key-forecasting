"""
Unit tests that check if tasks are consistent with our standards

"""

import pandas as pd
import pytest

from benchmark import ALL_TASKS
from benchmark.base import BaseTask


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


@pytest.mark.parametrize("task", ALL_TASKS)
def test_at_least_one_context_source_active(task):
    """
    Test the task marks at least one kind of context as being used

    """
    assert len(task._context_sources) > 0
