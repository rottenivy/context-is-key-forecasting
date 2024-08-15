"""
Unit tests that check if tasks are consistent with our standards

"""

import pandas as pd
import pytest

from benchmark import ALL_TASKS
from benchmark.base import BaseTask, ALLOWED_CONTEXT_SOURCES, ALLOWED_SKILLS


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
def test_context_sources(task):
    """
    Test the task marks at least one kind of context as being used and that all listed context sources are allowed.

    """
    assert len(task._context_sources) > 0
    assert all(ctx in ALLOWED_CONTEXT_SOURCES for ctx in task._context_sources)


@pytest.mark.parametrize("task", ALL_TASKS)
def test_skills(task):
    """
    Test the task marks at least three kinds of skill as being used and that all listed skills are allowed.

    """
    assert len(task._skills) > 2
    assert all(skill in ALLOWED_SKILLS for skill in task._skills)
