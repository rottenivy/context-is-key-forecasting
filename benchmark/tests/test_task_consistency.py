import pytest

from benchmark import ALL_TASKS
from benchmark.base import BaseTask


@pytest.mark.parametrize("task", ALL_TASKS)
def test_inherits_base(task):
    """
    Test that each task inherits the base class

    """
    assert issubclass(task, BaseTask)
