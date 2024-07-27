import pytest

from benchmark import ALL_TASKS


@pytest.mark.parametrize("task", ALL_TASKS)
def test_instance_randomness(task):
    """
    Test that each task can produce random instances

    """
    instance_1 = task(seed=1)
    instance_2 = task(seed=2)

    # Check that the data in the past_time and future_time are different
    same_data = instance_1.past_time.to_csv(index=False) == instance_2.past_time.to_csv(
        index=False
    )

    # Check if the background, constraints, and scenario are the same
    same_background = instance_1.background == instance_2.background
    same_constraints = instance_1.constraints == instance_2.constraints
    same_scenario = instance_1.scenario == instance_2.scenario

    assert not (same_data and same_background and same_constraints and same_scenario)
