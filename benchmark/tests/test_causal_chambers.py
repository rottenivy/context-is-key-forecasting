"""
Unit tests for causal chambers tasks

"""

import pandas as pd

from benchmark.tasks.causal_chambers import WindTunnelTask, SpeedFromLoad


def test_verbalization():

    task = SpeedFromLoad()
    print(task.background)
    task.random_instance()
