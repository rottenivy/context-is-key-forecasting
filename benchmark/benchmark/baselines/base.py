"""
Base classes for baselines

"""

import numpy as np

from abc import ABC, abstractmethod


from ..base import BaseTask


class Baseline(ABC):
    """
    Base class for baselines

    """

    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, task_instance: BaseTask, n_samples: int) -> np.ndarray:
        """
        Infer forecast for a task instance using the baseline

        """
        pass

    @property
    def cache_name(self) -> str:
        return self.__class__.__name__
