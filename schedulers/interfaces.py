# Directory: schedulers/interfaces.py
"""
Interfaces for schedulers.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from models import Task


class Scheduler(ABC):
    """Base scheduler interface."""

    @abstractmethod
    def schedule(self) -> List[Task]:
        """
        Schedule tasks in an optimal order.

        Returns:
            List[Task]: Ordered list of tasks
        """
        pass
