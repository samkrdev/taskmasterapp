# Directory: assignment/interfaces.py
"""
Interfaces for task assignment models.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from models import Task, Employee


class AssignmentModel(ABC):
    """Base interface for task assignment models."""

    @abstractmethod
    def assign(
        self, tasks: List[Task], employees: List[Employee]
    ) -> Dict[str, Optional[str]]:
        """
        Assign tasks to employees.

        Args:
            tasks: List of tasks to assign
            employees: List of employees available for assignment

        Returns:
            Dict mapping task IDs to employee IDs (or None if unassigned)
        """
        pass
