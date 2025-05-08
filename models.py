# Directory: models.py
"""
Core data models for the task scheduling system.
"""
from dataclasses import dataclass, field
from typing import Set, List, Optional
from datetime import datetime


@dataclass
class Task:
    """Task model representing a work item to be scheduled and assigned."""

    id: str
    name: str
    required_skills: Set[str]
    estimated_hours: int
    due_date: datetime
    priority: int
    dependencies: Set[str] = field(default_factory=set)

    def __repr__(self) -> str:
        """Improved string representation for debugging."""
        return (
            f"Task({self.id}, name={self.name}, skills={self.required_skills}, "
            f"hours={self.estimated_hours}, priority={self.priority}, "
            f"deps={self.dependencies})"
        )


@dataclass
class Employee:
    """Employee model representing a resource that can be assigned to tasks."""

    id: str
    name: str
    skills: Set[str]
    weekly_available_hours: int
    efficiency_rating: float
    assigned_tasks: List[str] = field(default_factory=list)
    assigned_hours: int = 0

    def __post_init__(self):
        """Initialize default values for collections if None."""
        if self.assigned_tasks is None:
            self.assigned_tasks = []
        if self.assigned_hours is None:
            self.assigned_hours = 0

    def __repr__(self) -> str:
        """Improved string representation for debugging."""
        return (
            f"Employee({self.id}, name={self.name}, skills={self.skills}, "
            f"hours={self.weekly_available_hours}, efficiency={self.efficiency_rating})"
        )

    def can_handle_task(self, task: Task) -> bool:
        """Check if employee has the required skills for a task."""
        return bool(self.skills.intersection(task.required_skills))

    def compute_task_duration(self, task: Task) -> int:
        """Compute how long it would take this employee to complete a task."""
        return int(task.estimated_hours / self.efficiency_rating)

    def has_availability_for(self, task: Task) -> bool:
        """Check if employee has enough available hours for a task."""
        needed_hours = self.compute_task_duration(task)
        return (self.weekly_available_hours - self.assigned_hours) >= needed_hours
