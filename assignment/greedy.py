# Directory: assignment/greedy.py
"""
Greedy task assignment implementation.
"""
import math
import random
from typing import List, Dict, Optional, Set
from models import Task, Employee
from assignment.interfaces import AssignmentModel
from utils.logger import logger


class GreedyAssigner(AssignmentModel):
    """
    Greedy task assignment model.

    This model assigns tasks based on a simple greedy heuristic,
    considering task priority and employee workload balance.
    """

    def __init__(self, load_penalty: float = 1.0):
        """
        Initialize the greedy assigner.

        Args:
            load_penalty: Penalty factor for employee workload
        """
        self.load_penalty = load_penalty

    def assign(
        self, tasks: List[Task], employees: List[Employee]
    ) -> Dict[str, Optional[str]]:
        """
        Assign tasks to employees using greedy approach.

        Args:
            tasks: List of tasks to assign
            employees: List of employees to assign to

        Returns:
            Dict mapping task IDs to employee IDs (or None if unassigned)
        """
        # Reset employee assignments
        for emp in employees:
            emp.assigned_tasks = []
            emp.assigned_hours = 0

        # Initialize all tasks as unassigned
        assigned = {t.id: None for t in tasks}
        completed_tasks = set()

        # Sort tasks by priority (highest first)
        sorted_tasks = sorted(tasks, key=lambda t: -t.priority)

        # Process each task in priority order
        for t in sorted_tasks:
            # Skip if already assigned
            if assigned[t.id] is not None:
                continue

            # Skip if dependencies are not met
            if not t.dependencies.issubset(completed_tasks):
                continue

            # Find best employee for this task
            best_employee = None
            best_score = -float("inf")

            for emp in employees:
                # Skip if employee doesn't have required skills
                if not emp.skills.intersection(t.required_skills):
                    continue

                # Calculate hours needed based on efficiency
                needed_hours = math.ceil(t.estimated_hours / emp.efficiency_rating)

                # Skip if employee doesn't have enough available hours
                available = emp.weekly_available_hours - emp.assigned_hours
                if available < needed_hours:
                    continue

                # Calculate assignment score
                # Consider employee efficiency and current workload
                workload_ratio = (
                    emp.assigned_hours / emp.weekly_available_hours
                    if emp.weekly_available_hours > 0
                    else 1.0
                )

                # Higher efficiency, lower workload, and higher task priority are better
                score = (
                    emp.efficiency_rating * t.priority
                    - self.load_penalty * workload_ratio * t.priority
                )

                # Update best employee if better score found
                if score > best_score:
                    best_score = score
                    best_employee = emp

            # Make assignment if suitable employee found
            if best_employee:
                needed_hours = math.ceil(
                    t.estimated_hours / best_employee.efficiency_rating
                )
                assigned[t.id] = best_employee.id
                best_employee.assigned_tasks.append(t.id)
                best_employee.assigned_hours += needed_hours
                completed_tasks.add(t.id)

        return assigned
