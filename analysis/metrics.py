# Directory: analysis/metrics.py
"""
Metrics calculation and analysis for task assignments.
"""
import numpy as np
from typing import List, Dict, Optional, Set
import math
from datetime import datetime
from models import Task, Employee
from utils.logger import logger


def compute_detailed_metrics(
    tasks: List[Task],
    assigned_dict: Dict[str, Optional[str]],
    employees: List[Employee],
) -> Dict[str, float]:
    """
    Compute detailed performance metrics for an assignment solution.

    Args:
        tasks: List of tasks
        assigned_dict: Dictionary mapping task IDs to employee IDs
        employees: List of employees

    Returns:
        Dict of metric names to metric values
    """
    emp_map = {emp.id: emp for emp in employees}

    # 1. Workload Balance Ratio (Lower is better)
    # Calculate as standard deviation of workload / mean workload
    workload_by_emp = {emp.id: 0 for emp in employees}

    for task in tasks:
        emp_id = assigned_dict.get(task.id)
        if emp_id and emp_id in emp_map:
            emp = emp_map[emp_id]
            needed_hours = math.ceil(task.estimated_hours / emp.efficiency_rating)
            workload_by_emp[emp_id] += needed_hours

    workloads = list(workload_by_emp.values())
    mean_workload = np.mean(workloads) if workloads else 0
    std_workload = np.std(workloads) if workloads else 0
    workload_balance_ratio = std_workload / mean_workload if mean_workload != 0 else 0

    # 2. Assigned Priority Ratio (Higher is better)
    # Calculate as (priority sum of assigned tasks) / (total priority)
    total_priority = sum(task.priority for task in tasks)
    assigned_priority_sum = sum(
        task.priority for task in tasks if assigned_dict.get(task.id) is not None
    )
    assigned_priority_ratio = (
        assigned_priority_sum / total_priority if total_priority != 0 else 0
    )

    # 3. Dependency Completion Rate (Higher is better)
    # Calculate % of assigned tasks that have all dependencies completed first
    task_start_time = {}
    task_finish_time = {}
    emp_available_time = {emp.id: 0 for emp in employees}
    met_dependencies = 0
    assigned_tasks_count = 0

    for task in tasks:
        emp_id = assigned_dict.get(task.id)
        if emp_id is None or emp_id not in emp_map:
            continue

        assigned_tasks_count += 1
        emp = emp_map[emp_id]
        start = emp_available_time[emp_id]
        duration = math.ceil(task.estimated_hours / emp.efficiency_rating)
        finish = start + duration

        task_start_time[task.id] = start
        task_finish_time[task.id] = finish
        emp_available_time[emp_id] = finish

        # Check if all dependencies are completed before this task starts
        deps_met = True
        for dep in task.dependencies:
            if dep not in task_finish_time or task_finish_time[dep] > start:
                deps_met = False
                break

        if deps_met:
            met_dependencies += 1

    dependency_completion_rate = (
        (met_dependencies / assigned_tasks_count * 100)
        if assigned_tasks_count > 0
        else 0
    )

    # 4. Resource Utilization (Higher is better)
    total_available_hours = sum(emp.weekly_available_hours for emp in employees)
    total_assigned_hours = sum(workload_by_emp.values())
    resource_utilization = (
        (total_assigned_hours / total_available_hours * 100)
        if total_available_hours > 0
        else 0
    )

    # 5. Task Coverage (Higher is better)
    task_coverage = (assigned_tasks_count / len(tasks) * 100) if len(tasks) > 0 else 0

    return {
        "workload_balance_ratio": workload_balance_ratio,
        "assigned_priority_ratio": assigned_priority_ratio,
        "dependency_completion_rate": dependency_completion_rate,
        "resource_utilization": resource_utilization,
        "task_coverage": task_coverage,
    }


def compare_assignment_methods(
    task_metrics: Dict[str, Dict[str, float]]
) -> Dict[str, str]:
    """
    Compare assignment methods based on multiple metrics.

    Args:
        task_metrics: Dictionary mapping method names to metric dictionaries

    Returns:
        Dict mapping metric names to best method names
    """
    if not task_metrics:
        return {}

    best_methods = {}
    all_metrics = set()

    # Collect all metrics from all methods
    for method_metrics in task_metrics.values():
        all_metrics.update(method_metrics.keys())

    # For each metric, find the best method
    for metric in all_metrics:
        best_method = None
        best_value = None

        for method, metrics in task_metrics.items():
            if metric not in metrics:
                continue

            value = metrics[metric]

            # For workload_balance_ratio, lower is better
            if metric == "workload_balance_ratio":
                if best_value is None or value < best_value:
                    best_value = value
                    best_method = method
            else:
                # For other metrics, higher is better
                if best_value is None or value > best_value:
                    best_value = value
                    best_method = method

        if best_method:
            best_methods[metric] = best_method

    return best_methods
