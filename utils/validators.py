# Directory: utils/validators.py
"""
Validation utilities for tasks and employees.
"""
import networkx as nx
import matplotlib.pyplot as plt
import math
from typing import List, Dict, Set, Optional
from models import Task, Employee
from utils.logger import logger
import numpy as np


def validate_tasks(tasks: List[Task]) -> bool:
    """Validate tasks for circular dependencies."""
    graph = nx.DiGraph()

    # Add all tasks and dependencies to the graph
    for task in tasks:
        graph.add_node(task.id)
        for dep in task.dependencies:
            graph.add_edge(dep, task.id)

    # Check if the graph is a DAG (Directed Acyclic Graph)
    if not nx.is_directed_acyclic_graph(graph):
        logger.error("Tasks have circular dependencies.")
        return False

    return True


def create_comparison_dashboard(
    methods_data: Dict[str, Dict[str, float]],
    filename: Optional[str] = None,
) -> None:
    """
    Create a comprehensive dashboard comparing different assignment methods.

    Args:
        methods_data: Dictionary mapping method names to metric dictionaries
        filename: File to save the dashboard
    """
    if not methods_data:
        logger.warning("No data provided for dashboard.")
        return

    # Set a unified style for the plots
    plt.style.use("seaborn-v0_8-whitegrid")

    # Create a figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Assignment Methods Comparison Dashboard", fontsize=16)

    # Flatten axes for easier indexing
    axs = axs.flatten()

    # Extract methods and metrics
    methods = list(methods_data.keys())
    metrics = set()
    for m_data in methods_data.values():
        metrics.update(m_data.keys())

    metrics = sorted(list(metrics))

    # 1. Bar chart for workload balance (lower is better)
    if "workload_balance_ratio" in metrics:
        ax = axs[0]
        balance_data = {
            m: methods_data[m].get("workload_balance_ratio", 0) for m in methods
        }
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(methods)))
        bars = ax.bar(balance_data.keys(), balance_data.values(), color=colors)

        # Add data labels
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        ax.set_title("Workload Balance Ratio (Lower is Better)")
        ax.set_xlabel("Assignment Method")
        ax.set_ylabel("Std Dev / Mean Workload")
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    # 2. Bar chart for resource utilization (higher is better)
    if "resource_utilization" in metrics:
        ax = axs[1]
        util_data = {m: methods_data[m].get("resource_utilization", 0) for m in methods}
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(methods)))
        bars = ax.bar(util_data.keys(), util_data.values(), color=colors)

        # Add data labels
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        ax.set_title("Resource Utilization (Higher is Better)")
        ax.set_xlabel("Assignment Method")
        ax.set_ylabel("Utilization Percentage")
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    # 3. Radar chart for multiple metrics (higher is better)
    ax = axs[2]
    relevant_metrics = [m for m in metrics if m != "workload_balance_ratio"]

    if relevant_metrics:
        # Number of metrics (variables)
        num_metrics = len(relevant_metrics)

        # Set angles for radar chart
        angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Close the polygon

        # Plot for each method
        for i, method in enumerate(methods):
            values = []
            for metric in relevant_metrics:
                # Normalize to 0-1 range for radar chart
                if metric in methods_data[method]:
                    values.append(methods_data[method][metric] / 100)
                else:
                    values.append(0)

            # Close the polygon
            values += values[:1]

            # Plot the polygon
            ax.plot(angles, values, linewidth=2, label=method)
            ax.fill(angles, values, alpha=0.1)

        # Set labels for radar chart
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace("_", " ").title() for m in relevant_metrics])

        # Add legend and title
        ax.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))
        ax.set_title("Normalized Performance Metrics (Higher is Better)")

    # 4. Summary table
    ax = axs[3]
    ax.axis("tight")
    ax.axis("off")

    # Prepare table data
    table_data = []
    for metric in metrics:
        row = [metric.replace("_", " ").title()]

        for method in methods:
            if metric in methods_data[method]:
                # Format based on metric type
                if metric == "workload_balance_ratio":
                    row.append(f"{methods_data[method][metric]:.3f}")
                elif (
                    "ratio" in metric
                    or "rate" in metric
                    or "utilization" in metric
                    or "coverage" in metric
                ):
                    row.append(f"{methods_data[method][metric]:.1f}%")
                else:
                    row.append(f"{methods_data[method][metric]:.2f}")
            else:
                row.append("N/A")

        table_data.append(row)

    # Create the table
    table = ax.table(
        cellText=table_data,
        colLabels=["Metric"] + methods,
        loc="center",
        cellLoc="center",
    )

    # Configure table appearance
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save the dashboard
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        logger.info(f"Comparison dashboard saved as {filename}")

    plt.show()


def validate_assignments(
    tasks: List[Task], employees: List[Employee], assignments: Dict[str, Optional[str]]
) -> bool:
    """
    Validate task assignments for feasibility.

    Checks:
    1. Every assigned task has an employee with required skills
    2. No employee is overloaded
    3. All task dependencies are honored

    Args:
        tasks: List of tasks
        employees: List of employees
        assignments: Dictionary mapping task IDs to employee IDs

    Returns:
        bool: True if assignments are valid, False otherwise
    """
    emp_map = {emp.id: emp for emp in employees}
    task_map = {task.id: task for task in tasks}

    # Track employee workloads
    workloads = {emp.id: 0 for emp in employees}

    # Track assigned tasks for dependency validation
    assigned_tasks = set()

    # Validate each assignment
    for task_id, emp_id in assignments.items():
        # Skip unassigned tasks
        if emp_id is None:
            continue

        # Get task and employee objects
        task = task_map.get(task_id)
        emp = emp_map.get(emp_id)

        # Check if task exists
        if task is None:
            logger.error(f"Task {task_id} not found in tasks list.")
            return False

        # Check if employee exists
        if emp is None:
            logger.error(f"Employee {emp_id} not found in employees list.")
            return False

        # Check skill match
        if not emp.skills.intersection(task.required_skills):
            logger.error(
                f"Employee {emp_id} lacks required skills for task {task_id}. "
                f"Employee skills: {emp.skills}, Task requires: {task.required_skills}"
            )
            return False

        # Check workload capacity
        needed_hours = math.ceil(task.estimated_hours / emp.efficiency_rating)
        workloads[emp_id] += needed_hours

        if workloads[emp_id] > emp.weekly_available_hours:
            logger.error(
                f"Employee {emp_id} is overloaded. Assigned: {workloads[emp_id]} hours, "
                f"Available: {emp.weekly_available_hours} hours."
            )
            return False

        # Mark task as assigned
        assigned_tasks.add(task_id)

    # Check dependencies
    for task_id, emp_id in assignments.items():
        # Skip unassigned tasks
        if emp_id is None:
            continue

        task = task_map.get(task_id)

        # Check if all dependencies are assigned
        for dep_id in task.dependencies:
            if dep_id not in assigned_tasks:
                logger.error(
                    f"Task {task_id} depends on task {dep_id} which is not assigned."
                )
                return False

    return True


def test_task_validation():
    """Test the task validation function."""
    from datetime import datetime

    # Create a set of tasks with circular dependencies
    cyc_tasks = [
        Task(
            id="T1",
            name="A",
            required_skills={"Drilling"},
            estimated_hours=5,
            due_date=datetime(2023, 10, 2),
            priority=1,
            dependencies={"T2"},
        ),
        Task(
            id="T2",
            name="B",
            required_skills={"Milling"},
            estimated_hours=8,
            due_date=datetime(2023, 10, 3),
            priority=3,
            dependencies={"T3"},
        ),
        Task(
            id="T3",
            name="C",
            required_skills={"Lathe"},
            estimated_hours=6,
            due_date=datetime(2023, 10, 4),
            priority=2,
            dependencies={"T1"},
        ),
    ]

    assert not validate_tasks(cyc_tasks), "Cycle not detected!"

    # Create a set of tasks with linear dependencies
    linear_tasks = [
        Task(
            id="T1",
            name="A",
            required_skills={"Drilling"},
            estimated_hours=5,
            due_date=datetime(2023, 10, 2),
            priority=1,
            dependencies=set(),
        ),
        Task(
            id="T2",
            name="B",
            required_skills={"Milling"},
            estimated_hours=8,
            due_date=datetime(2023, 10, 3),
            priority=3,
            dependencies={"T1"},
        ),
        Task(
            id="T3",
            name="C",
            required_skills={"Lathe"},
            estimated_hours=6,
            due_date=datetime(2023, 10, 4),
            priority=2,
            dependencies={"T2"},
        ),
    ]

    assert validate_tasks(linear_tasks), "Valid chain marked invalid!"

    # Test assignment validation
    employees = [
        Employee(
            id="E1",
            name="Employee 1",
            skills={"Drilling"},
            weekly_available_hours=20,
            efficiency_rating=1.0,
        ),
        Employee(
            id="E2",
            name="Employee 2",
            skills={"Milling", "Lathe"},
            weekly_available_hours=30,
            efficiency_rating=0.8,
        ),
    ]

    # Valid assignments
    valid_assignments = {
        "T1": "E1",  # Drilling task to E1 (has Drilling skill)
        "T2": "E2",  # Milling task to E2 (has Milling skill)
        "T3": "E2",  # Lathe task to E2 (has Lathe skill)
    }

    assert validate_assignments(
        linear_tasks, employees, valid_assignments
    ), "Valid assignments marked invalid!"

    # Invalid assignments (wrong skills)
    invalid_skills = {
        "T1": "E2",  # E2 doesn't have Drilling skill
        "T2": "E2",
        "T3": "E1",  # E1 doesn't have Lathe skill
    }

    assert not validate_assignments(
        linear_tasks, employees, invalid_skills
    ), "Invalid skills not detected!"

    # Invalid assignments (overloaded)
    overloaded_tasks = [
        Task(
            id="T4",
            name="Big Task",
            required_skills={"Drilling"},
            estimated_hours=30,  # Exceeds E1's capacity
            due_date=datetime(2023, 10, 5),
            priority=1,
            dependencies=set(),
        )
    ]

    overloaded_assignments = {"T4": "E1"}

    assert not validate_assignments(
        overloaded_tasks, employees, overloaded_assignments
    ), "Overloaded employee not detected!"

    logger.info("Task and assignment validation tests passed.")
