"""
Enhanced visualization utilities for task scheduling and assignment.

This module provides improved functions for visualizing task dependencies,
assignment results, and performance metrics with better alignment and
reduced overlapping.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timedelta
from openpyxl import Workbook
from openpyxl.utils import get_column_letter
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
import math

from models import Task, Employee
from utils.logger import logger


def get_employee_color(employee: Employee) -> str:
    """
    Return a color based on the employee's primary skill.

    Args:
        employee: The employee object

    Returns:
        str: Color name for visualization
    """
    # Skill to color mapping
    SKILL_COLOR_MAP = {
        "Drilling": "#3498db",  # Blue
        "Milling": "#2ecc71",  # Green
        "Lathe": "#e67e22",  # Orange
        "CNC": "#9b59b6",  # Purple
        "NDT": "#e74c3c",  # Red
        "Boring": "#8b4513",  # Brown
    }

    if not employee.skills:
        return "gray"

    # Get the first skill in the set (primary skill)
    primary_skill = next(iter(employee.skills))
    return SKILL_COLOR_MAP.get(primary_skill, "gray")


def plot_gantt_chart_for_assignments(
    approach_name: str,
    tasks: List[Task],
    assigned_dict: Dict[str, Optional[str]],
    employees: List[Employee],
    filename: Optional[str] = None,
) -> None:
    """
    Create an improved Gantt chart for task assignments with better alignment of capacity limit bars.

    Args:
        approach_name: Name of the assignment approach
        tasks: List of tasks
        assigned_dict: Dictionary mapping task IDs to employee IDs
        employees: List of employees
        filename: File to save the plot (None to display only)
    """
    # Create mapping for employee indices and employee objects
    employee_index_map = {emp.id: i for i, emp in enumerate(employees)}
    employee_map = {emp.id: emp for emp in employees}

    # Setup the plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Task start and end times per employee
    employee_schedule = {
        emp.id: [] for emp in employees
    }  # List of (task_id, start, end, due_date) per employee
    unassigned_tasks = []

    # Calculate start and end times for each task
    current_time = {emp.id: 0 for emp in employees}  # Current time per employee

    # Compute task start and end times
    for t in tasks:
        assigned_emp_id = assigned_dict.get(t.id)

        if assigned_emp_id is not None and assigned_emp_id in employee_map:
            emp = employee_map[assigned_emp_id]

            # Calculate duration based on employee efficiency
            duration = math.ceil(t.estimated_hours / emp.efficiency_rating)

            # Task starts at the current time for this employee
            start_time = current_time[assigned_emp_id]
            end_time = start_time + duration

            # Add task to employee's schedule
            employee_schedule[assigned_emp_id].append(
                (t.id, start_time, end_time, t.due_date)
            )

            # Update current time for this employee
            current_time[assigned_emp_id] = end_time
        else:
            # Track unassigned tasks
            unassigned_tasks.append(t.id)

    # Define color map for employees based on their skills
    employee_colors = {emp.id: get_employee_color(emp) for emp in employees}

    # Improved bar height for better visibility
    bar_height = 0.6

    # Plot the bars for assigned tasks
    for emp_id, schedule in employee_schedule.items():
        emp_idx = employee_index_map[emp_id]
        emp_color = employee_colors[emp_id]

        for task_id, start, end, due_date in schedule:
            task = next(t for t in tasks if t.id == task_id)

            # Calculate task duration
            duration = end - start

            # Plot the bar
            bar = ax.barh(
                y=emp_idx,
                width=duration,
                left=start,
                height=bar_height,
                align="center",
                color=emp_color,
                edgecolor="black",
                alpha=0.8,
            )

            # Calculate due date relative position (in hours from project start)
            project_start = datetime(2023, 10, 1)
            due_hours = (due_date - project_start).total_seconds() / 3600

            # Determine if task is potentially overdue
            is_overdue = end > due_hours

            # Add task ID and due date within the bar
            ax.text(
                start + duration / 2,
                emp_idx,
                f"{task_id}\nDue: {due_date.strftime('%m-%d')}",
                va="center",
                ha="center",
                color=(
                    "white"
                    if emp_color in ["#3498db", "#9b59b6", "#e74c3c", "#8b4513"]
                    else "black"
                ),
                fontsize=8,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.1", fc="none", ec="none", alpha=0.1),
            )

            # Add an indicator if task might be overdue
            if is_overdue:
                ax.text(end, emp_idx, "⚠️", va="center", ha="left", fontsize=10)

    # Get overall maximum time for x-axis limits
    all_end_times = [
        end for schedule in employee_schedule.values() for _, _, end, _ in schedule
    ]
    max_time = max(
        [emp.weekly_available_hours for emp in employees] + all_end_times + [40]
    )  # Default to 40 if empty

    # Add capacity markers for each employee - IMPROVED ALIGNMENT
    for i, emp in enumerate(employees):
        # Calculate position in axes coordinates
        y_pos = i

        # Draw full-height capacity line using axvspan for better visibility
        ax.axvspan(
            emp.weekly_available_hours - 0.1,  # Slight offset for visibility
            emp.weekly_available_hours + 0.1,  # Slight width for visibility
            ymin=(i - 0.4) / len(employees),  # Align precisely with the row height
            ymax=(i + 0.4) / len(employees),  # Align precisely with the row height
            color="red",
            alpha=0.3,
        )

        # Draw the actual limit line
        ax.axvline(
            x=emp.weekly_available_hours,
            ymin=(i - 0.4) / len(employees),
            ymax=(i + 0.4) / len(employees),
            color="red",
            linestyle="-",
            linewidth=2.0,
        )

        # Add limit label with improved positioning and background
        ax.text(
            emp.weekly_available_hours + 0.5,
            i,
            f"Limit: {emp.weekly_available_hours}h",
            va="center",
            ha="left",
            fontsize=9,
            color="red",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="red", alpha=0.7),
        )

    # Plot unassigned tasks at the bottom with improved visibility
    if unassigned_tasks:
        unassigned_row = len(employees)

        # Add a more prominent separator line
        ax.axhline(
            y=unassigned_row - 0.5,
            color="black",
            linestyle="-",
            linewidth=1.5,
            alpha=0.8,
        )

        # Add unassigned tasks label with better visibility
        ax.text(
            0,
            unassigned_row,
            f"Unassigned Tasks: {', '.join(unassigned_tasks)}",
            va="center",
            ha="left",
            color="black",
            fontsize=10,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="mistyrose", ec="red", alpha=0.8),
        )

    # Configure plot appearance
    ax.set_yticks(range(len(employees)))
    ax.set_yticklabels(
        [f"{emp.id}: {emp.name} ({', '.join(emp.skills)})" for emp in employees]
    )
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Employees (colored by primary skill)")
    ax.set_title(f"Task Assignment Gantt Chart - {approach_name}")

    # Add grid for readability with improved visibility
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    # Add alternating row backgrounds for better readability
    for i in range(len(employees)):
        if i % 2 == 0:  # Every other row
            ax.axhspan(i - 0.5, i + 0.5, color="lightgray", alpha=0.15)

    # Add skill color legend
    skill_patches = []
    for skill, color in {
        "Drilling": "#3498db",
        "Milling": "#2ecc71",
        "Lathe": "#e67e22",
        "CNC": "#9b59b6",
        "NDT": "#e74c3c",
        "Boring": "#8b4513",
    }.items():
        if any(skill in emp.skills for emp in employees):
            skill_patches.append(mpatches.Patch(color=color, label=skill))

    ax.legend(handles=skill_patches, title="Employee Skills", loc="upper right")

    # Set reasonable x-axis limits
    ax.set_xlim(0, max_time * 1.1)  # Add 10% padding

    plt.tight_layout()

    # Save or display
    if filename:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        logger.info(f"Gantt chart saved as {filename}")

    plt.show()


def draw_aco_network(
    task_sequence: List[Task],
    title: str = "ACO Task Sequence",
    filename: Optional[str] = None,
    show_attributes: bool = True,
) -> None:
    """
    Draw the task sequence determined by ACO with improved layout to reduce overlapping.
    Uses a structured layout with hair pin bends to prevent edge overlaps.

    Args:
        task_sequence: Sequence of tasks as determined by ACO
        title: Plot title
        filename: File to save the plot (None to display only)
        show_attributes: Whether to show task attributes
    """
    # Create a directed graph for the sequence
    G = nx.DiGraph()

    # Add tasks to the graph with attributes
    task_attrs = {}
    for i, t in enumerate(task_sequence):
        G.add_node(t.id)
        task_attrs[t.id] = {
            "idx": i,
            "skills": ", ".join(t.required_skills),
            "hours": t.estimated_hours,
            "priority": t.priority,
        }

    # Add sequence edges
    for i in range(len(task_sequence) - 1):
        G.add_edge(task_sequence[i].id, task_sequence[i + 1].id)

    # Create a larger figure for better spacing
    plt.figure(figsize=(16, 10))

    # Create a custom layout to avoid overlapping
    try:
        # Try using graphviz for better layout if available
        try:
            from networkx.drawing.nx_agraph import graphviz_layout

            # Use specific layout parameters to improve separation and routing
            pos = graphviz_layout(
                G,
                prog="dot",
                args="-Grankdir=LR -Goverlap=false -Gsplines=ortho -Gsep=0.5 -Gnodesep=1.0 -Granksep=2.0",
            )
        except ImportError:
            # Custom grid-based layout if graphviz not available
            logger.warning("Graphviz not available, creating custom grid layout")
            pos = {}

            # Calculate optimal grid dimensions
            nodes = list(G.nodes())
            n_nodes = len(nodes)
            grid_width = int(np.sqrt(n_nodes) * 1.5)  # Make grid wider than tall
            grid_height = int(np.ceil(n_nodes / grid_width))

            # Position nodes in a grid pattern
            for i, node in enumerate(nodes):
                row = i // grid_width
                col = i % grid_width

                # Calculate position with spacing
                x = col * 3.0

                # Stagger rows to reduce direct overlaps
                y = row * 2.0
                if col % 2 == 1:
                    y += 0.5  # Stagger alternating columns

                pos[node] = (x, y)
    except Exception as e:
        logger.warning(f"Error creating layout: {e}. Using basic spring layout.")
        pos = nx.spring_layout(G, k=2.0, iterations=500, seed=42)

    # Set node colors based on priority
    node_colors = []
    for node in G.nodes():
        priority = task_attrs[node]["priority"]
        color_intensity = 0.2 + (priority / 5) * 0.8
        node_colors.append((0.1, 0.5, color_intensity))

    # Draw nodes with good size for readability
    nx.draw_networkx_nodes(
        G, pos, node_size=2000, node_color=node_colors, edgecolors="black", alpha=0.8
    )

    # Draw edges with hair pin routing to prevent overlaps
    edge_styles = []
    for edge in G.edges():
        # Calculate custom routing parameters for each edge
        source, target = edge
        source_x, source_y = pos[source]
        target_x, target_y = pos[target]

        # Calculate routing parameters - create a hairpin bend
        # If nodes are far apart horizontally, bend more
        dist = abs(target_x - source_x)
        rad = min(0.5, 0.1 + 0.05 * dist)

        # Alternate direction for odd/even edges to prevent overlaps
        index = list(G.edges()).index(edge)
        if index % 2 == 0:
            rad *= -1

        edge_styles.append(rad)

    # Draw edges with custom routing
    for i, edge in enumerate(G.edges()):
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[edge],
            width=1.5,
            arrowsize=15,
            arrowstyle="-|>",
            edge_color="gray",
            connectionstyle=f"arc3,rad={edge_styles[i]}",
        )

    # Draw node labels (IDs only)
    nx.draw_networkx_labels(
        G, pos, font_size=10, font_weight="bold", font_color="white"
    )

    # Draw additional task information with careful positioning
    if show_attributes:
        for node in G.nodes():
            x, y = pos[node]
            attr = task_attrs[node]

            # Display task index in a compact label below node
            plt.text(
                x,
                y - 0.2,
                f"#{attr['idx'] + 1} | Hrs: {attr['hours']} | Pri: {attr['priority']}",
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8),
                fontsize=9,
                fontweight="bold",
            )

            # Display skills in a separate location to reduce overlap
            plt.text(
                x,
                y - 0.4,
                f"Skills: {attr['skills']}",
                ha="center",
                va="center",
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    fc="lightyellow",
                    ec="goldenrod",
                    alpha=0.8,
                ),
                fontsize=8,
            )

    plt.title(title, fontsize=16, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()

    # Save or display
    if filename:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        logger.info(f"ACO network plot saved as {filename}")

    plt.show()


# Keep the other visualization functions (not modified)
def draw_dependency_graph(
    tasks: List[Task], filename: Optional[str] = None, layout: str = "spring"
) -> None:
    """
    Draw a graph of task dependencies.

    Args:
        tasks: List of tasks
        filename: File to save the plot (None to display only)
        layout: Graph layout algorithm ('spring', 'dot', 'circular', 'spectral', etc.)
    """
    # Create a directed graph
    G = nx.DiGraph()

    # Add tasks and dependencies to the graph
    task_info = {}  # Store task information for node labels
    for task in tasks:
        # Store task info (skills, hours, priority)
        task_info[task.id] = {
            "skills": ", ".join(task.required_skills),
            "hours": task.estimated_hours,
            "priority": task.priority,
            "due": task.due_date.strftime("%m/%d"),
        }

        # Add node with task ID
        G.add_node(task.id)

        # Add edges for dependencies
        for dep_id in task.dependencies:
            G.add_edge(dep_id, task.id)

    # Set up the plot
    plt.figure(figsize=(12, 9))

    # Choose layout based on parameter
    if layout == "dot":
        try:
            from networkx.drawing.nx_agraph import graphviz_layout

            pos = graphviz_layout(G, prog="dot")
        except ImportError:
            logger.warning("Graphviz not available, falling back to spring layout")
            pos = nx.spring_layout(G, seed=42)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "spectral":
        pos = nx.spectral_layout(G)
    else:  # Default to spring layout
        pos = nx.spring_layout(G, seed=42)

    # Set node colors based on priority
    node_colors = []
    for node in G.nodes():
        priority = task_info[node]["priority"]
        # Map priority 1-5 to color intensity
        color_intensity = 0.2 + (priority / 5) * 0.8
        node_colors.append((0.1, 0.5, color_intensity))

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, node_size=1500, node_color=node_colors, edgecolors="black", alpha=0.8
    )

    # Draw edges
    nx.draw_networkx_edges(
        G, pos, width=1.5, edge_color="gray", arrowsize=20, arrowstyle="-|>"
    )

    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")

    # Add node annotations (skills, hours, due date)
    for node, (x, y) in pos.items():
        info = task_info[node]
        label = f"Skills: {info['skills']}\nHours: {info['hours']}\nDue: {info['due']}"
        plt.annotate(
            label,
            xy=(x, y),
            xytext=(x, y - 0.1),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
            ha="center",
            va="top",
            fontsize=8,
        )

    plt.title("Task Dependency Graph", fontsize=16)
    plt.axis("off")
    plt.tight_layout()

    # Save or display
    if filename:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        logger.info(f"Dependency graph saved as {filename}")

    plt.show()
