import streamlit as st
import sys
import os
import io
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from copy import deepcopy
import logging
import json
import base64
from PIL import Image

# Import the required modules from the provided code
# Note: These would need to be in the same directory as the Streamlit app
from models import Task, Employee
from config import AppConfig
from utils.logger import setup_logger
from utils.generators import DataGenerator
from utils.validators import validate_tasks
from schedulers.aco import ACOScheduler
from assignment.rl import CentralizedRL
from assignment.greedy import GreedyAssigner
from assignment.refinement import RefinementRL
from analysis.metrics import compute_detailed_metrics, compare_assignment_methods
from visualization import (
    draw_dependency_graph,
    draw_aco_network,
    plot_gantt_chart_for_assignments,
    plot_rewards_across_episodes,
    plot_epsilon_decay,
    plot_metric_bar_chart,
    create_comparison_dashboard,
)

# Set page config
st.set_page_config(
    page_title="Task Scheduling and Assignment System",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Custom class to capture stdout and stderr for logging display
class StreamlitLogger(io.StringIO):
    def __init__(self, container):
        super().__init__()
        self.container = container
        self.log_content = ""

    def write(self, text):
        self.log_content += text
        self.container.markdown(f"```\n{self.log_content}\n```")
        return len(text)

    def flush(self):
        pass


# Function to run the simulation with progress updates
def run_simulation(num_tasks, num_employees, num_episodes, container, log_container):
    # Capture logs
    logger = setup_logger(level=logging.INFO)
    handler = logging.StreamHandler(StreamlitLogger(log_container))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.handlers = [handler]

    # Create directories for outputs
    os.makedirs("plots", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Display progress message
    progress_text = container.empty()
    progress_bar = container.progress(0)
    progress_text.text("Initializing simulation...")

    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Load default configuration
    config = AppConfig()
    config.refinement.num_episodes = num_episodes

    # Generate test data
    progress_text.text("Generating test data...")
    progress_bar.progress(10)

    data_generator = DataGenerator(seed=config.seed)
    tasks, employees = data_generator.generate_scenario(num_tasks, num_employees)

    # Validate tasks for circular dependencies
    if not validate_tasks(tasks):
        container.error("Invalid task dependencies. Exiting simulation.")
        return None

    # Initialize ACO scheduler
    progress_text.text("Initializing ACO scheduler...")
    progress_bar.progress(20)

    aco_scheduler = ACOScheduler(tasks, config.to_dict())

    # Generate task sequence
    progress_text.text("Generating task sequence using ACO...")
    progress_bar.progress(30)

    aco_seq = aco_scheduler.schedule()

    # Create deep copies of employees for different algorithms
    employees_for_q = deepcopy(employees)
    employees_for_sarsa = deepcopy(employees)
    employees_for_greedy = deepcopy(employees)

    # Initialize RL models
    progress_text.text("Initializing RL models...")
    progress_bar.progress(40)

    rl_q_model = CentralizedRL(config.to_dict(), algorithm="q_learning")
    rl_q_model.employees = employees_for_q

    rl_sarsa_model = CentralizedRL(config.to_dict(), algorithm="sarsa")
    rl_sarsa_model.employees = employees_for_sarsa

    # Initialize Greedy assigner
    greedy_assigner = GreedyAssigner(load_penalty=1.0)

    # Train RL models
    progress_text.text(f"Training RL models for {config.refinement.num_episodes} episodes...")
    progress_bar.progress(50)

    epsilon_history_q = []
    epsilon_history_sarsa = []
    q_reward_history = []
    sarsa_reward_history = []

    for episode in range(config.refinement.num_episodes):
        # Update progress every 10% of episodes
        if episode % max(1, config.refinement.num_episodes // 10) == 0:
            progress = 50 + (episode / config.refinement.num_episodes) * 20
            progress_bar.progress(int(progress))
            progress_text.text(f"Training episodes: {episode + 1}/{config.refinement.num_episodes}")

        # Periodically update the task sequence using ACO
        if (episode + 1) % 300 == 0:
            new_seq = aco_scheduler.schedule()
            q_reward = rl_q_model.run_episode(new_seq, training=True)
            sarsa_reward = rl_sarsa_model.run_episode(new_seq, training=True)
        else:
            q_reward = rl_q_model.run_episode(aco_seq, training=True)
            sarsa_reward = rl_sarsa_model.run_episode(aco_seq, training=True)

        # Record rewards and epsilon values
        q_reward_history.append(q_reward)
        sarsa_reward_history.append(sarsa_reward)
        epsilon_history_q.append(rl_q_model.epsilon)
        epsilon_history_sarsa.append(rl_sarsa_model.epsilon)

        # Epsilon decay schedule
        if episode < 0.7 * config.refinement.num_episodes:
            fraction = episode / (0.7 * config.refinement.num_episodes)

            # Linear decay
            rl_q_model.epsilon = rl_q_model.initial_epsilon - fraction * (
                    rl_q_model.initial_epsilon - rl_q_model.min_epsilon
            )
            rl_sarsa_model.epsilon = rl_sarsa_model.initial_epsilon - fraction * (
                    rl_sarsa_model.initial_epsilon - rl_sarsa_model.min_epsilon
            )
        else:
            # Minimum epsilon for exploration
            rl_q_model.epsilon = rl_q_model.min_epsilon
            rl_sarsa_model.epsilon = rl_sarsa_model.min_epsilon

        # Learning rate decay
        rl_q_model.lr = max(0.01, rl_q_model.lr * rl_q_model.lr_decay)
        rl_sarsa_model.lr = max(0.01, rl_sarsa_model.lr * rl_sarsa_model.lr_decay)

    # Get initial assignments from each model
    progress_text.text("Generating initial assignments...")
    progress_bar.progress(70)

    assigned_q_before = rl_q_model.assign(aco_seq, employees_for_q)
    assigned_sarsa_before = rl_sarsa_model.assign(aco_seq, employees_for_sarsa)
    assigned_greedy_before = greedy_assigner.assign(aco_seq, employees_for_greedy)

    # Calculate detailed metrics for initial assignments
    progress_text.text("Computing metrics for initial assignments...")
    progress_bar.progress(75)

    metrics_q_before = compute_detailed_metrics(
        aco_seq, assigned_q_before, employees_for_q
    )
    metrics_sarsa_before = compute_detailed_metrics(
        aco_seq, assigned_sarsa_before, employees_for_sarsa
    )
    metrics_greedy_before = compute_detailed_metrics(
        aco_seq, assigned_greedy_before, employees_for_greedy
    )

    # Refine assignments using RefinementRL
    progress_text.text("Refining assignments using RefinementRL...")
    progress_bar.progress(80)

    refinement_rl_q = RefinementRL(
        tasks=aco_seq,
        employees=employees,
        num_episodes=config.refinement.num_episodes,
        alpha=config.refinement.learning_rate,
        gamma=config.refinement.discount_factor,
        epsilon=config.refinement.epsilon,
        epsilon_decay=config.refinement.epsilon_decay,
        min_epsilon=config.refinement.min_epsilon,
    )

    optimized_q = refinement_rl_q.optimize_assignments(assigned_q_before)

    refinement_rl_sarsa = RefinementRL(
        tasks=aco_seq,
        employees=employees,
        num_episodes=config.refinement.num_episodes,
        alpha=config.refinement.learning_rate,
        gamma=config.refinement.discount_factor,
        epsilon=config.refinement.epsilon,
        epsilon_decay=config.refinement.epsilon_decay,
        min_epsilon=config.refinement.min_epsilon,
    )

    optimized_sarsa = refinement_rl_sarsa.optimize_assignments(assigned_sarsa_before)

    # Calculate detailed metrics for refined assignments
    progress_text.text("Computing metrics for refined assignments...")
    progress_bar.progress(85)

    metrics_q_after = compute_detailed_metrics(aco_seq, optimized_q, employees)
    metrics_sarsa_after = compute_detailed_metrics(aco_seq, optimized_sarsa, employees)

    # Compare all methods
    all_methods_metrics = {
        "Q-learning Before": metrics_q_before,
        "SARSA Before": metrics_sarsa_before,
        "Greedy": metrics_greedy_before,
        "Q-learning After": metrics_q_after,
        "SARSA After": metrics_sarsa_after,
    }

    best_methods = compare_assignment_methods(all_methods_metrics)

    # Create visualizations
    progress_text.text("Generating visualizations...")
    progress_bar.progress(90)

    # Draw visualizations for all methods
    draw_dependency_graph(tasks, "plots/task_dependencies.png")
    draw_aco_network(aco_seq, "ACO Task Sequence", "plots/aco_network.png")

    plot_gantt_chart_for_assignments(
        "Q-learning Before",
        aco_seq,
        assigned_q_before,
        employees_for_q,
        filename="plots/gantt_q_before.png",
    )
    plot_gantt_chart_for_assignments(
        "SARSA Before",
        aco_seq,
        assigned_sarsa_before,
        employees_for_sarsa,
        filename="plots/gantt_sarsa_before.png",
    )
    plot_gantt_chart_for_assignments(
        "Greedy",
        aco_seq,
        assigned_greedy_before,
        employees_for_greedy,
        filename="plots/gantt_greedy.png",
    )
    plot_gantt_chart_for_assignments(
        "Refined Q-learning",
        aco_seq,
        optimized_q,
        employees,
        filename="plots/gantt_q_after.png",
    )
    plot_gantt_chart_for_assignments(
        "Refined SARSA",
        aco_seq,
        optimized_sarsa,
        employees,
        filename="plots/gantt_sarsa_after.png",
    )

    plot_epsilon_decay(
        epsilon_history_q, epsilon_history_sarsa, filename="plots/epsilon_decay.png"
    )
    plot_rewards_across_episodes(
        q_reward_history,
        sarsa_reward_history,
        window_size=50,
        episode_interval=50,
        filename="plots/rewards_history.png",
    )

    for metric in metrics_q_before.keys():
        metric_data = {
            method: metrics[metric] for method, metrics in all_methods_metrics.items()
        }
        plot_metric_bar_chart(
            metric, metric_data, filename=f"plots/{metric}_comparison.png"
        )

    # Create comparison dashboard
    create_comparison_dashboard(all_methods_metrics, filename="plots/dashboard.png")

    progress_text.text("Simulation completed successfully!")
    progress_bar.progress(100)

    # Return the results
    return {
        "tasks": aco_seq,
        "employees": employees,
        "assignments": {
            "q_before": assigned_q_before,
            "sarsa_before": assigned_sarsa_before,
            "greedy": assigned_greedy_before,
            "q_after": optimized_q,
            "sarsa_after": optimized_sarsa,
        },
        "metrics": all_methods_metrics,
        "best_methods": best_methods,
        "training": {
            "q_rewards": q_reward_history,
            "sarsa_rewards": sarsa_reward_history,
            "q_epsilon": epsilon_history_q,
            "sarsa_epsilon": epsilon_history_sarsa,
        },
    }


# Function to get an image as base64 for display
def get_image_as_base64(file_path):
    with open(file_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


# Function to display images with download buttons
def display_image_with_download(title, image_path, container):
    try:
        img = Image.open(image_path)
        container.subheader(title)
        container.image(img)

        # Add download button
        with open(image_path, "rb") as file:
            btn = container.download_button(
                label=f"Download {title}",
                data=file,
                file_name=os.path.basename(image_path),
                mime="image/png"
            )
    except Exception as e:
        container.warning(f"Could not display {title}: {e}")


# Main Streamlit app
def main():
    st.title("Task Scheduling and Assignment System")
    st.markdown("""
    This application simulates task scheduling using Ant Colony Optimization (ACO) 
    and task assignment using Reinforcement Learning (RL) and Greedy approaches.
    """)

    # Sidebar for inputs
    st.sidebar.header("Simulation Parameters")

    num_tasks = st.sidebar.slider(
        "Number of Tasks",
        min_value=10,
        max_value=50,
        value=30,
        help="The number of tasks to generate for the simulation"
    )

    num_employees = st.sidebar.slider(
        "Number of Employees",
        min_value=5,
        max_value=20,
        value=10,
        help="The number of employees to generate for the simulation"
    )

    num_episodes = st.sidebar.slider(
        "Training Episodes",
        min_value=100,
        max_value=2000,
        value=500,
        step=100,
        help="The number of episodes for training the RL models"
    )

    # Run simulation button
    if st.sidebar.button("Run Simulation", type="primary"):
        # Create tabs for different outputs
        tab1, tab2, tab3, tab4 = st.tabs(["Logs", "Results", "Gantt Charts", "Metrics"])

        # Log container in the first tab
        log_container = tab1.empty()

        # Progress container in the first tab
        progress_container = tab1.container()

        # Run simulation
        results = run_simulation(
            num_tasks,
            num_employees,
            num_episodes,
            progress_container,
            log_container
        )

        if results:
            # Display results in the second tab
            with tab2:
                st.subheader("Task Dependency Graph")
                st.image("plots/task_dependencies.png")

                st.subheader("ACO Task Sequence")
                st.image("plots/aco_network.png")

                st.subheader("Training Results")
                col1, col2 = st.columns(2)

                col1.image("plots/rewards_history.png", caption="Rewards History")
                col2.image("plots/epsilon_decay.png", caption="Epsilon Decay")

                st.subheader("Best Methods by Metric")
                best_methods_df = pd.DataFrame(
                    list(results["best_methods"].items()),
                    columns=["Metric", "Best Method"]
                )
                st.dataframe(best_methods_df, use_container_width=True)

            # Display Gantt charts in the third tab
            with tab3:
                st.subheader("Gantt Charts for Task Assignments")
                col1, col2 = st.columns(2)

                display_image_with_download(
                    "Q-learning Before Refinement",
                    "plots/gantt_q_before.png",
                    col1
                )
                display_image_with_download(
                    "SARSA Before Refinement",
                    "plots/gantt_sarsa_before.png",
                    col2
                )

                col1, col2 = st.columns(2)
                display_image_with_download(
                    "Greedy Assignment",
                    "plots/gantt_greedy.png",
                    col1
                )
                display_image_with_download(
                    "Q-learning After Refinement",
                    "plots/gantt_q_after.png",
                    col2
                )

                display_image_with_download(
                    "SARSA After Refinement",
                    "plots/gantt_sarsa_after.png",
                    st
                )

            # Display metrics in the fourth tab
            with tab4:
                st.subheader("Performance Metrics Comparison")
                st.image("plots/dashboard.png")

                # Display individual metric comparisons
                st.subheader("Individual Metric Comparisons")
                metrics = ["workload_balance_ratio", "assigned_priority_ratio",
                           "dependency_completion_rate", "resource_utilization",
                           "task_coverage"]

                for i in range(0, len(metrics), 2):
                    cols = st.columns(2)
                    for j in range(2):
                        if i + j < len(metrics):
                            metric = metrics[i + j]
                            image_path = f"plots/{metric}_comparison.png"
                            if os.path.exists(image_path):
                                display_image_with_download(
                                    metric.replace("_", " ").title(),
                                    image_path,
                                    cols[j]
                                )

                # Display metrics as tables
                st.subheader("Metrics Data")

                # Convert metrics to DataFrame for display
                metrics_df = pd.DataFrame(results["metrics"])
                st.dataframe(metrics_df, use_container_width=True)

                # Add download button for metrics as CSV
                csv = metrics_df.to_csv(index=False)
                st.download_button(
                    label="Download Metrics as CSV",
                    data=csv,
                    file_name="metrics.csv",
                    mime="text/csv",
                )
        else:
            st.error("Simulation failed. Please check the logs for details.")

    # About section in the sidebar
    st.sidebar.markdown("---")
    st.sidebar.header("About")
    st.sidebar.info("""
    This application demonstrates a task scheduling and assignment system using:
    * Ant Colony Optimization (ACO) for task scheduling
    * Reinforcement Learning (Q-learning and SARSA) for task assignment
    * Greedy algorithm for comparison
    * Refinement algorithms to optimize assignments

    The system optimizes for workload balance, resource utilization, and task priorities.
    """)


if __name__ == "__main__":
    main()