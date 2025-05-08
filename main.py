# Directory: main.py
"""
Main application entry point for task scheduling system.
"""
import os
import argparse
import random
import numpy as np
import json
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import time
from copy import deepcopy

from models import Task, Employee
from config import AppConfig
from utils.logger import logger, setup_logger
from utils.generators import DataGenerator
from utils.validators import validate_tasks, validate_assignments
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
    # plot_assigned_priority_violin,
    export_to_excel,
    create_comparison_dashboard,
)
from optimization import optimize_hyperparameters


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """
    Load configuration from file or use defaults.

    Args:
        config_path: Path to configuration file

    Returns:
        AppConfig: Application configuration
    """
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config_dict = json.load(f)
            return AppConfig.from_dict(config_dict)
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
            logger.info("Using default configuration.")
            return AppConfig()
    else:
        return AppConfig()


def setup_directories() -> None:
    """Create necessary directories for outputs."""
    dirs = ["plots", "output", "logs"]
    for directory in dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")


def run_simulation(config: AppConfig) -> Dict[str, Any]:
    """
    Run a complete task scheduling and assignment simulation.

    Args:
        config: Application configuration

    Returns:
        Dict[str, Any]: Simulation results
    """
    # Set up random seeds for reproducibility
    random.seed(config.seed)
    np.random.seed(config.seed)

    # Generate test data
    data_generator = DataGenerator(seed=config.seed)
    tasks, employees = data_generator.generate_scenario(40, 20)

    # Validate tasks for circular dependencies
    if not validate_tasks(tasks):
        logger.error("Invalid task dependencies. Exiting simulation.")
        return {}

    # Initialize ACO scheduler
    logger.info("Initializing ACO scheduler...")
    aco_scheduler = ACOScheduler(tasks, config.to_dict())

    # Generate task sequence
    logger.info("Generating task sequence using ACO...")
    aco_seq = aco_scheduler.schedule()
    logger.info(f"Task sequence generated: {[t.id for t in aco_seq]}")

    # Create deep copies of employees for different algorithms
    employees_for_q = deepcopy(employees)
    employees_for_sarsa = deepcopy(employees)
    employees_for_greedy = deepcopy(employees)

    # Initialize RL models
    logger.info("Initializing RL models...")
    rl_q_model = CentralizedRL(config.to_dict(), algorithm="q_learning")
    rl_q_model.employees = employees_for_q

    rl_sarsa_model = CentralizedRL(config.to_dict(), algorithm="sarsa")
    rl_sarsa_model.employees = employees_for_sarsa

    # Initialize Greedy assigner
    logger.info("Initializing Greedy assigner...")
    greedy_assigner = GreedyAssigner(load_penalty=1.0)

    # Train RL models
    logger.info(f"Training RL models for {config.refinement.num_episodes} episodes...")

    epsilon_history_q = []
    epsilon_history_sarsa = []
    q_reward_history = []
    sarsa_reward_history = []

    for episode in range(config.refinement.num_episodes):
        # Periodically update the task sequence using ACO
        if (episode + 1) % 300 == 0:
            logger.info(f"Episode {episode+1}: Updating task sequence using ACO...")
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

        # Log progress
        if (episode + 1) % 300 == 0 or episode == 0:
            logger.info(
                f"Episode {episode+1}/{config.refinement.num_episodes}: "
                f"Q-learning reward = {q_reward:.2f}, "
                f"SARSA reward = {sarsa_reward:.2f}, "
                f"Epsilon = {rl_q_model.epsilon:.3f}"
            )

    # Get initial assignments from each model
    logger.info("Generating initial assignments...")
    assigned_q_before = rl_q_model.assign(aco_seq, employees_for_q)
    assigned_sarsa_before = rl_sarsa_model.assign(aco_seq, employees_for_sarsa)
    assigned_greedy_before = greedy_assigner.assign(aco_seq, employees_for_greedy)

    # Calculate detailed metrics for initial assignments
    logger.info("Computing metrics for initial assignments...")
    metrics_q_before = compute_detailed_metrics(
        aco_seq, assigned_q_before, employees_for_q
    )
    metrics_sarsa_before = compute_detailed_metrics(
        aco_seq, assigned_sarsa_before, employees_for_sarsa
    )
    metrics_greedy_before = compute_detailed_metrics(
        aco_seq, assigned_greedy_before, employees_for_greedy
    )

    # Log initial metrics
    logger.info(f"Q-learning initial metrics: {metrics_q_before}")
    logger.info(f"SARSA initial metrics: {metrics_sarsa_before}")
    logger.info(f"Greedy initial metrics: {metrics_greedy_before}")

    # Refine assignments using RefinementRL
    logger.info("Refining assignments using RefinementRL...")
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
    logger.info("Computing metrics for refined assignments...")
    metrics_q_after = compute_detailed_metrics(aco_seq, optimized_q, employees)
    metrics_sarsa_after = compute_detailed_metrics(aco_seq, optimized_sarsa, employees)

    # Log refined metrics
    logger.info(f"Refined Q-learning metrics: {metrics_q_after}")
    logger.info(f"Refined SARSA metrics: {metrics_sarsa_after}")

    # Compare all methods
    all_methods_metrics = {
        "Q-learning Before": metrics_q_before,
        "SARSA Before": metrics_sarsa_before,
        "Greedy": metrics_greedy_before,
        "Q-learning After": metrics_q_after,
        "SARSA After": metrics_sarsa_after,
    }

    best_methods = compare_assignment_methods(all_methods_metrics)
    logger.info(f"Best methods by metric: {best_methods}")

    # Create visualizations
    logger.info("Generating visualizations...")

    # Task dependency graph
    draw_dependency_graph(tasks, "plots/task_dependencies.png")

    # ACO network visualization
    draw_aco_network(aco_seq, "ACO Task Sequence", "plots/aco_network.png")

    # Gantt charts
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

    # Training metrics plots
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

    # Metric plots
    for metric in metrics_q_before.keys():
        metric_data = {
            method: metrics[metric] for method, metrics in all_methods_metrics.items()
        }
        plot_metric_bar_chart(
            metric, metric_data, filename=f"plots/{metric}_comparison.png"
        )

    # Assignment priority distribution
    assignments = {
        "Q-learning Before": assigned_q_before,
        "SARSA Before": assigned_sarsa_before,
        "Greedy": assigned_greedy_before,
        "Q-learning After": optimized_q,
        "SARSA After": optimized_sarsa,
    }
    # plot_assigned_priority_violin(
    #     assignments, aco_seq, filename="plots/priority_distribution.png"
    # )

    # Dashboard visualization
    create_comparison_dashboard(all_methods_metrics, filename="plots/dashboard.png")

    # Export to Excel
    logger.info("Exporting results to Excel...")
    export_to_excel(
        "output/assignment_report.xlsx",
        employees_q=employees_for_q,
        employees_sarsa=employees_for_sarsa,
        employees_greedy=employees_for_greedy,
        tasks=aco_seq,
        assigned_q_before=assigned_q_before,
        assigned_sarsa_before=assigned_sarsa_before,
        assigned_greedy_before=assigned_greedy_before,
        assigned_q_after=optimized_q,
        assigned_sarsa_after=optimized_sarsa,
    )

    # Return comprehensive results
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


def main():

    """Main application entry point."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Task Scheduling and Assignment System"
    )
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument(
        "--optimize", action="store_true", help="Run hyperparameter optimization"
    )
    parser.add_argument(
        "--trials", type=int, default=30, help="Number of optimization trials"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set the logging level",
    )
    args = parser.parse_args()

    # Set up logging
    log_level = getattr(logging, args.log_level)
    setup_logger(level=log_level)

    # Create necessary directories
    setup_directories()

    # Load configuration
    config = load_config(args.config)

    # Run hyperparameter optimization if requested
    if args.optimize:
        logger.info(f"Running hyperparameter optimization with {args.trials} trials...")
        optimized_config = optimize_hyperparameters(n_trials=args.trials)

        # Save optimized configuration
        config_path = "output/optimized_config.json"
        with open(config_path, "w") as f:
            json.dump(optimized_config, f, indent=2)
        logger.info(f"Optimized configuration saved to {config_path}")

        # Update current config
        config = AppConfig.from_dict(optimized_config)

    # Run simulation
    logger.info("Starting task scheduling simulation...")
    results = run_simulation(config)

    if results:
        # Save results summary
        summary_path = "output/simulation_summary.json"

        # Extract serializable parts of the results
        serializable_results = {
            "assignments": {
                k: {task_id: emp_id for task_id, emp_id in v.items()}
                for k, v in results["assignments"].items()
            },
            "metrics": results["metrics"],
            "best_methods": results["best_methods"],
        }

        with open(summary_path, "w") as f:
            json.dump(serializable_results, f, indent=2)

        logger.info(f"Simulation summary saved to {summary_path}")
        logger.info("Simulation completed successfully!")
    else:
        logger.error("Simulation failed!")


if __name__ == "__main__":
    start_time = time.time()
    import logging

    main()
    print("--- %s seconds ---" % round((time.time() - start_time),2))
