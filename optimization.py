"""
Hyperparameter optimization using Optuna.

This module provides functions for optimizing the hyperparameters of the task scheduling
and assignment system using the Optuna optimization framework. It includes functions 
for defining the optimization objective, running experiments with specific parameter sets,
and managing the optimization process.
"""

import random
import math
import optuna
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from typing import List, Dict, Tuple, Any, Optional
from copy import deepcopy
from datetime import datetime
from joblib import Parallel, delayed

from models import Task, Employee
from schedulers.aco import ACOScheduler
from assignment.rl import CentralizedRL
from utils.generators import DataGenerator
from utils.validators import validate_tasks
from utils.logger import logger
from config import AppConfig


def objective(trial: optuna.Trial) -> Tuple[float, float]:
    """
    Optuna objective function for hyperparameter optimization.

    This is a multi-objective function that optimizes for:
    1. Maximizing the number of assigned tasks
    2. Minimizing the variance in rewards

    Args:
        trial: Optuna trial object

    Returns:
        Tuple[float, float]: Negated assigned count and reward variance
    """
    # Ensure consistent seed for fair comparison
    trial_seed = 42
    random.seed(trial_seed)
    np.random.seed(trial_seed)

    # Define hyperparameters to optimize
    aco_alpha = trial.suggest_float("ACO_ALPHA", 0.5, 3.0, step=0.1)
    aco_beta = trial.suggest_float("ACO_BETA", 0.5, 4.0, step=0.1)
    aco_rho = trial.suggest_float("ACO_RHO", 0.1, 0.9, step=0.05)
    aco_max_iter = trial.suggest_int("ACO_MAX_ITER", 10, 60, step=10)
    aco_num_ants = trial.suggest_int("ACO_NUM_ANTS", 5, 30, step=5)

    rl_initial_lr = trial.suggest_float("RL_INITIAL_LR", 0.01, 0.2, step=0.01)
    rl_lr_decay = trial.suggest_float("RL_LR_DECAY", 0.95, 0.999, step=0.001)
    rl_discount_factor = trial.suggest_float(
        "RL_DISCOUNT_FACTOR", 0.85, 0.99, step=0.01
    )
    rl_initial_epsilon = trial.suggest_float("RL_INITIAL_EPSILON", 0.1, 0.8, step=0.05)
    rl_epsilon_decay = trial.suggest_float("RL_EPSILON_DECAY", 0.9, 0.999, step=0.005)
    rl_min_epsilon = trial.suggest_float("RL_MIN_EPSILON", 0.01, 0.1, step=0.01)
    rl_use_ucb = trial.suggest_categorical("RL_USE_UCB", [True, False])

    if rl_use_ucb:
        rl_ucb_c = trial.suggest_float("RL_UCB_C", 0.5, 2.0, step=0.1)
    else:
        rl_ucb_c = 1.0

    reward_feasible = trial.suggest_float("REWARD_FEASIBLE", 5.0, 20.0)
    reward_infeasible = trial.suggest_float("REWARD_INFEASIBLE", -20.0, -5.0)
    reward_skip_base = -2.0  # Fixed
    reward_dep_bonus = 3.0  # Fixed

    balance_coeff = trial.suggest_float("BALANCE_COEFF", 0.1, 1.0, step=0.1)
    utilization_coeff = trial.suggest_float("UTILIZATION_COEFF", 1.0, 10.0, step=0.5)

    # Create hyperparameter dictionary
    hparams = {
        "ACO_ALPHA": aco_alpha,
        "ACO_BETA": aco_beta,
        "ACO_RHO": aco_rho,
        "ACO_MAX_ITER": aco_max_iter,
        "ACO_NUM_ANTS": aco_num_ants,
        "RL_INITIAL_LR": rl_initial_lr,
        "RL_LR_DECAY": rl_lr_decay,
        "RL_DISCOUNT_FACTOR": rl_discount_factor,
        "RL_INITIAL_EPSILON": rl_initial_epsilon,
        "RL_EPSILON_DECAY": rl_epsilon_decay,
        "RL_MIN_EPSILON": rl_min_epsilon,
        "RL_USE_UCB": rl_use_ucb,
        "RL_UCB_C": rl_ucb_c,
        "REWARD_FEASIBLE": reward_feasible,
        "REWARD_INFEASIBLE": reward_infeasible,
        "REWARD_SKIP_BASE": reward_skip_base,
        "REWARD_DEP_BONUS": reward_dep_bonus,
        "BALANCE_COEFF": balance_coeff,
        "UTILIZATION_COEFF": utilization_coeff,
        # Additional hyper-parameters for refinement (fixed)
        "COMP_LEARNING_RATE": 0.1,
        "COMP_DISCOUNT_FACTOR": 0.95,
        "COMP_EPSILON": 0.2,
        "COMP_MIN_EPSILON": 0.01,
        "COMP_EPSILON_DECAY": 0.995,
        "COMP_WORKLOAD_WEIGHT": 0.4,
        "COMP_SKILLS_WEIGHT": 0.3,
        "COMP_EFFICIENCY_WEIGHT": 0.3,
    }

    # Run experiment with the current hyperparameters
    return run_experiment_with_hparams(hparams)


def run_experiment_with_hparams(
    hparams: Dict[str, Any],
    num_tasks: int = 30,
    num_employees: int = 10,
    episodes: int = 300,
) -> Tuple[float, float]:
    """
    Run an experiment with specific hyperparameters.

    Args:
        hparams: Hyperparameter dictionary
        num_tasks: Number of tasks to generate
        num_employees: Number of employees to generate
        episodes: Number of training episodes

    Returns:
        Tuple[float, float]: Negated assigned count and reward variance
    """
    # Generate data
    data_generator = DataGenerator(seed=42)
    tasks, employees_master = data_generator.generate_scenario(num_tasks, num_employees)

    # Validate tasks
    if not validate_tasks(tasks):
        logger.error("Invalid task dependencies detected. Exiting experiment.")
        return 0.0, 0.0

    # Initialize ACO scheduler
    aco_scheduler = ACOScheduler(tasks, hparams)
    aco_seq = aco_scheduler.schedule()

    # Initialize RL models
    employees_for_q = deepcopy(employees_master)
    employees_for_sarsa = deepcopy(employees_master)

    rl_q_model = CentralizedRL(hparams, algorithm="q_learning")
    rl_q_model.employees = employees_for_q

    rl_sarsa_model = CentralizedRL(hparams, algorithm="sarsa")
    rl_sarsa_model.employees = employees_for_sarsa

    # Train models
    q_rewards = []
    sarsa_rewards = []

    for _ in range(episodes):
        # Run training episode
        q_r = rl_q_model.run_episode(aco_seq, training=True)
        s_r = rl_sarsa_model.run_episode(aco_seq, training=True)

        # Record rewards
        q_rewards.append(q_r)
        sarsa_rewards.append(s_r)

        # Decay epsilon
        rl_q_model.epsilon = max(
            hparams["RL_MIN_EPSILON"], rl_q_model.epsilon * hparams["RL_EPSILON_DECAY"]
        )
        rl_sarsa_model.epsilon = max(
            hparams["RL_MIN_EPSILON"],
            rl_sarsa_model.epsilon * hparams["RL_EPSILON_DECAY"],
        )

        # Decay learning rate
        rl_q_model.lr = max(0.01, rl_q_model.lr * rl_q_model.lr_decay)
        rl_sarsa_model.lr = max(0.01, rl_sarsa_model.lr * rl_sarsa_model.lr_decay)

    # Evaluate final assignments
    q_final_dict = rl_q_model.assign(aco_seq, employees_for_q)
    sarsa_final_dict = rl_sarsa_model.assign(aco_seq, employees_for_sarsa)

    # Calculate metrics
    assigned_q = sum(1 for v in q_final_dict.values() if v is not None)
    assigned_sarsa = sum(1 for v in sarsa_final_dict.values() if v is not None)

    # Average assigned count (higher is better, so negate for minimization)
    combined_assigned_count = (assigned_q + assigned_sarsa) / 2.0

    # Reward variance (lower is better)
    variance_q = np.var(q_rewards) if len(q_rewards) > 1 else 0.0
    variance_sarsa = np.var(sarsa_rewards) if len(sarsa_rewards) > 1 else 0.0
    combined_variance = (variance_q + variance_sarsa) / 2.0

    # Return objectives (negative assigned count for minimization)
    return -combined_assigned_count, combined_variance


def run_parallel_experiment(
    params: Dict[str, Any]
) -> Tuple[float, float, Dict[str, float]]:
    """
    Run a single experiment with given parameters for parallel execution.

    Args:
        params: Dictionary containing hyperparameters

    Returns:
        Tuple of (assigned count, reward variance, detailed metrics)
    """
    assigned_count, variance = run_experiment_with_hparams(params)

    # Calculate additional metrics for analysis
    additional_metrics = {
        "assigned_count": -assigned_count,  # Convert back to positive
        "reward_variance": variance,
        "combined_score": -assigned_count * 0.7 + variance * 0.3,  # Weighted score
    }

    return assigned_count, variance, additional_metrics


def optimize_hyperparameters(
    n_trials: int = 30,
    study_name: str = "task_scheduler_optimization",
    output_path: Optional[str] = "output/optimized_config.json",
    n_jobs: int = -1,
) -> Dict[str, Any]:
    """
    Run Optuna hyperparameter optimization and return best parameters.

    Args:
        n_trials: Number of optimization trials
        study_name: Name for the Optuna study
        output_path: Path to save optimized parameters, None to skip saving

    Returns:
        Dict[str, Any]: Best hyperparameters found
    """
    logger.info(f"Starting Optuna hyperparameter tuning with {n_trials} trials...")

    # Create multi-objective study
    study = optuna.create_study(
        directions=["minimize", "minimize"],  # Minimize both objectives
        sampler=optuna.samplers.NSGAIISampler(seed=42),
        study_name=study_name,
    )

    # Run optimization
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True, n_jobs=n_jobs)

    # Log results
    logger.info("Optuna best trials (Pareto Front):")
    for i, trial in enumerate(study.best_trials):
        logger.info(f"  Trial {i+1}: Values: {trial.values}, Params: {trial.params}")

    # Try to plot Pareto front
    try:
        import matplotlib.pyplot as plt

        fig = optuna.visualization.plot_pareto_front(study)
        plt.title("Pareto Front - Task Assignments vs. Reward Stability")
        plt.xlabel("Negative Assigned Tasks (lower is better)")
        plt.ylabel("Reward Variance (lower is better)")

        # Save figure
        os.makedirs("plots", exist_ok=True)
        plt.savefig("plots/pareto_front.png", dpi=150, bbox_inches="tight")
        logger.info("Pareto front plot saved as plots/pareto_front.png")
        plt.close()
    except Exception as e:
        logger.warning(f"Could not plot Pareto front: {e}")

    # Get best parameters from first Pareto-optimal solution
    best_trial = study.best_trials[0]
    best_params = best_trial.params

    # Create configuration with best parameters
    config = AppConfig()

    # Update relevant configs based on optimized parameters
    if "ACO_ALPHA" in best_params:
        config.aco.alpha = best_params["ACO_ALPHA"]
    if "ACO_BETA" in best_params:
        config.aco.beta = best_params["ACO_BETA"]
    if "ACO_RHO" in best_params:
        config.aco.rho = best_params["ACO_RHO"]
    if "ACO_MAX_ITER" in best_params:
        config.aco.max_iter = best_params["ACO_MAX_ITER"]
    if "ACO_NUM_ANTS" in best_params:
        config.aco.num_ants = best_params["ACO_NUM_ANTS"]

    # Update RL config
    if "RL_INITIAL_LR" in best_params:
        config.rl.initial_lr = best_params["RL_INITIAL_LR"]
    if "RL_LR_DECAY" in best_params:
        config.rl.lr_decay = best_params["RL_LR_DECAY"]
    if "RL_DISCOUNT_FACTOR" in best_params:
        config.rl.discount_factor = best_params["RL_DISCOUNT_FACTOR"]
    if "RL_INITIAL_EPSILON" in best_params:
        config.rl.initial_epsilon = best_params["RL_INITIAL_EPSILON"]
    if "RL_MIN_EPSILON" in best_params:
        config.rl.min_epsilon = best_params["RL_MIN_EPSILON"]
    if "RL_EPSILON_DECAY" in best_params:
        config.rl.epsilon_decay = best_params["RL_EPSILON_DECAY"]
    if "RL_USE_UCB" in best_params:
        config.rl.use_ucb = best_params["RL_USE_UCB"]
    if "RL_UCB_C" in best_params:
        config.rl.ucb_c = best_params["RL_UCB_C"]

    # Update reward config
    if "REWARD_FEASIBLE" in best_params:
        config.reward.feasible = best_params["REWARD_FEASIBLE"]
    if "REWARD_INFEASIBLE" in best_params:
        config.reward.infeasible = best_params["REWARD_INFEASIBLE"]
    if "BALANCE_COEFF" in best_params:
        config.reward.balance_coeff = best_params["BALANCE_COEFF"]
    if "UTILIZATION_COEFF" in best_params:
        config.reward.utilization_coeff = best_params["UTILIZATION_COEFF"]

    # Save optimized configuration if output path provided
    if output_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Convert config to dictionary and save as JSON
        config_dict = config.to_dict()
        with open(output_path, "w") as f:
            json.dump(config_dict, f, indent=2)
        logger.info(f"Optimized configuration saved to {output_path}")

    return config.to_dict()


def sensitivity_analysis(
    base_params: Dict[str, Any],
    param_ranges: Dict[str, List[float]],
    metric_function: callable,
    n_jobs: int = -1,
) -> Dict[str, List[Tuple[float, float]]]:
    """
    Perform sensitivity analysis on selected parameters.

    Args:
        base_params: Base hyperparameter values
        param_ranges: Dictionary mapping parameter names to lists of values to test
        metric_function: Function that takes parameters and returns a metric value
        n_jobs: Number of parallel jobs (default: -1 uses all cores)

    Returns:
        Dictionary mapping parameter names to lists of (param_value, metric_value) pairs
    """
    results = {}

    for param_name, param_values in param_ranges.items():
        logger.info(f"Running sensitivity analysis for {param_name}")
        param_results = []

        # Create parameter sets, one for each value to test
        param_sets = []
        for value in param_values:
            # Create a copy of base parameters and update the target parameter
            params = base_params.copy()
            params[param_name] = value
            param_sets.append(params)

        # Run experiments in parallel
        experiment_results = Parallel(n_jobs=n_jobs)(
            delayed(metric_function)(params) for params in param_sets
        )

        # Store results
        for value, result in zip(param_values, experiment_results):
            param_results.append((value, result))

        results[param_name] = param_results

    return results


def plot_sensitivity_results(
    sensitivity_results: Dict[str, List[Tuple[float, float]]],
    output_dir: str = "plots",
    filename_prefix: str = "sensitivity",
) -> None:
    """
    Plot the results of sensitivity analysis.

    Args:
        sensitivity_results: Results from sensitivity_analysis
        output_dir: Directory to save plots
        filename_prefix: Prefix for output filenames
    """
    os.makedirs(output_dir, exist_ok=True)

    for param_name, results in sensitivity_results.items():
        # Extract values for plotting
        param_values = [r[0] for r in results]
        metric_values = [r[1] for r in results]

        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(param_values, metric_values, "o-", linewidth=2, markersize=8)
        plt.xlabel(f"Parameter Value: {param_name}")
        plt.ylabel("Performance Metric")
        plt.title(f"Sensitivity Analysis for {param_name}")
        plt.grid(True, alpha=0.3)

        # Add trend line
        z = np.polyfit(param_values, metric_values, 1)
        p = np.poly1d(z)
        plt.plot(param_values, p(param_values), "r--", alpha=0.7)

        # Save the plot
        output_path = f"{output_dir}/{filename_prefix}_{param_name}.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Sensitivity plot for {param_name} saved to {output_path}")