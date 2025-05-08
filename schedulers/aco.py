# Directory: schedulers/aco.py
"""
Ant Colony Optimization Scheduler Implementation.
"""
import random
from datetime import datetime

import numpy as np
from typing import List, Dict, Any, Tuple
from models import Task
from schedulers.interfaces import Scheduler
from utils.logger import logger



class ACOScheduler(Scheduler):
    """
    Scheduler using Ant Colony Optimization algorithm.

    This implementation uses ant colony optimization to find an optimal
    task sequence that respects dependencies and optimizes for priority,
    due dates, and estimated effort.
    """

    def __init__(self, tasks: List[Task], config: Dict[str, Any]):
        """
        Initialize ACO scheduler.

        Args:
            tasks: List of tasks to schedule
            config: Configuration dictionary with ACO parameters
        """
        self.tasks = tasks
        self.n_tasks = len(tasks)
        self.alpha = config["ACO_ALPHA"]
        self.beta = config["ACO_BETA"]
        self.rho = config["ACO_RHO"]
        self.num_ants = config["ACO_NUM_ANTS"]
        self.max_iter = config["ACO_MAX_ITER"]
        self.heuristic = self._compute_heuristic()
        self.pheromone = np.ones((self.n_tasks, self.n_tasks), dtype=np.float32)
        self.task_indices = {task.id: i for i, task in enumerate(tasks)}
        self.best_sequence = None
        self.best_score = float("inf")

    def _compute_heuristic(self) -> np.ndarray:
        """
        Compute the heuristic matrix for ACO.

        The heuristic represents the desirability of scheduling one task
        after another, based on priority, estimated hours, due date, and
        dependency relationships.

        Returns:
            np.ndarray: Matrix of heuristic values
        """
        mat = np.zeros((self.n_tasks, self.n_tasks), dtype=np.float32)

        # Calculate normalization factors
        max_hours = max(t.estimated_hours for t in self.tasks)
        max_priority = max(t.priority for t in self.tasks)
        earliest_due = min(t.due_date for t in self.tasks)
        latest_due = max(t.due_date for t in self.tasks)
        max_day_span = max(1, (latest_due - earliest_due).days)

        # Pre-compute task attributes
        hours_arr = np.array([t.estimated_hours for t in self.tasks], dtype=np.float32)
        prio_arr = np.array([t.priority for t in self.tasks], dtype=np.float32)
        due_arr = np.array(
            [(t.due_date - earliest_due).days for t in self.tasks], dtype=np.float32
        )

        # Fill the heuristic matrix
        for i in range(self.n_tasks):
            t1 = self.tasks[i]
            for j in range(self.n_tasks):
                if i == j:
                    mat[i, j] = 0.0  # No self-transitions
                else:
                    hr_ratio = hours_arr[j] / max_hours
                    pr_ratio = prio_arr[j] / max_priority
                    day_offset = due_arr[j] / max_day_span

                    # Dependency factor: strongly discourage scheduling a task before its dependencies
                    dep_factor = 0.05 if self.tasks[j].id in t1.dependencies else 1.0

                    # Combine factors with appropriate weights
                    mat[i, j] = (
                        0.3 * pr_ratio  # Higher priority is better
                        + 0.3 * (1.0 - hr_ratio)  # Lower hours is better
                        + 0.2 * (1.0 - day_offset)  # Earlier due date is better
                        + 0.2 * dep_factor  # Respect dependencies
                    )

        return mat

    def _construct_solution(self) -> np.ndarray:
        """
        Construct a solution (sequence of tasks) for one ant.

        Returns:
            np.ndarray: Sequence of task indices
        """
        sequence = []
        visited = set()

        # Start with tasks that have no dependencies
        available = [i for i, t in enumerate(self.tasks) if not t.dependencies]

        while len(sequence) < self.n_tasks:
            if not available:
                # This should not happen if the tasks are properly validated
                raise ValueError(
                    "No valid sequence found – check for circular dependencies!"
                )

            # Select the next task probabilistically based on pheromone and heuristic
            if random.random() < 0.9:  # Exploitation vs exploration balance
                # Calculate probabilities for each available task
                probs = []

                if sequence:  # If we have at least one task in the sequence
                    last_task = sequence[-1]
                    for task_idx in available:
                        # Calculate transition probability
                        pheromone_val = self.pheromone[last_task, task_idx]
                        heuristic_val = self.heuristic[last_task, task_idx]

                        # Apply ACO formula: τ^α * η^β
                        prob = (pheromone_val**self.alpha) * (heuristic_val**self.beta)
                        probs.append((task_idx, prob))

                    # Normalize probabilities
                    total = sum(p[1] for p in probs)
                    if total > 0:
                        probs = [(idx, p / total) for idx, p in probs]

                        # Select next task using roulette wheel selection
                        r = random.random()
                        cumulative = 0
                        selected = available[0]  # Default

                        for idx, prob in probs:
                            cumulative += prob
                            if r <= cumulative:
                                selected = idx
                                break
                    else:
                        # If all probabilities are zero (e.g., no pheromone yet)
                        selected = random.choice(available)
                else:
                    # First task selection is random from available
                    selected = random.choice(available)
            else:
                # Pure exploration: random choice
                selected = random.choice(available)

            # Add the selected task to the sequence
            sequence.append(selected)
            visited.add(self.tasks[selected].id)

            # Update available tasks based on dependencies
            available = [
                i
                for i, t in enumerate(self.tasks)
                if i not in sequence and t.dependencies.issubset(visited)
            ]

        return np.array(sequence)

    def _evaluate_sequence(self, seq: np.ndarray) -> float:
        """
        Evaluate a task sequence.

        Lower score is better. The evaluation considers:
        - Dependency violations
        - Overdue tasks
        - Priority-weighted completion times

        Args:
            seq: Sequence of task indices

        Returns:
            float: Evaluation score (lower is better)
        """
        score = 0.0
        current_time = 0
        completed_tasks = set()

        # Setup time reference
        start_reference = int(datetime(2023, 10, 1).timestamp())

        for idx in seq:
            task = self.tasks[idx]

            # Check for dependency violations
            violated_deps = sum(
                1 for dep_id in task.dependencies if dep_id not in completed_tasks
            )
            score += 1000 * violated_deps  # Severe penalty for dependency violations

            # Check for overdue tasks
            finish_time = current_time + task.estimated_hours
            hypothetical_finish_ts = start_reference + finish_time * 3600
            overdue = max(
                0, (hypothetical_finish_ts - int(task.due_date.timestamp())) / 3600
            )
            score += 100 * overdue  # Penalty for being overdue

            # Priority-weighted completion time
            score += finish_time * task.priority

            # Update state
            current_time = finish_time
            completed_tasks.add(task.id)

        return score

    def _update_pheromones(
        self, solutions: List[np.ndarray], scores: List[float]
    ) -> None:
        """
        Update pheromone trails based on solution quality.

        Args:
            solutions: List of solution sequences
            scores: Corresponding evaluation scores
        """
        # Apply evaporation to all edges
        self.pheromone *= 1 - self.rho

        # Find elite solutions (top 20%)
        solution_score_pairs = list(zip(solutions, scores))
        solution_score_pairs.sort(key=lambda x: x[1])

        # Apply elitism - better solutions get more pheromone reinforcement
        num_elite = max(1, int(0.2 * len(solutions)))
        elite_pairs = solution_score_pairs[:num_elite]

        # Update pheromones for all elite solutions
        for rank, (solution, score) in enumerate(elite_pairs):
            # Higher rank (lower index) gets more weight
            weight = (num_elite - rank) / num_elite

            # Calculate pheromone deposit amount
            if score == 0:  # Avoid division by zero
                delta = 1.0
            else:
                delta = weight * (10.0 / score)  # Scale by solution quality

            # Apply pheromone to edges in the solution
            for i in range(len(solution) - 1):
                self.pheromone[solution[i], solution[i + 1]] += delta

    def schedule(self) -> List[Task]:
        """
        Schedule tasks using Ant Colony Optimization.

        Returns:
            List[Task]: Sequence of tasks in the best order found
        """
        # Initialize best solution tracking
        best_seq = None
        best_score = float("inf")

        # Main ACO iterations
        for iteration in range(self.max_iter):
            # Generate solutions from all ants
            solutions = []
            scores = []

            for _ in range(self.num_ants):
                # Each ant constructs a solution
                sol = self._construct_solution()
                sc = self._evaluate_sequence(sol)

                solutions.append(sol)
                scores.append(sc)

                # Update best solution if needed
                if sc < best_score:
                    best_score = sc
                    best_seq = sol.copy()

            # Update pheromone trails
            self._update_pheromones(solutions, scores)

            # Logging
            if (iteration + 1) % 10 == 0:
                logger.debug(
                    f"ACO iteration {iteration+1}/{self.max_iter}, best score: {best_score:.2f}"
                )

        self.best_sequence = best_seq
        self.best_score = best_score

        if best_seq is None:
            logger.warning("No valid ACO solution found; returning tasks as-is.")
            return self.tasks

        # Return the tasks in the best sequence
        return [self.tasks[i] for i in best_seq]
