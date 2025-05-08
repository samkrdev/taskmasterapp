# Directory: assignment/refinement.py
"""
Refinement algorithms for task assignments.
"""
import random
import math
import numpy as np
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from copy import deepcopy
from models import Task, Employee
from utils.logger import logger


@dataclass
class CompatibilityState:
    """State representation for compatibility-based refinement."""

    employee_workloads: Dict[str, int]
    task_assignments: Dict[str, Optional[str]]

    def to_tuple(self) -> Tuple:
        """Convert state to hashable tuple representation."""
        workloads_tuple = tuple(
            sorted((eid, hrs) for eid, hrs in self.employee_workloads.items())
        )
        tasks_tuple = tuple(
            sorted((tid, emp) for tid, emp in self.task_assignments.items())
        )
        return (workloads_tuple, tasks_tuple)


class RefinementRL:
    """
    Reinforcement Learning based refinement for task assignments.

    This model optimizes an initial assignment by exploring reassignment options
    to improve workload balance and overall efficiency.
    """

    def __init__(
        self,
        tasks: List[Task],
        employees: List[Employee],
        num_episodes: int = 100,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 0.2,
        epsilon_decay: float = 0.99,
        min_epsilon: float = 0.01,
        balance_weight: float = 10.0,
    ):
        """
        Initialize the refinement model.

        Args:
            tasks: List of tasks to optimize
            employees: List of employees available for assignment
            num_episodes: Number of training episodes
            alpha: Learning rate
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_decay: Rate of exploration decay
            min_epsilon: Minimum exploration rate
            balance_weight: Weight for workload balance in rewards
        """
        # Convert lists to dictionaries for efficient lookup
        self.tasks = {t.id: t for t in tasks}
        self.employees = {e.id: e for e in employees}

        # RL parameters
        self.num_episodes = num_episodes
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.balance_weight = balance_weight

        # Q-learning data structures
        self.q_table: Dict[Tuple, Dict[Tuple[str, str], float]] = {}

        # Best results tracking
        self.best_assignments: Dict[str, Optional[str]] = {}
        self.best_score = float("-inf")

    def _init_state(
        self, initial_assignments: Dict[str, Optional[str]]
    ) -> CompatibilityState:
        """
        Initialize state from initial assignments.

        Args:
            initial_assignments: Initial task assignments

        Returns:
            CompatibilityState: Initial state
        """
        employee_workloads = {eid: 0 for eid in self.employees}
        task_assignments = dict(initial_assignments)

        # Calculate initial workloads
        for t_id, emp_id in task_assignments.items():
            if emp_id is not None and emp_id in self.employees:
                task_obj = self.tasks[t_id]
                e_obj = self.employees[emp_id]
                needed_hours = math.ceil(
                    task_obj.estimated_hours / e_obj.efficiency_rating
                )
                employee_workloads[emp_id] += needed_hours

        return CompatibilityState(employee_workloads, task_assignments)

    def get_all_actions(self, state: CompatibilityState) -> List[Tuple[str, str]]:
        """
        Get all possible actions (task reassignments) from the current state.

        Args:
            state: Current state

        Returns:
            List of (task_id, employee_id) pairs representing possible reassignments
        """
        actions = []

        # For each task, consider reassigning to a different employee
        for t_id, current_emp in state.task_assignments.items():
            task_obj = self.tasks[t_id]

            # Skip if dependencies aren't met
            if task_obj.dependencies and not all(
                state.task_assignments.get(dep) is not None
                for dep in task_obj.dependencies
            ):
                continue

            # Consider each employee as a potential reassignment
            for e_id, e_obj in self.employees.items():
                # Skip if same employee (no change)
                if e_id == current_emp:
                    continue

                # Skip if employee doesn't have required skills
                if not (e_obj.skills & task_obj.required_skills):
                    continue

                # Check if employee has capacity
                needed = math.ceil(task_obj.estimated_hours / e_obj.efficiency_rating)
                if (
                    state.employee_workloads[e_id] + needed
                ) <= e_obj.weekly_available_hours:
                    actions.append((t_id, e_id))

        return actions

    def execute_action(
        self, state: CompatibilityState, action: Tuple[str, str]
    ) -> CompatibilityState:
        """
        Execute an action (task reassignment) and return the new state.

        Args:
            state: Current state
            action: (task_id, new_employee_id) reassignment

        Returns:
            CompatibilityState: New state after action
        """
        t_id, new_emp_id = action
        next_state = deepcopy(state)
        task_obj = self.tasks[t_id]
        new_emp = self.employees[new_emp_id]

        # Calculate hours needed for new assignment
        needed_new = math.ceil(task_obj.estimated_hours / new_emp.efficiency_rating)

        # Handle current assignment if any
        old_emp_id = next_state.task_assignments[t_id]
        if old_emp_id is not None and old_emp_id != new_emp_id:
            old_emp = self.employees[old_emp_id]
            needed_old = math.ceil(task_obj.estimated_hours / old_emp.efficiency_rating)
            # Reduce workload from previous employee
            next_state.employee_workloads[old_emp_id] -= needed_old

        # Add workload to new employee
        next_state.employee_workloads[new_emp_id] += needed_new

        # Update assignment
        next_state.task_assignments[t_id] = new_emp_id

        return next_state

    def calculate_reward(
        self,
        old_state: CompatibilityState,
        action: Tuple[str, str],
        new_state: CompatibilityState,
    ) -> float:
        """
        Calculate reward for a state transition.

        Args:
            old_state: Previous state
            action: Action taken
            new_state: Resulting state

        Returns:
            float: Reward value
        """
        # Extract workload information
        old_loads = list(old_state.employee_workloads.values())
        new_loads = list(new_state.employee_workloads.values())

        # Calculate metrics
        old_max = max(old_loads) if old_loads else 0
        new_max = max(new_loads) if new_loads else 0

        old_std = np.std(old_loads) if len(old_loads) > 1 else 0
        new_std = np.std(new_loads) if len(new_loads) > 1 else 0

        # Calculate reward components
        reward = 0.0

        # Reward for reducing maximum workload
        if new_max < old_max:
            reward += 10.0

        # Reward for improving workload balance
        if new_std < old_std:
            reward += self.balance_weight * (old_std - new_std)

        # Small penalty if nothing improved
        if reward == 0.0:
            reward = -1.0

        return reward

    def select_action(
        self, state_key: Tuple, actions: List[Tuple[str, str]]
    ) -> Tuple[str, str]:
        """
        Select an action using epsilon-greedy policy.

        Args:
            state_key: Current state key
            actions: List of possible actions

        Returns:
            Tuple[str, str]: Selected action
        """
        # Exploration
        if random.random() < self.epsilon:
            return random.choice(actions)

        # Exploitation
        if state_key not in self.q_table:
            self.q_table[state_key] = {a: 0.0 for a in actions}

        q_vals = self.q_table[state_key]

        # Find action with highest Q-value
        max_q = max(q_vals[a] for a in actions)
        best_actions = [a for a in actions if q_vals[a] == max_q]

        return random.choice(best_actions)  # Break ties randomly

    def train(self, initial_assignments: Dict[str, Optional[str]]) -> None:
        """
        Train the refinement model.

        Args:
            initial_assignments: Initial task assignments to refine
        """
        logger.debug(f"Starting refinement training for {self.num_episodes} episodes")

        for ep in range(self.num_episodes):
            # Initialize state from initial assignments
            current_state = self._init_state(initial_assignments)
            step = 0
            max_steps = 3 * len(self.tasks)  # Limit steps per episode

            # Run episode until termination
            while step < max_steps:
                step += 1

                # Get state representation and available actions
                s_key = current_state.to_tuple()
                actions = self.get_all_actions(current_state)

                # Terminate if no actions available
                if not actions:
                    break

                # Initialize Q-values if needed
                if s_key not in self.q_table:
                    self.q_table[s_key] = {a: 0.0 for a in actions}

                # Select and execute action
                action = self.select_action(s_key, actions)
                next_state = self.execute_action(current_state, action)

                # Calculate reward
                ns_key = next_state.to_tuple()
                reward = self.calculate_reward(current_state, action, next_state)

                # Initialize Q-values for next state if needed
                if ns_key not in self.q_table:
                    next_actions = self.get_all_actions(next_state)
                    self.q_table[ns_key] = {a: 0.0 for a in next_actions}

                # Update Q-value
                old_q = self.q_table[s_key][action]
                next_max_q = (
                    max(self.q_table[ns_key].values()) if self.q_table[ns_key] else 0.0
                )
                new_q = old_q + self.alpha * (reward + self.gamma * next_max_q - old_q)
                self.q_table[s_key][action] = new_q

                # Move to next state
                current_state = next_state

            # Evaluate final solution
            final_score = self._evaluate_solution(current_state)

            # Update best solution if improvement found
            if final_score > self.best_score:
                self.best_score = final_score
                self.best_assignments = dict(current_state.task_assignments)

            # Decay exploration rate
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

            # Log progress
            if (ep + 1) % 20 == 0:
                logger.debug(
                    f"Refinement episode {ep+1}/{self.num_episodes}, score: {final_score:.2f}, epsilon: {self.epsilon:.3f}"
                )

    def _evaluate_solution(self, state: CompatibilityState) -> float:
        """
        Evaluate the quality of a solution.

        Lower score is better for workload standard deviation and maximum load.

        Args:
            state: State to evaluate

        Returns:
            float: Evaluation score
        """
        loads = list(state.employee_workloads.values())

        if not loads:
            return 0.0

        # Higher (negative) score means better balance and lower max load
        return -(max(loads) + (np.std(loads) if len(loads) > 1 else max(loads)))

    def optimize_assignments(
        self, initial_assignments: Dict[str, Optional[str]]
    ) -> Dict[str, Optional[str]]:
        """
        Optimize task assignments using refinement.

        Args:
            initial_assignments: Initial task assignments

        Returns:
            Dict[str, Optional[str]]: Optimized task assignments
        """
        # Train the refinement model
        self.train(initial_assignments)

        # Return the best found assignments
        return self.best_assignments
