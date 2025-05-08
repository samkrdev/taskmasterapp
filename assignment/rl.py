# Directory: assignment/rl.py
"""
Reinforcement learning based assignment models.
"""
import random
import math
from datetime import datetime

import numpy as np
from typing import List, Dict, Optional, Tuple, Set, Any
from copy import deepcopy
from models import Task, Employee
from assignment.interfaces import AssignmentModel
from utils.logger import logger


class CentralizedRL(AssignmentModel):
    """
    Centralized Reinforcement Learning assignment model.

    This model uses Q-learning or SARSA to learn and optimize
    task assignments to employees.
    """

    def __init__(self, config: Dict[str, Any], algorithm: str = "q_learning"):
        """
        Initialize the RL assignment model.

        Args:
            config: Configuration dictionary
            algorithm: "q_learning" or "sarsa"
        """
        self.algorithm = algorithm

        # RL parameters
        self.lr = config["RL_INITIAL_LR"]
        self.lr_decay = config["RL_LR_DECAY"]
        self.gamma = config["RL_DISCOUNT_FACTOR"]
        self.initial_epsilon = config["RL_INITIAL_EPSILON"]
        self.min_epsilon = config["RL_MIN_EPSILON"]
        self.epsilon = self.initial_epsilon
        self.epsilon_decay = config["RL_EPSILON_DECAY"]
        self.use_ucb = config.get("RL_USE_UCB", True)
        self.ucb_c = config.get("RL_UCB_C", 1.0)

        # Reward parameters
        self.balance_coeff = config.get("BALANCE_COEFF", 0.5)
        self.utilization_coeff = config.get("UTILIZATION_COEFF", 5.0)
        self.rw_feasible = config["REWARD_FEASIBLE"]
        self.rw_infeasible = config["REWARD_INFEASIBLE"]
        self.rw_skip_base = config["REWARD_SKIP_BASE"]
        self.rw_dep_bonus = config["REWARD_DEP_BONUS"]

        # State-action representation
        self.q_table: Dict[str, List[float]] = {}
        self.state_action_counts: Dict[str, List[int]] = {}
        self.initial_q_value = 1.0

        # Task and employee data
        self.tasks: List[Task] = []
        self.employees: List[Employee] = []

        # Performance tracking
        self.training_rewards = []
        self.epsilon_history = []

    def get_state_key(
        self, task_idx: int, remaining_hours: List[int], assigned_so_far: Set[str]
    ) -> str:
        """
        Create a string key representing the current state.

        This representation captures the current task, employee availability,
        and task assignment status.

        Args:
            task_idx: Index of the current task
            remaining_hours: List of remaining hours for each employee
            assigned_so_far: Set of task IDs that have been assigned

        Returns:
            str: String key for the state
        """
        # Create a normalized representation of remaining hours
        normalized = []
        for i, hours in enumerate(remaining_hours):
            max_hours = self.employees[i].weekly_available_hours
            norm_val = hours / max_hours if max_hours > 0 else 0
            normalized.append(round(norm_val, 1))

        # Include current task information
        current_task = self.tasks[task_idx]
        priority_bin = current_task.priority

        # Include due date information
        base_date = datetime(2023, 10, 1)
        days_until_due = (current_task.due_date - base_date).days
        due_bin = days_until_due // 3

        # Include assigned tasks so far (sorted for determinism)
        assigned_str = "_".join(sorted(assigned_so_far))

        # Include normalized remaining hours
        binned_remaining = "_".join(map(str, normalized))

        # Combine all information into a single string key
        return (
            f"{task_idx}|P{priority_bin}|D{due_bin}|R{binned_remaining}|A{assigned_str}"
        )

    def choose_action(self, state_key: str) -> int:
        """
        Choose an action using epsilon-greedy policy.

        Args:
            state_key: Current state key

        Returns:
            int: Action index (0 = skip, 1...N = assign to employee)
        """
        num_actions = len(self.employees) + 1  # +1 for skip action

        # Ensure state exists in q_table
        if state_key not in self.q_table:
            self.q_table[state_key] = [self.initial_q_value] * num_actions
            self.state_action_counts[state_key] = [1] * num_actions

        # Exploration - pick random action
        if random.random() < self.epsilon:
            action = random.randint(0, num_actions - 1)
        else:
            # Exploitation with UCB or standard max Q
            if self.use_ucb:
                # Apply Upper Confidence Bound for exploration
                total = sum(self.state_action_counts[state_key])
                ucb_values = []

                for a, q in enumerate(self.q_table[state_key]):
                    count = self.state_action_counts[state_key][a]
                    # UCB formula: Q-value + c * sqrt(ln(total_visits) / visit_count)
                    bonus = self.ucb_c * math.sqrt(math.log(total) / count)
                    ucb_values.append(q + bonus)

                # Find best action with UCB value
                max_val = max(ucb_values)
                best_actions = [i for i, v in enumerate(ucb_values) if v == max_val]
                action = random.choice(best_actions)  # Break ties randomly
            else:
                # Standard greedy selection based on Q values
                qvals = self.q_table[state_key]
                max_val = max(qvals)
                best_actions = [i for i, v in enumerate(qvals) if v == max_val]
                action = random.choice(best_actions)  # Break ties randomly

        # Update visit count for this state-action
        self.state_action_counts[state_key][action] += 1

        return action

    def greedy_action(self, state_key: str) -> int:
        """
        Choose the best action based only on learned Q-values.

        Args:
            state_key: Current state key

        Returns:
            int: Action index (0 = skip, 1...N = assign to employee)
        """
        num_actions = len(self.employees) + 1

        # Ensure state exists in q_table
        if state_key not in self.q_table:
            self.q_table[state_key] = [self.initial_q_value] * num_actions
            self.state_action_counts[state_key] = [1] * num_actions

        # Choose action with highest Q-value
        qvals = self.q_table[state_key]
        max_val = max(qvals)
        best_actions = [i for i, v in enumerate(qvals) if v == max_val]

        return random.choice(best_actions)  # Break ties randomly

    def update_q(
        self, state_key: str, action: int, reward: float, next_state_key: Optional[str]
    ) -> None:
        """
        Update Q-values based on experience.

        Args:
            state_key: Current state key
            action: Action taken
            reward: Reward received
            next_state_key: Next state key (None if terminal)
        """
        num_actions = len(self.employees) + 1

        # Ensure state exists in q_table
        if state_key not in self.q_table:
            self.q_table[state_key] = [self.initial_q_value] * num_actions
            self.state_action_counts[state_key] = [1] * num_actions

        # Get current Q-value
        old_val = self.q_table[state_key][action]

        if next_state_key is not None:
            # Ensure next state exists in q_table
            if next_state_key not in self.q_table:
                self.q_table[next_state_key] = [self.initial_q_value] * num_actions
                self.state_action_counts[next_state_key] = [1] * num_actions

            # Update using Q-learning or SARSA
            if self.algorithm == "q_learning":
                # Q-learning: Q(s,a) = Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
                next_max = max(self.q_table[next_state_key])
                new_val = old_val + self.lr * (reward + self.gamma * next_max - old_val)
            else:
                # SARSA: Q(s,a) = Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]
                next_act = self.choose_action(
                    next_state_key
                )  # Different from Q-learning
                next_q = self.q_table[next_state_key][next_act]
                new_val = old_val + self.lr * (reward + self.gamma * next_q - old_val)
        else:
            # Terminal state update: Q(s,a) = Q(s,a) + α[r - Q(s,a)]
            new_val = old_val + self.lr * (reward - old_val)

        # Update Q-table
        self.q_table[state_key][action] = new_val

    def compute_skip_reward(
        self, task: Task, assigned_so_far: Set[str], current_time: int
    ) -> float:
        """
        Compute reward for skipping a task.

        Args:
            task: Task being skipped
            assigned_so_far: Set of task IDs already assigned
            current_time: Current simulation time

        Returns:
            float: Reward value
        """
        # Base penalty for skipping
        reward = self.rw_skip_base

        # Higher penalty for skipping high priority tasks
        if task.priority >= 4:
            reward -= 5.0

        # Penalty for skipping tasks with unsatisfied dependencies
        unsatisfied = len(task.dependencies - assigned_so_far)
        reward -= unsatisfied * 10.0

        # Penalty for skipping tasks close to due date
        base_ts = datetime(2023, 10, 1).timestamp()
        task_due_ts = task.due_date.timestamp()
        hours_left = (task_due_ts - base_ts) / 3600 - current_time
        if hours_left < 24:
            reward -= 5.0

        return reward

    def compute_assignment_reward(
        self,
        task: Task,
        chosen_employee: Employee,
        assigned_so_far: Set[str],
        current_time: int,
    ) -> float:
        """
        Compute reward for assigning a task to an employee.

        Args:
            task: Task being assigned
            chosen_employee: Employee chosen for assignment
            assigned_so_far: Set of task IDs already assigned
            current_time: Current simulation time

        Returns:
            float: Reward value
        """
        # Base reward proportional to task priority
        reward = self.rw_feasible * task.priority

        # Bonus for completing task dependencies
        if task.dependencies and task.dependencies.issubset(assigned_so_far):
            reward += self.rw_dep_bonus

        # Calculate expected completion time
        needed = math.ceil(task.estimated_hours / chosen_employee.efficiency_rating)
        finish_time = current_time + needed

        # Check if assignment will be overdue and adjust reward
        base_ts = datetime(2023, 10, 1).timestamp()
        task_due_ts = task.due_date.timestamp()
        expected_finish_ts = base_ts + finish_time * 3600
        hours_overdue = (expected_finish_ts - task_due_ts) / 3600

        # Bonus for early completion
        if hours_overdue < 0:
            reward += abs(hours_overdue) * 2

        return reward

    def compute_workload_balance_reward(self) -> float:
        """
        Compute reward for overall workload balance.

        Returns:
            float: Reward value (negative for imbalance)
        """
        assigned_hours = [emp.assigned_hours for emp in self.employees]

        if len(assigned_hours) > 1:
            # Use standard deviation as a measure of imbalance
            workload_std = np.std(assigned_hours)
            return -self.balance_coeff * workload_std

        return 0.0

    def can_assign(
        self, employee: Employee, task: Task, assigned_so_far: Set[str]
    ) -> bool:
        """
        Check if a task can be assigned to an employee.

        Args:
            employee: Target employee
            task: Task to assign
            assigned_so_far: Set of task IDs already assigned

        Returns:
            bool: True if assignment is feasible
        """
        # Check dependencies
        if not task.dependencies.issubset(assigned_so_far):
            return False

        # Check skills
        if not employee.skills.intersection(task.required_skills):
            return False

        # Check available hours
        needed = math.ceil(task.estimated_hours / employee.efficiency_rating)
        return (employee.weekly_available_hours - employee.assigned_hours) >= needed

    def run_episode(self, task_sequence: List[Task], training: bool = True) -> float:
        """
        Run one episode of RL training or evaluation.

        Args:
            task_sequence: Sequence of tasks to process
            training: Whether to update Q-values

        Returns:
            float: Total reward for the episode
        """
        self.tasks = task_sequence

        # Reset employee assignments
        for emp in self.employees:
            emp.assigned_tasks = []
            emp.assigned_hours = 0

        total_reward = 0.0
        assigned_so_far = set()
        current_time = 0

        # Process each task in the sequence
        for i, tsk in enumerate(task_sequence):
            remaining_hrs = [
                emp.weekly_available_hours - emp.assigned_hours
                for emp in self.employees
            ]

            # Get current state
            st_key = self.get_state_key(i, remaining_hrs, assigned_so_far)

            # Choose action (skip or assign to employee)
            action = (
                self.choose_action(st_key) if training else self.greedy_action(st_key)
            )

            if action == 0:  # Skip this task
                # Compute reward and potentially update Q-value
                rew = self.compute_skip_reward(tsk, assigned_so_far, current_time)

                # Get next state if not the last task
                next_st = None
                if i < len(task_sequence) - 1:
                    next_remaining = [
                        emp.weekly_available_hours - emp.assigned_hours
                        for emp in self.employees
                    ]
                    next_st = self.get_state_key(i + 1, next_remaining, assigned_so_far)

                # Update Q-value if in training mode
                if training:
                    self.update_q(st_key, action, rew, next_st)

                total_reward += rew
            else:  # Assign to employee
                employee_idx = action - 1  # Adjust for skip action
                chosen_employee = self.employees[employee_idx]

                # Check if assignment is feasible
                if self.can_assign(chosen_employee, tsk, assigned_so_far):
                    # Compute reward for assignment
                    rew = self.compute_assignment_reward(
                        tsk, chosen_employee, assigned_so_far, current_time
                    )

                    # Update employee workload
                    needed = math.ceil(
                        tsk.estimated_hours / chosen_employee.efficiency_rating
                    )
                    chosen_employee.assigned_tasks.append(tsk.id)
                    chosen_employee.assigned_hours += needed

                    # Update tracking
                    assigned_so_far.add(tsk.id)
                    current_time += needed
                else:
                    # Penalty for infeasible assignment
                    rew = self.rw_infeasible

                # Get next state if not the last task
                next_st = None
                if i < len(task_sequence) - 1:
                    next_remaining = [
                        emp.weekly_available_hours - emp.assigned_hours
                        for emp in self.employees
                    ]
                    next_st = self.get_state_key(i + 1, next_remaining, assigned_so_far)

                # Update Q-value if in training mode
                if training:
                    self.update_q(st_key, action, rew, next_st)

                total_reward += rew

        # Add workload balance reward
        total_reward += self.compute_workload_balance_reward()

        # Add utilization reward
        num_idle_employees = sum(1 for emp in self.employees if emp.assigned_hours == 0)
        rew_utilization = (
            self.utilization_coeff
            * (len(self.employees) - num_idle_employees)
            / len(self.employees)
        )
        total_reward += rew_utilization

        return total_reward

    def train(
        self,
        task_sequence: List[Task],
        episodes: int = 300,
        aco_scheduler: Optional["ACOScheduler"] = None,
    ) -> Tuple[List[float], List[float]]:
        """
        Train the RL model.

        Args:
            task_sequence: Sequence of tasks to train on
            episodes: Number of training episodes
            aco_scheduler: Optional ACO scheduler for periodic sequence updates

        Returns:
            Tuple of reward history and epsilon history
        """
        self.tasks = task_sequence
        rewards_per_episode = []
        epsilon_history = []

        for ep in range(episodes):
            # Periodically update task sequence using ACO (if provided)
            if aco_scheduler is not None and ((ep + 1) % 100 == 0):
                new_seq = aco_scheduler.schedule()
                ep_reward = self.run_episode(new_seq, training=True)
            else:
                ep_reward = self.run_episode(task_sequence, training=True)

            # Record metrics
            rewards_per_episode.append(ep_reward)
            epsilon_history.append(self.epsilon)

            # Epsilon decay schedule
            if ep < 0.7 * episodes:
                fraction = ep / (0.7 * episodes)
                self.epsilon = self.initial_epsilon - fraction * (
                    self.initial_epsilon - self.min_epsilon
                )
            else:
                self.epsilon = self.min_epsilon

            # Learning rate decay
            self.lr = max(0.01, self.lr * self.lr_decay)

            # Log progress
            if (ep + 1) % 100 == 0:
                logger.debug(
                    f"Episode {ep+1}/{episodes}, reward: {ep_reward:.2f}, epsilon: {self.epsilon:.3f}"
                )

        # Store for later analysis
        self.training_rewards = rewards_per_episode
        self.epsilon_history = epsilon_history

        return rewards_per_episode, epsilon_history

    def assign(
        self, tasks: List[Task], employees: List[Employee]
    ) -> Dict[str, Optional[str]]:
        """
        Assign tasks to employees using the trained model.

        Args:
            tasks: List of tasks to assign
            employees: List of employees to assign to

        Returns:
            Dict mapping task IDs to employee IDs (or None if unassigned)
        """
        self.tasks = tasks
        self.employees = employees
        self.epsilon = 0.0  # No exploration during assignment

        # Reset employee assignments
        for emp in self.employees:
            emp.assigned_tasks = []
            emp.assigned_hours = 0

        assigned_dict = {}
        assigned_so_far = set()

        # Assign each task in sequence
        for i, tsk in enumerate(tasks):
            remaining_hrs = [
                emp.weekly_available_hours - emp.assigned_hours
                for emp in self.employees
            ]

            # Get current state and best action
            st_key = self.get_state_key(i, remaining_hrs, assigned_so_far)
            act = self.greedy_action(st_key)

            if act == 0:  # Skip this task
                assigned_dict[tsk.id] = None
            else:
                employee_idx = act - 1
                chosen_employee = self.employees[employee_idx]

                # Validate assignment
                if self.can_assign(chosen_employee, tsk, assigned_so_far):
                    assigned_dict[tsk.id] = chosen_employee.id

                    # Update employee workload
                    needed = math.ceil(
                        tsk.estimated_hours / chosen_employee.efficiency_rating
                    )
                    chosen_employee.assigned_tasks.append(tsk.id)
                    chosen_employee.assigned_hours += needed

                    # Track assigned tasks
                    assigned_so_far.add(tsk.id)
                else:
                    assigned_dict[tsk.id] = None

        return assigned_dict
