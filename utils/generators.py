"""
Utility functions for generating tasks and employees.
"""
import random
import math
from typing import List, Set, Dict, Optional, Tuple
from datetime import datetime, timedelta
from faker import Faker
from models import Task, Employee
from utils.logger import logger


class DataGenerator:
    """Generator for test data including tasks and employees."""

    def __init__(self, seed: int = 42, config: Optional[Dict[str, any]] = None):
        """
        Initialize the generator with a specific seed and optional configuration.

        Args:
            seed: Random seed for reproducibility
            config: Optional configuration settings
        """
        self.seed = seed
        self.fake = Faker()
        self.fake.seed_instance(seed)
        random.seed(seed)

        # Configuration for data generation
        self.config = config or {
            "skills": ["Drilling", "Milling", "Lathe", "CNC", "NDT", "Boring"],
            "employee_hours_min": 20,
            "employee_hours_max": 40,
            "employee_efficiency_min": 0.7,
            "employee_efficiency_max": 1.0,
            "task_hours_min": 5,
            "task_hours_max": 8,
            "task_priority_min": 1,
            "task_priority_max": 5,
            "task_due_date_min_days": 1,
            "task_due_date_max_days": 7,
            "dependency_ratio": 0.1,  # 10% of tasks may have dependencies
            "project_start_date": datetime(2023, 10, 1)
        }

    def generate_tasks(self, num_tasks: int) -> List[Task]:
        """
        Generate a list of tasks with realistic dependencies.

        Args:
            num_tasks: Number of tasks to generate

        Returns:
            List[Task]: Generated tasks
        """
        skills = self.config["skills"]
        tasks = []
        base_date = self.config["project_start_date"]
        task_min_due_dates = {}

        # First create one task for each skill to ensure coverage
        for i, skill in enumerate(skills):
            task_hours = self.fake.random_int(
                min=self.config["task_hours_min"],
                max=self.config["task_hours_max"]
            )
            dd = base_date + timedelta(
                days=self.fake.random_int(
                    min=self.config["task_due_date_min_days"],
                    max=self.config["task_due_date_max_days"]
                )
            )
            task_id = f"T{i + 1}"

            task = Task(
                id=task_id,
                name=f"{skill} Task {i + 1}",
                required_skills={skill},  # Single skill required
                estimated_hours=task_hours,
                due_date=dd,
                priority=self.fake.random_int(
                    min=self.config["task_priority_min"],
                    max=self.config["task_priority_max"]
                ),
                dependencies=set(),
            )

            tasks.append(task)
            task_min_due_dates[task_id] = dd
            logger.debug(f"Created task: {task}")

        # Then create tasks that might have dependencies
        num_dependencies = max(1, math.ceil(self.config["dependency_ratio"] * num_tasks))

        # Select indices of tasks that will have dependencies
        dependency_task_indices = set(
            random.sample(
                range(len(skills), num_tasks),
                min(num_dependencies, num_tasks - len(skills)),
            )
        )

        for i in range(len(skills), num_tasks):
            # Randomly select a skill for this task
            skill = random.choice(skills)
            task_id = f"T{i + 1}"

            # Generate task parameters
            task_hours = self.fake.random_int(
                min=self.config["task_hours_min"],
                max=self.config["task_hours_max"]
            )

            dependencies = set()
            min_due_date = base_date

            # If this task has dependencies
            if i in dependency_task_indices:
                possible_deps = range(1, i)  # Can only depend on previous tasks

                if possible_deps:
                    # Determine how many dependencies to create
                    num_deps = self.fake.random_int(
                        min=1,
                        max=math.ceil(self.config["dependency_ratio"] * len(possible_deps))
                    )

                    # Randomly select dependencies
                    chosen_deps = random.sample(
                        list(possible_deps),
                        min(num_deps, len(possible_deps))
                    )

                    dependencies = set(f"T{dep}" for dep in chosen_deps)

                    # Calculate minimum due date based on dependencies
                    for dep_id in dependencies:
                        dep_task = next(t for t in tasks if t.id == dep_id)

                        # Due date must be after dependency completion
                        dep_completion = task_min_due_dates[dep_id] + timedelta(
                            hours=dep_task.estimated_hours
                        )

                        if dep_completion > min_due_date:
                            min_due_date = dep_completion

            # Set due date to be at least one day after dependencies
            due_date = min_due_date + timedelta(days=1)

            task = Task(
                id=task_id,
                name=f"{skill} Task {i + 1}",
                required_skills={skill},  # Single skill required
                estimated_hours=task_hours,
                due_date=due_date,
                priority=self.fake.random_int(
                    min=self.config["task_priority_min"],
                    max=self.config["task_priority_max"]
                ),
                dependencies=dependencies,
            )

            tasks.append(task)
            task_min_due_dates[task_id] = due_date

            # Log task with dependencies
            if dependencies:
                logger.debug(f"Created task with dependencies: {task}")

        logger.info(f"Generated {len(tasks)} tasks, {num_dependencies} with dependencies.")
        return tasks

    def generate_employees(self, num_employees: int) -> List[Employee]:
        """
        Generate a list of employees, each with exactly one skill.

        Args:
            num_employees: Number of employees to generate

        Returns:
            List[Employee]: Generated employees
        """
        skills = self.config["skills"]
        employees = []

        # First create one employee for each skill to ensure coverage
        for i, skill in enumerate(skills):
            employee = Employee(
                id=f"E{i + 1}",
                name=self.fake.name(),
                skills={skill},  # Exactly one skill per employee
                weekly_available_hours=self.fake.random_int(
                    min=self.config["employee_hours_min"],
                    max=self.config["employee_hours_max"]
                ),
                efficiency_rating=round(
                    self.fake.random.uniform(
                        self.config["employee_efficiency_min"],
                        self.config["employee_efficiency_max"]
                    ),
                    2
                ),
                assigned_tasks=[],
                assigned_hours=0,
            )

            employees.append(employee)
            logger.debug(f"Created employee: {employee}")

        # Then create additional employees with random single skills
        for i in range(len(skills), num_employees):
            # Select one random skill
            skill = random.choice(skills)

            employee = Employee(
                id=f"E{i + 1}",
                name=self.fake.name(),
                skills={skill},  # Exactly one skill per employee
                weekly_available_hours=self.fake.random_int(
                    min=self.config["employee_hours_min"],
                    max=self.config["employee_hours_max"]
                ),
                efficiency_rating=round(
                    self.fake.random.uniform(
                        self.config["employee_efficiency_min"],
                        self.config["employee_efficiency_max"]
                    ),
                    2
                ),
                assigned_tasks=[],
                assigned_hours=0,
            )

            employees.append(employee)
            logger.debug(f"Created employee with skill {skill}: {employee}")

        # Verify the distribution of skills
        skill_distribution = {}
        for emp in employees:
            skill = next(iter(emp.skills))
            skill_distribution[skill] = skill_distribution.get(skill, 0) + 1

        logger.info(f"Generated {len(employees)} employees with skill distribution: {skill_distribution}")
        return employees

    def ensure_task_coverage(self, tasks: List[Task], employees: List[Employee]) -> List[Task]:
        """
        Ensure all employee skills have corresponding tasks.

        This function checks if there are employees with skills that don't have
        corresponding tasks, and creates additional tasks if needed.

        Args:
            tasks: List of existing tasks
            employees: List of employees

        Returns:
            List[Task]: Updated list of tasks with additions if needed
        """
        # Count skills required by existing tasks
        task_skill_count = {}
        for t in tasks:
            for skill in t.required_skills:
                task_skill_count[skill] = task_skill_count.get(skill, 0) + 1

        # Count skills available from employees
        employee_skill_count = {}
        for emp in employees:
            skill = next(iter(emp.skills))
            employee_skill_count[skill] = employee_skill_count.get(skill, 0) + 1

        # Create additional tasks for uncovered skills
        additional_tasks = []
        for skill, count in employee_skill_count.items():
            current_count = task_skill_count.get(skill, 0)

            # If there are no tasks requiring this skill, create one
            if current_count < 1:
                new_task = Task(
                    id=f"T_extra_{skill}",
                    name=f"Extra {skill} Task",
                    required_skills={skill},  # Single skill required
                    estimated_hours=self.fake.random_int(
                        min=self.config["task_hours_min"],
                        max=self.config["task_hours_max"]
                    ),
                    due_date=self.config["project_start_date"] + timedelta(days=7),
                    priority=3,  # Medium priority
                    dependencies=set(),
                )
                additional_tasks.append(new_task)
                logger.info(f"Added extra task {new_task.id} for skill {skill}.")

        # Add the new tasks to the existing list
        if additional_tasks:
            tasks.extend(additional_tasks)
            logger.info(f"Added {len(additional_tasks)} tasks to ensure skill coverage.")
        else:
            logger.info("All employee skills already have corresponding tasks.")

        return tasks

    def generate_scenario(
            self,
            num_tasks: int,
            num_employees: int
    ) -> Tuple[List[Task], List[Employee]]:
        """
        Generate a complete scenario with tasks and employees.

        This is a convenience method that generates both tasks and employees,
        and ensures task coverage for all employee skills.

        Args:
            num_tasks: Number of tasks to generate
            num_employees: Number of employees to generate

        Returns:
            Tuple[List[Task], List[Employee]]: Generated tasks and employees
        """
        employees = self.generate_employees(num_employees)
        tasks = self.generate_tasks(num_tasks)

        # Ensure all employee skills have corresponding tasks
        tasks = self.ensure_task_coverage(tasks, employees)

        return tasks, employees