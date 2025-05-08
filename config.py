# Directory: config.py
"""
Configuration management for the application.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class ACOConfig:
    """Configuration for Ant Colony Optimization."""

    alpha: float = 1.2
    beta: float = 3.0
    rho: float = 0.4
    max_iter: int = 50
    num_ants: int = 20


@dataclass
class RLConfig:
    """Configuration for Reinforcement Learning."""

    initial_lr: float = 0.1
    lr_decay: float = 0.995
    discount_factor: float = 0.95
    initial_epsilon: float = 0.4
    min_epsilon: float = 0.01
    epsilon_decay: float = 0.995
    use_ucb: bool = True
    ucb_c: float = 1.0


@dataclass
class RewardConfig:
    """Configuration for reward functions in RL."""

    feasible: float = 10.0
    infeasible: float = -15.0
    skip_base: float = -2.0
    dep_bonus: float = 3.0
    balance_coeff: float = 0.5
    utilization_coeff: float = 5.0


@dataclass
class RefinementConfig:
    """Configuration for refinement algorithms."""

    learning_rate: float = 0.1
    discount_factor: float = 0.95
    epsilon: float = 0.2
    min_epsilon: float = 0.01
    epsilon_decay: float = 0.995
    workload_weight: float = 0.4
    skills_weight: float = 0.3
    efficiency_weight: float = 0.3
    num_episodes: int = 1500


@dataclass
class AppConfig:
    """Main application configuration."""

    seed: int = 42
    aco: ACOConfig = field(default_factory=ACOConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    refinement: RefinementConfig = field(default_factory=RefinementConfig)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "AppConfig":
        """Create a configuration from a dictionary."""
        # Extract each section and create appropriate config objects
        aco_config = ACOConfig(
            **{
                k.split("_", 1)[1].lower(): v
                for k, v in config_dict.items()
                if k.startswith("ACO_")
            }
        )

        rl_config = RLConfig(
            **{
                k.split("_", 1)[1].lower(): v
                for k, v in config_dict.items()
                if k.startswith("RL_")
            }
        )

        reward_config = RewardConfig(
            feasible=config_dict.get("REWARD_FEASIBLE", 10.0),
            infeasible=config_dict.get("REWARD_INFEASIBLE", -15.0),
            skip_base=config_dict.get("REWARD_SKIP_BASE", -2.0),
            dep_bonus=config_dict.get("REWARD_DEP_BONUS", 3.0),
            balance_coeff=config_dict.get("BALANCE_COEFF", 0.5),
            utilization_coeff=config_dict.get("UTILIZATION_COEFF", 5.0),
        )

        refinement_config = RefinementConfig(
            learning_rate=config_dict.get("COMP_LEARNING_RATE", 0.1),
            discount_factor=config_dict.get("COMP_DISCOUNT_FACTOR", 0.95),
            epsilon=config_dict.get("COMP_EPSILON", 0.2),
            min_epsilon=config_dict.get("COMP_MIN_EPSILON", 0.01),
            epsilon_decay=config_dict.get("COMP_EPSILON_DECAY", 0.995),
            workload_weight=config_dict.get("COMP_WORKLOAD_WEIGHT", 0.4),
            skills_weight=config_dict.get("COMP_SKILLS_WEIGHT", 0.3),
            efficiency_weight=config_dict.get("COMP_EFFICIENCY_WEIGHT", 0.3),
            num_episodes=config_dict.get("NUM_EPISODES", 1500),
        )

        return cls(
            seed=config_dict.get("SEED", 42),
            aco=aco_config,
            rl=rl_config,
            reward=reward_config,
            refinement=refinement_config,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a flat dictionary."""
        result = {"SEED": self.seed}

        # Add ACO config
        for key, value in vars(self.aco).items():
            result[f"ACO_{key.upper()}"] = value

        # Add RL config
        for key, value in vars(self.rl).items():
            result[f"RL_{key.upper()}"] = value

        # Add Reward config
        for key, value in vars(self.reward).items():
            if key in ["feasible", "infeasible", "skip_base", "dep_bonus"]:
                result[f"REWARD_{key.upper()}"] = value
            else:
                # Special case for balance_coeff and utilization_coeff
                result[f"{key.upper()}_COEFF"] = value

        # Add Refinement config
        for key, value in vars(self.refinement).items():
            if key == "num_episodes":
                result["NUM_EPISODES"] = value
            else:
                result[f"COMP_{key.upper()}"] = value

        return result
