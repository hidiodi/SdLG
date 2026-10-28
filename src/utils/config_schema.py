import dataclasses
from typing import Any, Dict, Optional, Tuple, Union
import jax.numpy as jnp


@dataclasses.dataclass(frozen=True)
class StaticEnvConfig:
    """Static parameters that affect environment shape and JIT compilation."""
    env_name: str
    num_envs: int
    num_steps: int
    max_episode_steps: int
    optimistic_resets: bool = False

@dataclasses.dataclass(frozen=True)
class StaticModelConfig:
    """Static parameters that affect model architecture and JIT compilation."""
    rnn_type: str
    n_layers: int
    d_model: int
    n_heads: int
    d_head: int
    approx_rank: int
    window_size: int
    use_swa: bool = True

@dataclasses.dataclass
class DynamicTrainingConfig:
    """Dynamic parameters that can be changed without triggering recompilation."""
    total_timesteps: int
    learning_rate: float
    gamma: float
    gae_lambda: float
    clip_eps: float
    ent_coef: float
    vf_coef: float
    max_grad_norm: float
    update_epochs: int
    minibatch_size: int
    log_interval: int

def validate_config(config_dict: dict) -> Tuple[StaticEnvConfig, StaticModelConfig, DynamicTrainingConfig]:
    """Validates the configuration dictionary against strict schemas."""
    # This function expects a nested dict mirroring the hydra structure
    env_dict = config_dict.get("env", {})
    model_dict = config_dict.get("model", {})
    train_dict = config_dict.get("training", {})

    static_env = StaticEnvConfig(
        env_name=env_dict.get("env_name", "Craftax-Symbolic-v1"),
        num_envs=env_dict.get("num_envs", 256),
        num_steps=env_dict.get("num_steps", 64),
        max_episode_steps=env_dict.get("max_episode_steps", 10000),
        optimistic_resets=env_dict.get("optimistic_resets", False)
    )

    static_model = StaticModelConfig(
        rnn_type=model_dict.get("rnn_type", "hybrid"),
        n_layers=model_dict.get("n_layers", 4),
        d_model=model_dict.get("d_model", 256),
        n_heads=model_dict.get("n_heads", 8),
        d_head=model_dict.get("d_head", 32),
        approx_rank=model_dict.get("approx_rank", 1),
        window_size=model_dict.get("window_size", 32),
        use_swa=model_dict.get("use_swa", True)
    )

    dynamic_training = DynamicTrainingConfig(
        total_timesteps=train_dict.get("total_timesteps", 10000000),
        learning_rate=train_dict.get("learning_rate", 2e-4),
        gamma=train_dict.get("gamma", 0.99),
        gae_lambda=train_dict.get("gae_lambda", 0.95),
        clip_eps=train_dict.get("clip_eps", 0.2),
        ent_coef=train_dict.get("ent_coef", 0.01),
        vf_coef=train_dict.get("vf_coef", 0.5),
        max_grad_norm=train_dict.get("max_grad_norm", 0.5),
        update_epochs=train_dict.get("update_epochs", 4),
        minibatch_size=train_dict.get("minibatch_size", 2048),
        log_interval=train_dict.get("log_interval", 10)
    )

    return static_env, static_model, dynamic_training
