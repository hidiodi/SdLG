"""
Centralized Environment Instantiation Factory
This standardizes the instantiation of environments to guarantee baseline configurations
(like deterministic seed handling) are uniform across all algorithms.
"""

def make_env(env_id: str, config: dict):
    """
    Instantiates an environment by its identifier string (e.g., 'tmaze', 'craftax', 'xminigrid').

    Args:
        env_id: String identifier for the environment.
        config: A dictionary containing environment-specific parameters.

    Returns:
        The initialized environment object and its parameters.
    """
    if env_id.lower() == "tmaze":
        from memory_gym.envs.tmaze import TMaze
        # Dummy config translation placeholder
        env = TMaze()
        env_params = None
        return env, env_params
    elif env_id.lower() == "craftax_full":
        from src.envs.craftax import make_craftax_env
        env, env_params = make_craftax_env(config)
        return env, env_params
    elif env_id.lower() == "xminigrid":
        from src.envs.xminigrid import make_xminigrid_env
        env, env_params = make_xminigrid_env(config)
        return env, env_params
    elif env_id.lower() == "brax_hopper":
        # Placeholder for Stage 2 Brax POMDP implementation
        from src.envs.brax_pomdp import make_brax_hopper_pomdp
        env, env_params = make_brax_hopper_pomdp(config)
        return env, env_params
    else:
        raise ValueError(f"Unknown environment identifier: {env_id}")
