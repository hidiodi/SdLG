# src/envs/builder.py

import jax
import gymnax
from src.envs.wrappers import FlattenObservationWrapper, LogWrapper, AutoResetEnvWrapper


def build_purejaxrl_env(config_env, config_train):
    env_name = config_env.env_name

    if env_name == "craftax" or "Craftax" in env_name:
        from craftax.craftax_env import make_craftax_env_from_name
        env = make_craftax_env_from_name(env_name, auto_reset=not config_env.optimistic_resets)
        env_params = env.default_params
        env = LogWrapper(env)

    elif "tmaze" in env_name.lower():
        from src.envs.tmaze.jax_tmaze import PassiveTMaze
        env = PassiveTMaze(corridor_length=config_env.num_steps)
        env_params = env.default_params
        env = AutoResetEnvWrapper(env)
        env = LogWrapper(env)

    elif "MysteryPath" in env_name:
        from src.envs.tmaze.jax_mystery_path import MysteryPath
        is_grid = "Grid" in env_name
        env = MysteryPath(is_grid=is_grid)
        env_params = env.default_params
        env = AutoResetEnvWrapper(env)
        env = LogWrapper(env)

    elif "popgym" in env_name.lower() or env_name in ["PositionOnlyCartPoleEasy", "StatelessCartPole", "RepeatPreviousEasy", "RepeatPreviousMedium", "AutoencodeEasy"]:
        from src.envs.popgym_wrapper import make_popgym_env
        env, env_params = make_popgym_env(env_name, num_envs=config_env.num_envs)

    elif "bsuite" in env_name.lower():
        env, env_params = gymnax.make(env_name)
        env = FlattenObservationWrapper(env)
        env = AutoResetEnvWrapper(env)
        env = LogWrapper(env)

    else:
        env, env_params = gymnax.make(env_name)
        env = FlattenObservationWrapper(env)
        env = AutoResetEnvWrapper(env)
        env = LogWrapper(env)

    return env, env_params