# src/envs/tmaze/jax_tmaze.py

import jax
import jax.numpy as jnp
from flax import struct
from typing import Tuple, Any
from gymnax.environments import environment, spaces


@struct.dataclass
class TEnvState:
    step_count: jnp.ndarray  # ()
    cue_direction: jnp.ndarray  # ()
    corridor_position: jnp.ndarray  # ()


@struct.dataclass
class TEnvParams:
    corridor_length: int = 64
    max_steps: int = 200

class PassiveTMaze(environment.Environment):
    """
    A compile-safe, JAX-pure Passive T-Maze.
    Agent starts at step 0, receives a cue [cue_direction, 0, 0] (0 or 1).
    Steps 1 to corridor_length-1: Walks down identical corridors with observations [0, 1, 0].
    Step corridor_length: Reaches decision junction [0, 0, 1].
    Must output the correct action matching the initial cue to receive reward 1.0.
    """

    def __init__(self, corridor_length: int = 64):
        super().__init__()
        self.corridor_length = corridor_length

    @property
    def default_params(self) -> TEnvParams:
        return TEnvParams(corridor_length=self.corridor_length)

    def step_env(self, key: jax.Array, state: TEnvState, action: int, params: TEnvParams) -> Tuple[
        jnp.ndarray, TEnvState, float, bool, dict]:
        # Check transition conditions
        is_decision_point = (state.corridor_position == params.corridor_length)

        # Compute rewards only at the decision point
        correct_action = (action == state.cue_direction)
        reward = jnp.where(is_decision_point, jnp.where(correct_action, 1.0, 0.0), 0.0)

        # Advance state
        next_position = state.corridor_position + 1
        done = (next_position > params.corridor_length) | (state.step_count >= params.max_steps)

        next_state = TEnvState(
            step_count=state.step_count + 1,
            cue_direction=state.cue_direction,
            corridor_position=next_position
        )

        obs = self.get_obs(next_state, params)
        return obs, next_state, reward, done, {}

    def reset_env(self, key: jax.Array, params: TEnvParams) -> Tuple[jnp.ndarray, TEnvState]:
        rng, _rng = jax.random.split(key)
        cue = jax.random.randint(_rng, (), minval=0, maxval=2)

        state = TEnvState(
            step_count=jnp.array(0, dtype=jnp.int32),
            cue_direction=cue,
            corridor_position=jnp.array(0, dtype=jnp.int32)
        )
        return self.get_obs(state, params), state

    def get_obs(self, state: TEnvState, params: TEnvParams) -> jnp.ndarray:
        """
        Symbolic observation mapping:
        - At step 0: [cue_direction (0 or 1), 0.0, 0.0]
        - Inside corridor: [0.0, 1.0, 0.0]
        - At decision point: [0.0, 0.0, 1.0]
        """
        is_start = (state.corridor_position == 0)
        is_junction = (state.corridor_position == params.corridor_length)
        is_corridor = (~is_start) & (~is_junction)

        obs = jnp.zeros((3,))
        obs = jnp.where(is_start, obs.at[0].set(state.cue_direction.astype(jnp.float32)), obs)
        obs = jnp.where(is_corridor, obs.at[1].set(1.0), obs)
        obs = jnp.where(is_junction, obs.at[2].set(1.0), obs)

        return obs

    @property
    def name(self) -> str:
        return "PassiveTMaze-v1"

    @property
    def num_actions(self) -> int:
        return 2

    def action_space(self, params: TEnvParams) -> spaces.Discrete:
        return spaces.Discrete(2)

    def observation_space(self, params: TEnvParams) -> spaces.Box:
        return spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=jnp.float32)