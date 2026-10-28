# src/envs/tmaze/jax_mystery_path.py

import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import spaces


@struct.dataclass
class EnvState:
    pos: jnp.ndarray  # float32 coordinates for smooth movement
    origin: jnp.ndarray  # int32
    target: jnp.ndarray  # int32
    path_grid: jnp.ndarray  # bool
    visited_grid: jnp.ndarray  # bool
    time: int


@struct.dataclass
class EnvParams:
    max_steps_in_episode: int = 128
    is_grid: bool = True
    reward_path_progress: float = 0.0
    agent_speed: float = 1.0


class MysteryPath:
    """
    Pure JAX implementation of the Memory Gym 'Mystery Path' environment.
    (Pleines et al. 2023)
    """

    def __init__(self, is_grid=True):
        self.is_grid = is_grid
        if is_grid:
            self.default_params = EnvParams(max_steps_in_episode=128, is_grid=True, reward_path_progress=0.0,
                                            agent_speed=1.0)
        else:
            # 512 max steps, 9 actions (diagonal), smooth movement (0.25 tiles/step), dense reward (0.1)
            self.default_params = EnvParams(max_steps_in_episode=512, is_grid=False, reward_path_progress=0.1,
                                            agent_speed=0.25)

    def action_space(self, params=None):
        return spaces.Discrete(4 if self.is_grid else 9)

    def observation_space(self, params=None):
        return spaces.Box(0.0, 1.0, (64, 64, 3), dtype=jnp.float32)

    def _get_obs(self, state):
        origin_grid = jnp.zeros((7, 7)).at[state.origin[0], state.origin[1]].set(1.0)
        target_grid = jnp.zeros((7, 7)).at[state.target[0], state.target[1]].set(1.0)

        env_img = jnp.stack([jnp.zeros((7, 7)), origin_grid, target_grid], axis=-1)
        env_img = jnp.repeat(jnp.repeat(env_img, 9, axis=0), 9, axis=1)
        env_img = jnp.pad(env_img, ((0, 1), (0, 1), (0, 0)), constant_values=0.0)

        # Smooth agent rendering reflecting exact sub-grid float coordinates
        agent_img = jnp.zeros((64, 64))
        px = jnp.clip(jnp.floor(state.pos[0] * 9).astype(jnp.int32), 0, 55)
        py = jnp.clip(jnp.floor(state.pos[1] * 9).astype(jnp.int32), 0, 55)
        agent_img = jax.lax.dynamic_update_slice(agent_img, jnp.ones((9, 9)), (px, py))

        return env_img.at[..., 0].set(agent_img)

    def reset(self, key, params=None):
        params = params if params is not None else self.default_params
        k1, k2, k3, k4 = jax.random.split(key, 4)

        start_y = jax.random.randint(k1, (), 1, 6)
        end_y = jax.random.randint(k2, (), 1, 6)

        num_y = jnp.abs(end_y - start_y)
        dir_y = jnp.sign(end_y - start_y)

        moves_x = jnp.full(6, 1)
        moves_y = jnp.where(jnp.arange(6) < num_y, 2, 0)
        all_moves = jax.random.permutation(k3, jnp.concatenate([moves_x, moves_y]))

        def step_path(carry, move):
            cx, cy = carry
            nx = cx + (move == 1)
            ny = cy + (move == 2) * dir_y
            return jnp.array([nx, ny]), jnp.array([nx, ny])

        _, path_coords = jax.lax.scan(step_path, jnp.array([0, start_y]), all_moves)

        xs = jnp.concatenate([jnp.array([0]), path_coords[:, 0]])
        ys = jnp.concatenate([jnp.array([start_y]), path_coords[:, 1]])

        # 4-Way Cardinal Rotation (Paper Spec)
        cardinal_choice = jax.random.randint(k4, (), 0, 4)

        def rot_0(): return xs, ys

        def rot_1(): return ys, 6 - xs

        def rot_2(): return 6 - xs, 6 - ys

        def rot_3(): return 6 - ys, xs

        xs, ys = jax.lax.switch(cardinal_choice, [rot_0, rot_1, rot_2, rot_3])

        path_grid = jnp.zeros((7, 7), dtype=jnp.bool_)
        path_grid = path_grid.at[xs, ys].set(True)

        origin = jnp.array([xs[0], ys[0]], dtype=jnp.int32)
        target = jnp.array([xs[-1], ys[-1]], dtype=jnp.int32)

        visited_grid = jnp.zeros((7, 7), dtype=jnp.bool_)
        visited_grid = visited_grid.at[origin[0], origin[1]].set(True)

        state = EnvState(
            pos=origin.astype(jnp.float32),
            origin=origin,
            target=target,
            path_grid=path_grid,
            visited_grid=visited_grid,
            time=0
        )
        return self._get_obs(state), state

    def step(self, key, state, action, params=None):
        params = params if params is not None else self.default_params

        if self.is_grid:
            moves = jnp.array([[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=jnp.float32)
            move = moves[action]
        else:
            # 9-action space mapped perfectly from MultiDiscrete
            dx = (action // 3) - 1
            dy = (action % 3) - 1
            move = jnp.array([dx, dy], dtype=jnp.float32)

        new_pos = state.pos + move * params.agent_speed
        new_pos = jnp.clip(new_pos, 0.0, 6.999)

        tile_x = jnp.floor(new_pos[0]).astype(jnp.int32)
        tile_y = jnp.floor(new_pos[1]).astype(jnp.int32)

        on_path = state.path_grid[tile_x, tile_y]
        is_target = (tile_x == state.target[0]) & (tile_y == state.target[1])
        is_unvisited = ~state.visited_grid[tile_x, tile_y]

        final_pos = jnp.where(on_path, new_pos, state.origin.astype(jnp.float32))
        final_tile_x = jnp.where(on_path, tile_x, state.origin[0])
        final_tile_y = jnp.where(on_path, tile_y, state.origin[1])

        reward = jax.lax.cond(
            on_path,
            lambda: jnp.where(is_target, 1.0, jnp.where(is_unvisited, params.reward_path_progress, 0.0)),
            lambda: 0.0
        )

        new_visited = state.visited_grid.at[final_tile_x, final_tile_y].set(True)
        done = is_target | (state.time + 1 >= params.max_steps_in_episode)

        new_state = state.replace(
            pos=final_pos,
            visited_grid=new_visited,
            time=state.time + 1
        )

        # Explicit success tracking eliminates reward-threshold guessing
        info = {"success": is_target}
        return self._get_obs(new_state), new_state, reward, done, info