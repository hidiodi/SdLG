# src/envs/popgym_wrapper.py

import jax
import jax.numpy as jnp
import numpy as np
import gymnasium as gym
from gymnax.environments import spaces
import popgym
from popgym.core.env import POPGymEnv as POPGymBaseEnv
from popgym.envs.position_only_cartpole import PositionOnlyCartPoleEasy
from popgym.envs.repeat_previous import RepeatPreviousEasy, RepeatPreviousMedium
from popgym.envs.autoencode import AutoencodeEasy

# Dictionary mapping clean environment keys to POPGym registered IDs
POPGYM_NAME_MAP = {
    # Continuous Tracking & Memory
    "PositionOnlyCartPoleEasy": "popgym-PositionOnlyCartPoleEasy-v0",
    "StatelessCartPoleEasy": "popgym-PositionOnlyCartPoleEasy-v0",
    "NoisyPositionOnlyCartPoleEasy": "popgym-NoisyPositionOnlyCartPoleEasy-v0",
    "NoisyStatelessCartPoleEasy": "popgym-NoisyPositionOnlyCartPoleEasy-v0",
    "VelocityOnlyCartpoleEasy": "popgym-VelocityOnlyCartpoleEasy-v0",
    "VelocityOnlyCartPoleEasy": "popgym-VelocityOnlyCartpoleEasy-v0",
    "PositionOnlyPendulumEasy": "popgym-PositionOnlyPendulumEasy-v0",
    "StatelessPendulumEasy": "popgym-PositionOnlyPendulumEasy-v0",
    "NoisyPositionOnlyPendulumEasy": "popgym-NoisyPositionOnlyPendulumEasy-v0",
    "NoisyStatelessPendulumEasy": "popgym-NoisyPositionOnlyPendulumEasy-v0",
    
    # Discrete Sequence Retrieval & Memory
    "RepeatPreviousEasy": "popgym-RepeatPreviousEasy-v0",
    "RepeatPreviousMedium": "popgym-RepeatPreviousMedium-v0",
    "RepeatFirstEasy": "popgym-RepeatFirstEasy-v0",
    "RepeatFirstMedium": "popgym-RepeatFirstMedium-v0",
    "AutoencodeEasy": "popgym-AutoencodeEasy-v0",
    
    # Grid & Game Memory
    "ConcentrationEasy": "popgym-ConcentrationEasy-v0",
    "ConcentrationMedium": "popgym-ConcentrationMedium-v0",
    "MinesweeperEasy": "popgym-MineSweeperEasy-v0",
    "MineSweeperEasy": "popgym-MineSweeperEasy-v0",
    "MinesweeperMedium": "popgym-MineSweeperMedium-v0",
    "MineSweeperMedium": "popgym-MineSweeperMedium-v0",
    "BattleshipEasy": "popgym-BattleshipEasy-v0",
    "CountRecallEasy": "popgym-CountRecallEasy-v0",
    "CountRecallMedium": "popgym-CountRecallMedium-v0",
    "HigherLowerEasy": "popgym-HigherLowerEasy-v0",
    "HigherLowerMedium": "popgym-HigherLowerMedium-v0",
    "MultiarmedBanditEasy": "popgym-MultiarmedBanditEasy-v0",
    "LabyrinthEscapeEasy": "popgym-LabyrinthEscapeEasy-v0",
    "LabyrinthEscapeMedium": "popgym-LabyrinthEscapeMedium-v0",
    "LabyrinthExploreEasy": "popgym-LabyrinthExploreEasy-v0",
    "LabyrinthExploreMedium": "popgym-LabyrinthExploreMedium-v0",
}

ENV_MAP = {
    "PositionOnlyCartPoleEasy": PositionOnlyCartPoleEasy,
    "StatelessCartPole": PositionOnlyCartPoleEasy,
    "RepeatPreviousEasy": lambda: RepeatPreviousEasy(num_decks=2),
    "RepeatPreviousMedium": RepeatPreviousMedium,
    "AutoencodeEasy": lambda: AutoencodeEasy(num_decks=1),
}


def preprocess_obs(obs, space):
    if isinstance(space, gym.spaces.MultiDiscrete) or hasattr(space, "nvec"):
        nvec = getattr(space, "nvec", None)
        if nvec is not None:
            one_hots = [np.eye(int(n), dtype=np.float32)[int(val)] for val, n in zip(obs, nvec)]
            return np.concatenate(one_hots, axis=-1)
    if isinstance(obs, (int, np.integer)):
        n = space.n if hasattr(space, 'n') else 4
        arr = np.zeros((n,), dtype=np.float32)
        arr[obs] = 1.0
        return arr
    elif isinstance(obs, tuple):
        encoded = []
        for o, s in zip(obs, space.spaces if hasattr(space, 'spaces') else [None] * len(obs)):
            if isinstance(o, (int, np.integer)):
                n = s.n if (s and hasattr(s, 'n')) else 4
                a = np.zeros((n,), dtype=np.float32)
                a[o] = 1.0
                encoded.append(a)
            else:
                encoded.append(np.asarray(o, dtype=np.float32).flatten())
        return np.concatenate(encoded, axis=-1)
    else:
        return np.asarray(obs, dtype=np.float32).flatten()


def get_obs_dim(env):
    sample_obs, _ = env.reset()
    vec = preprocess_obs(sample_obs, env.observation_space)
    return vec.shape[0]


from concurrent.futures import ThreadPoolExecutor

class POPGymEnv:
    """
    Single-environment POPGym Gymnax-compatible wrapper.
    Maintains a pool of Python env instances indexed by env_id to support
    batched execution via BatchEnvWrapper without state corruption.
    Uses ThreadPoolExecutor to parallelize Python environment stepping across CPU cores.
    """

    def __init__(self, env_name: str, max_envs: int = 2048, max_workers: int = 32):
        self.env_name = env_name
        gym_id = POPGYM_NAME_MAP.get(env_name, env_name)

        def create_env():
            if env_name in ENV_MAP:
                return ENV_MAP[env_name]()
            try:
                return gym.make(gym_id).unwrapped
            except Exception:
                pass
            if hasattr(popgym.envs, env_name):
                return getattr(popgym.envs, env_name)()
            if "CartPoleHard" in gym_id or "CartPoleHard" in env_name:
                return gym.make("popgym-PositionOnlyCartPoleHard-v0").unwrapped
            raise KeyError(f"Could not resolve POPGym environment: '{env_name}' (gym_id: '{gym_id}')")

        self.envs = [create_env() for _ in range(max_envs)]
        for env in self.envs:
            env.reset()

        sample_env = self.envs[0]

        if hasattr(sample_env.action_space, 'n'):
            self.action_dim = sample_env.action_space.n
            self.is_multidiscrete = False
            self.is_box = False
        elif hasattr(sample_env.action_space, 'nvec'):
            self.action_nvec = sample_env.action_space.nvec
            self.action_dim = int(np.prod(self.action_nvec))
            self.is_multidiscrete = True
            self.is_box = False
        elif isinstance(sample_env.action_space, gym.spaces.Box):
            self.num_bins = 5
            self.action_dim = self.num_bins ** sample_env.action_space.shape[0]
            self.is_multidiscrete = False
            self.is_box = True
            self.box_low = sample_env.action_space.low
            self.box_high = sample_env.action_space.high
        else:
            self.action_dim = sample_env.action_space.shape[0]
            self.is_multidiscrete = False
            self.is_box = False

        self.obs_requires_prev_action = getattr(sample_env, "obs_requires_prev_action", False)
        base_obs_dim = get_obs_dim(sample_env)
        if self.obs_requires_prev_action:
            self.obs_dim = base_obs_dim + self.action_dim
        else:
            self.obs_dim = base_obs_dim

        self.default_params = None
        self._max_episode_length = getattr(sample_env, "max_episode_length", 200)
        self._next_reset_idx = 0
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    @property
    def max_episode_length(self):
        return self._max_episode_length

    @max_episode_length.setter
    def max_episode_length(self, val):
        self._max_episode_length = val
        for e in self.envs:
            if hasattr(e, "max_episode_length"):
                e.max_episode_length = val

    def observation_space(self, params=None):
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

    def action_space(self, params=None):
        return spaces.Discrete(self.action_dim)

    def reset(self, rng, params=None):
        def _py_reset(key):
            env_idx = self._next_reset_idx % len(self.envs)
            self._next_reset_idx += 1
            key_arr = np.asarray(key).flatten()
            seed_val = int(abs(hash(tuple(key_arr)))) % (2 ** 31 - 1)
            raw_obs, _ = self.envs[env_idx].reset(seed=seed_val)
            vec_obs = preprocess_obs(raw_obs, self.envs[env_idx].observation_space)
            if self.obs_requires_prev_action:
                prev_act = np.zeros((self.action_dim,), dtype=np.float32)
                vec_obs = np.concatenate([vec_obs, prev_act], axis=-1)
            return vec_obs, np.int32(env_idx)

        res_shape = (
            jax.ShapeDtypeStruct((self.obs_dim,), jnp.float32),
            jax.ShapeDtypeStruct((), jnp.int32),
        )
        obs, env_state = jax.pure_callback(_py_reset, res_shape, rng, vmap_method="sequential")
        return obs, env_state

    def step(self, rng, state, action, params=None):
        def _py_batch_step(acts, env_indices):
            acts_arr = np.asarray(acts).ravel()
            indices_arr = np.asarray(env_indices).ravel()
            n = len(acts_arr)

            obs_b = np.empty((n, self.obs_dim), dtype=np.float32)
            rew_b = np.empty((n,), dtype=np.float32)
            done_b = np.empty((n,), dtype=bool)

            def _step_idx(k):
                i = int(indices_arr[k]) % len(self.envs)
                a = int(acts_arr[k])
                if self.is_multidiscrete:
                    a_env = np.unravel_index(a, self.action_nvec)
                elif self.is_box:
                    frac = a / max(1, self.action_dim - 1)
                    a_env = self.box_low + frac * (self.box_high - self.box_low)
                else:
                    a_env = a
                raw_obs, r, term, trunc, _ = self.envs[i].step(a_env)
                d = term or trunc
                if d:
                    raw_obs, _ = self.envs[i].reset()
                vec_obs = preprocess_obs(raw_obs, self.envs[i].observation_space)
                if self.obs_requires_prev_action:
                    prev_act = np.zeros((self.action_dim,), dtype=np.float32)
                    if a < self.action_dim:
                        prev_act[a] = 1.0
                    vec_obs = np.concatenate([vec_obs, prev_act], axis=-1)
                return k, vec_obs, float(r), bool(d)

            for k, vo, r, d in self.executor.map(_step_idx, range(n)):
                obs_b[k] = vo
                rew_b[k] = r
                done_b[k] = d

            return obs_b, rew_b, done_b

        res_shape = (
            jax.ShapeDtypeStruct((self.obs_dim,), jnp.float32),
            jax.ShapeDtypeStruct((), jnp.float32),
            jax.ShapeDtypeStruct((), jnp.bool_),
        )
        obs, rew, done = jax.pure_callback(_py_batch_step, res_shape, action, state, vmap_method="expand_dims")
        return obs, state, rew, done, {}


def make_popgym_env(env_key: str, num_envs: int = 128):
    """
    Constructs a POPGym environment instance using clean short-form keys.
    Falls back gracefully if full registered gym ID is passed directly.
    """
    env = POPGymEnv(env_key, max_envs=max(2048, num_envs * 4))
    from src.envs.wrappers import LogWrapper
    env = LogWrapper(env)
    return env, None

