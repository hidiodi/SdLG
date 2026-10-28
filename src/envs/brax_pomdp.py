"""
Brax POMDP Environment - Hopper (Velocity Masked)
Provides a continuous control kinematic physics environment where
momentum (velocity) is explicitly masked out to test recurrent memory capabilities.
No frame-stacking is applied.
"""

import jax
import jax.numpy as jnp
from brax import envs

class HopperPOMDP:
    """
    A wrapper around Brax's Hopper that creates a POMDP by masking out
    the x and z velocity, as well as angular velocities, forcing the agent
    to infer momentum over time.
    """
    def __init__(self):
        self.env = envs.get_environment("hopper")

        # Hopper observation size is typically 11.
        # Indices:
        # 0: z-coordinate of torso (height)
        # 1: angle of torso
        # 2: angle of thigh joint
        # 3: angle of leg joint
        # 4: angle of foot joint
        # 5: velocity of x-coordinate of torso (MASKED)
        # 6: velocity of z-coordinate of torso (MASKED)
        # 7: angular velocity of torso (MASKED)
        # 8: angular velocity of thigh joint (MASKED)
        # 9: angular velocity of leg joint (MASKED)
        # 10: angular velocity of foot joint (MASKED)
        self.mask_indices = jnp.array([5, 6, 7, 8, 9, 10])

    def _mask_obs(self, obs):
        """Sets the velocity indices to zero."""
        return obs.at[self.mask_indices].set(0.0)

    @property
    def observation_size(self):
        return self.env.observation_size

    @property
    def action_size(self):
        return self.env.action_size

    def reset(self, rng):
        state = self.env.reset(rng)
        masked_obs = self._mask_obs(state.obs)
        return state.replace(obs=masked_obs)

    def step(self, state, action):
        next_state = self.env.step(state, action)
        masked_obs = self._mask_obs(next_state.obs)
        return next_state.replace(obs=masked_obs)

def make_brax_hopper_pomdp(config=None):
    return HopperPOMDP(), None
