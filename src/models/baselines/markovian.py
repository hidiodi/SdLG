"""
PPO Markovian (Amnesia) Baseline

A completely memory-less baseline. It uses a simple feed-forward network.
The recurrent carry is passed through unchanged and untouched, proving that
any performance comes purely from immediate observations, not temporal sequence mapping.
"""
import jax
import jax.numpy as jnp
import flax.linen as nn

class Markovian_Module(nn.Module):
    config: dict

    @nn.compact
    def __call__(self, x, carry):
        hidden_dim = self.config.get("RNN", {}).get("HIDDEN_DIM", 256)

        # Purely feed-forward
        out = nn.Dense(features=hidden_dim)(x)
        out = jax.nn.relu(out)
        out = nn.Dense(features=hidden_dim)(out)
        out = jax.nn.relu(out)

        # Return the dummy carry untouched
        return carry, out

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        return jnp.zeros((batch_size, hidden_size))
