"""
PPO GRU Baseline
"""
import jax
import jax.numpy as jnp
import flax.linen as nn

class GRU_Module(nn.Module):
    config: dict

    @nn.compact
    def __call__(self, x, carry):
        hidden_dim = self.config.get("RNN", {}).get("HIDDEN_DIM", 256)
        gru = nn.GRUCell(features=hidden_dim)

        new_carry, out = gru(carry, x)
        return new_carry, out

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        return nn.GRUCell.initialize_carry(jax.random.PRNGKey(0), (batch_size,), hidden_size)
