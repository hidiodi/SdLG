"""
PPO LSTM Baseline
"""
import jax
import jax.numpy as jnp
import flax.linen as nn

class LSTM_Module(nn.Module):
    config: dict

    @nn.compact
    def __call__(self, x, carry):
        hidden_dim = self.config.get("RNN", {}).get("HIDDEN_DIM", 256)
        lstm = nn.LSTMCell(features=hidden_dim)

        # Unroll across time natively
        new_carry, out = lstm(carry, x)
        return new_carry, out

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy rng key since LSTMCell init carry doesn't strictly need random normal, zeros is standard
        return nn.LSTMCell.initialize_carry(jax.random.PRNGKey(0), (batch_size,), hidden_size)
