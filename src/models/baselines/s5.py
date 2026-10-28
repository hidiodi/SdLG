"""
S5 (Simplified State Space Sequence Model) Baseline
Implements real-valued block-diagonal approximations for JAX/XLA stability.
Adapted for standard discrete step recurrent rollout.
"""
import jax
import jax.numpy as jnp
import flax.linen as nn

class S5_Module(nn.Module):
    config: dict

    @nn.compact
    def __call__(self, x, carry):
        hidden_dim = self.config.get("RNN", {}).get("HIDDEN_DIM", 256)

        # Simple continuous-time approximated recurrence
        # h_t = A * h_{t-1} + B * x_t
        # y_t = C * h_t + D * x_t

        A = self.param('A', nn.initializers.normal(stddev=0.1), (hidden_dim,))
        B = self.param('B', nn.initializers.normal(stddev=0.1), (x.shape[-1], hidden_dim))
        C = self.param('C', nn.initializers.normal(stddev=0.1), (hidden_dim, hidden_dim))
        D = self.param('D', nn.initializers.normal(stddev=0.1), (x.shape[-1], hidden_dim))

        # Enforce stability via sigmoid on A
        A_stable = jax.nn.sigmoid(A)

        new_carry = A_stable * carry + jnp.dot(x, B)
        out = jnp.dot(new_carry, C) + jnp.dot(x, D)

        # Add non-linearity as standard in RL wrappers
        out = jax.nn.gelu(out)

        return new_carry, out

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        return jnp.zeros((batch_size, hidden_size))
