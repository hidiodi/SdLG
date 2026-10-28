# src/models/baselines/mamba.py
import math

import jax
import jax.numpy as jnp
import flax.linen as nn


class MambaCell(nn.Module):
    """
    A recurrent step of the Selective State Space Model (Mamba).
    Faithful to Gu & Dao (2023).
    """
    d_model: int
    d_state: int = 16
    expand: int = 2
    dt_rank: str = 'auto'

    def setup(self):
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank_dim = math.ceil(self.d_model / 16) if self.dt_rank == 'auto' else self.dt_rank

        # Projections
        self.in_proj = nn.Dense(self.d_inner * 2, use_bias=False)
        self.x_proj = nn.Dense(self.dt_rank_dim + self.d_state * 2, use_bias=False)
        self.dt_proj = nn.Dense(self.d_inner, use_bias=True)
        self.out_proj = nn.Dense(self.d_model, use_bias=False)

        # State Space Parameters
        # A is typically initialized to -exp(parameter) to ensure stability
        self.A_log = self.param('A_log',
                                lambda rng, shape: jnp.log(
                                    jnp.tile(jnp.arange(1, self.d_state + 1, dtype=jnp.float32), (self.d_inner, 1))),
                                (self.d_inner, self.d_state))
        self.D = self.param('D', nn.initializers.ones, (self.d_inner,))

    def __call__(self, x, carry):
        # x shape: (Batch, d_model)
        # carry (h) shape: (Batch, d_inner, d_state)
        h = carry

        # 1. Project input
        xz = self.in_proj(x)
        x_inner, z = jnp.split(xz, 2, axis=-1)
        x_inner = jax.nn.silu(x_inner)

        # 2. Data-dependent parameters (Selective mechanism)
        x_proj_out = self.x_proj(x_inner)
        dt, B, C = jnp.split(x_proj_out, [self.dt_rank_dim, self.dt_rank_dim + self.d_state], axis=-1)

        dt = jax.nn.softplus(self.dt_proj(dt))  # (Batch, d_inner)

        # 3. Discretization (Zero-order hold approximation)
        A = -jnp.exp(self.A_log)  # (d_inner, d_state)

        # Broadcast dt for matrix multiplications
        dt_expanded = jnp.expand_dims(dt, axis=-1)  # (Batch, d_inner, 1)

        # \bar{A} = exp(\Delta * A). Using a linear approximation for stability: 1 + \Delta * A
        deltaA = dt_expanded * A  # (Batch, d_inner, d_state)

        # For Mamba, typically exp(deltaA) is used, but 1 + delta A + ... is a valid fast approx.
        # We will use exact exp for fidelity to the paper.
        bar_A = jnp.exp(deltaA)

        # \bar{B} = \Delta * B
        B_expanded = jnp.expand_dims(B, axis=1)  # (Batch, 1, d_state)
        bar_B = dt_expanded * B_expanded  # (Batch, d_inner, d_state)

        # 4. State Update
        x_expanded = jnp.expand_dims(x_inner, axis=-1)  # (Batch, d_inner, 1)
        new_h = bar_A * h + bar_B * x_expanded  # (Batch, d_inner, d_state)

        # 5. Output
        C_expanded = jnp.expand_dims(C, axis=1)  # (Batch, 1, d_state)
        y = jnp.sum(new_h * C_expanded, axis=-1)  # (Batch, d_inner)

        # Add D skip connection and multiply by z gate
        y = y + x_inner * self.D
        y = y * jax.nn.silu(z)

        out = self.out_proj(y)

        return new_h, out

    @staticmethod
    def initialize_carry(batch_size, d_model, d_state=16, expand=2):
        d_inner = int(expand * d_model)
        return jnp.zeros((batch_size, d_inner, d_state), dtype=jnp.float32)