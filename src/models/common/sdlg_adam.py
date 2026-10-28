# src/models/common/sdlg_adam.py

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple, NamedTuple, Dict
from flax.linen.initializers import constant, orthogonal, zeros

from src.models.common.sdlg_hybrid import GatedSlidingWindowAttention, GTrXLGRUGate
from src.models.common.exact_sdlg import ExactSdLGCarry
from src.models.common.adam_gla import ADAMMultiScaleGLA


class SdLGADAMBlock(nn.Module):
    d_model: int
    n_heads: int
    d_k: int
    d_v: int
    d_ffc: int
    window_size: int
    variant: str = "adam"  # "additive", "delta", "adam"
    gru_bias: float = 2.0

    @nn.compact
    def __call__(self, inputs, terminations, carry: ExactSdLGCarry, deterministic=True):
        T, _ = inputs.shape

        # Step 1: Local Sliding Window Attention (SWA)
        norm_inputs_swa = nn.LayerNorm()(inputs)
        swa_out, new_swa_k, new_swa_v = GatedSlidingWindowAttention(
            d_model=self.d_model, d_k=self.d_k, d_v=self.d_v, window_size=self.window_size, name="swa"
        )(norm_inputs_swa, terminations, carry.swa_k, carry.swa_v, carry.step_count, deterministic=deterministic)

        x_swa = GTrXLGRUGate(self.d_model, self.gru_bias, name="gru_swa")(inputs, swa_out)

        # Step count tracking
        if T > 1:
            def compute_step_count(carry_sc, term):
                next_step = (carry_sc + 1.0) * (1.0 - term)
                return next_step, next_step
            _, step_counts = jax.lax.scan(compute_step_count, carry.step_count, terminations)
            new_step_count = step_counts[-1]
        else:
            new_step_count = (carry.step_count + 1.0) * (1.0 - terminations[0])

        # Step 2: Global Linear Memory (Additive vs Delta vs ADAM update)
        norm_inputs_gla = nn.LayerNorm()(x_swa)
        gla_out, (new_M, new_z), telemetry = ADAMMultiScaleGLA(
            d_model=self.d_model, n_heads=self.n_heads, d_k=self.d_k, d_v=self.d_v,
            variant=self.variant, name="global_linear_memory"
        )(norm_inputs_gla, terminations, (carry.M, carry.z))

        x_gla = GTrXLGRUGate(self.d_model, self.gru_bias, name="gru_gla")(x_swa, gla_out)

        # Step 3: Feed-Forward Network
        norm_inputs_ffn = nn.LayerNorm()(x_gla)
        ffn_out = nn.Sequential([
            nn.Dense(self.d_ffc, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)),
            nn.gelu,
            nn.Dense(self.d_model, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))
        ])(norm_inputs_ffn)

        out = GTrXLGRUGate(self.d_model, self.gru_bias, name="gru_ffn")(x_gla, ffn_out)

        new_carry = ExactSdLGCarry(M=new_M, z=new_z, swa_k=new_swa_k, swa_v=new_swa_v, step_count=new_step_count)
        return out, new_carry, telemetry


class UnbatchedSdLGADAM(nn.Module):
    n_layers: int
    d_model: int
    n_heads: int
    d_k: int
    d_v: int
    d_ffc: int
    window_size: int
    variant: str = "adam"

    @nn.compact
    def __call__(self, inputs, terminations, carry, deterministic=True):
        x = nn.Dense(self.d_model)(inputs)
        x = nn.relu(x)

        new_carry = []
        all_telemetry = {}

        for l in range(self.n_layers):
            x, next_carry_layer, telemetry = SdLGADAMBlock(
                d_model=self.d_model, n_heads=self.n_heads, d_k=self.d_k, d_v=self.d_v,
                d_ffc=self.d_ffc, window_size=self.window_size, variant=self.variant, name=f"block_{l}"
            )(x, terminations, carry[l], deterministic)

            new_carry.append(next_carry_layer)
            if l == self.n_layers - 1:
                all_telemetry = telemetry

        return x, tuple(new_carry), all_telemetry
