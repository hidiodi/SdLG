# src/models/common/sdlg_approx_adam.py

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple, NamedTuple, Dict
from flax.linen.initializers import constant, orthogonal, zeros

from src.models.common.sdlg_hybrid import (
    GatedSlidingWindowAttention,
    GTrXLGRUGate,
    rope_rotate,
    parallel_recurrent_scan,
    SdLGHybridCarry,
)


class GatedAttentionGLALayerADAM(nn.Module):
    d_model: int
    d_k: int
    d_v: int
    r: int
    variant: str = "adam"  # "additive", "delta", "adam"
    eta: float = 0.05
    eps: float = 1e-5
    use_rope_phase: bool = True
    use_softplus_denominator: bool = True
    use_per_rank_norm: bool = True

    @nn.compact
    def __call__(self, inputs, terminations, last_memory, carry_step_count):
        T, _ = inputs.shape
        tilde_k_prev, tilde_v_prev, theta_prev = last_memory
        omegas = jnp.linspace(-jnp.pi, jnp.pi, self.r)

        def compute_step_count(carry, term):
            next_step = (carry + 1.0) * (1.0 - term)
            return next_step, next_step

        def compute_phases(carry_theta, term):
            resetted_theta = carry_theta * (1.0 - term)
            next_theta = (resetted_theta + omegas) % (2 * jnp.pi)
            return next_theta, next_theta

        if T > 1:
            _, step_counts = jax.lax.scan(compute_step_count, carry_step_count, terminations)
            _, theta = jax.lax.scan(compute_phases, theta_prev, terminations)
        else:
            step_counts = jnp.expand_dims((carry_step_count + 1.0) * (1.0 - terminations[0]), axis=0)
            resetted_theta = theta_prev * (1.0 - terminations[0])
            theta = jnp.expand_dims((resetted_theta + omegas) % (2 * jnp.pi), axis=0)

        total_gla_dims = 3 * self.d_k + 2 * self.d_v
        joint_gla = nn.Dense(
            total_gla_dims, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0), name="joint_gla_proj"
        )(inputs)

        split_idx = [self.d_k, 2 * self.d_k, 2 * self.d_k + self.d_v, 2 * self.d_k + 2 * self.d_v]
        q, k, v, beta_raw, gamma_raw = jnp.split(joint_gla, split_idx, axis=-1)

        beta = jax.nn.sigmoid(beta_raw)
        gamma = jax.nn.sigmoid(gamma_raw)

        if self.use_rope_phase:
            q_rot = rope_rotate(q[..., None, :], theta)
            k_rot = rope_rotate(k[..., None, :], theta)

            if self.variant == "additive":
                target_v = v
            else:
                # Delta / ADAM: compute memory prediction error (v - v_hat)
                v_hat = jnp.sum(jnp.einsum('t r d, r d -> t r', q_rot, tilde_k_prev)[..., None] * tilde_v_prev, axis=-2)
                import jax
                v_hat = jax.lax.stop_gradient(v_hat)
                v_hat_proj = nn.Dense(self.d_v, kernel_init=orthogonal(1.0), bias_init=constant(0.0), name="v_hat_proj")(v_hat)
                target_v = v - v_hat_proj

            write_v = jnp.cos(theta)[..., None] * (beta * target_v)[..., None, :]
            write_k = gamma[..., None, :] * k_rot
        else:
            abs_theta = step_counts[..., None] * omegas
            q_rot = jnp.cos(abs_theta)[..., None] * q[..., None, :]
            k_rot = jnp.cos(abs_theta)[..., None] * k[..., None, :]

            if self.variant == "additive":
                target_v = v
            else:
                v_hat = jnp.sum(jnp.einsum('t r d, r d -> t r', q_rot, tilde_k_prev)[..., None] * tilde_v_prev, axis=-2)
                import jax
                v_hat = jax.lax.stop_gradient(v_hat)
                v_hat_proj = nn.Dense(self.d_v, kernel_init=orthogonal(1.0), bias_init=constant(0.0), name="v_hat_proj")(v_hat)
                target_v = v - v_hat_proj

            write_v = jnp.cos(abs_theta)[..., None] * (beta * target_v)[..., None, :]
            write_k = gamma[..., None, :] * k_rot

        # Erase gating for ADAM factorized update
        if self.variant == "adam":
            d_erase_k = jax.nn.sigmoid(
                nn.Dense(self.d_k, kernel_init=orthogonal(1.0), bias_init=constant(3.0), name="erase_k")(inputs)
            )
            discount_k = d_erase_k[..., None, :] * (1.0 - gamma[..., None, :]) * (1.0 - terminations[..., None, None])
        else:
            discount_k = (1.0 - gamma[..., None, :]) * (1.0 - terminations[..., None, None])

        discount_v = (1.0 - beta[..., None, :]) * (1.0 - terminations[..., None, None])

        if T > 1:
            final_tilde_k = parallel_recurrent_scan(tilde_k_prev, write_k, discount_k)
            final_tilde_v = parallel_recurrent_scan(tilde_v_prev, write_v, discount_v)
        else:
            final_tilde_k = jnp.expand_dims(discount_k[0] * tilde_k_prev + write_k[0], axis=0)
            final_tilde_v = jnp.expand_dims(discount_v[0] * tilde_v_prev + write_v[0], axis=0)

        dot_prod = jnp.sum(q_rot * final_tilde_k, axis=-1)
        num = jnp.sum(final_tilde_v * dot_prod[..., None], axis=-2)

        if self.use_per_rank_norm:
            den_sum = jnp.sum(final_tilde_k * q_rot, axis=(-2, -1))
        else:
            den_sum = jnp.sum(jnp.mean(final_tilde_k, axis=-2) * jnp.mean(q_rot, axis=-2), axis=-1)

        if self.use_softplus_denominator:
            den = jax.nn.softplus(den_sum) + self.eps
        else:
            den = jnp.maximum(den_sum, self.eps)

        attn_out = num / den[..., None]

        step_weight = jnp.clip((step_counts - 32) / 32, 0.0, 1.0)
        intrinsic_bonus = step_weight * (self.eta / jax.lax.stop_gradient(den))

        attn_out = nn.Dense(
            self.d_model, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0), name="out_proj"
        )(attn_out)
        g = jax.nn.sigmoid(
            nn.Dense(self.d_model, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0), name="g_proj")(inputs)
        )
        out = g * attn_out

        new_tilde_k = final_tilde_k[-1]
        new_tilde_v = final_tilde_v[-1]
        new_theta = theta[-1]
        new_step_count = step_counts[-1]

        return out, intrinsic_bonus, (new_tilde_k, new_tilde_v, new_theta), new_step_count


class SdLGApproxADAMBlock(nn.Module):
    d_model: int
    d_k: int
    d_v: int
    d_ffc: int
    r: int
    window_size: int
    layer_idx: int
    variant: str = "adam"
    eta: float = 0.05
    gru_bias: float = 2.0

    @nn.compact
    def __call__(self, inputs, terminations, carry, deterministic=True):
        T, _ = inputs.shape
        norm_inputs = nn.LayerNorm(name='ln1')(inputs)
        intrinsic_bonus = jnp.zeros_like(inputs[..., 0])

        if self.layer_idx % 2 == 0:
            attn_out, new_swa_k, new_swa_v = GatedSlidingWindowAttention(
                d_model=self.d_model, d_k=self.d_k, d_v=self.d_v, window_size=self.window_size, name="swa"
            )(norm_inputs, terminations, carry.swa_k, carry.swa_v, carry.step_count, deterministic=deterministic)

            if T > 1:
                def compute_step_count(carry_sc, term):
                    next_step = (carry_sc + 1.0) * (1.0 - term)
                    return next_step, next_step
                _, step_counts = jax.lax.scan(compute_step_count, carry.step_count, terminations)
                new_step_count = step_counts[-1]
            else:
                new_step_count = (carry.step_count + 1.0) * (1.0 - terminations[0])

            new_carry = SdLGHybridCarry(
                tilde_k=carry.tilde_k, tilde_v=carry.tilde_v, theta=carry.theta,
                swa_k=new_swa_k, swa_v=new_swa_v, step_count=new_step_count
            )
        else:
            attn_out, intrinsic_bonus, (new_tilde_k, new_tilde_v, new_theta), new_step_count = GatedAttentionGLALayerADAM(
                d_model=self.d_model, d_k=self.d_k, d_v=self.d_v, r=self.r, variant=self.variant, eta=self.eta, name="gla"
            )(norm_inputs, terminations, (carry.tilde_k, carry.tilde_v, carry.theta), carry.step_count)

            new_carry = SdLGHybridCarry(
                tilde_k=new_tilde_k, tilde_v=new_tilde_v, theta=new_theta,
                swa_k=carry.swa_k, swa_v=carry.swa_v, step_count=new_step_count
            )

        gating_out = GTrXLGRUGate(self.d_model, self.gru_bias, name='gru1')(inputs, attn_out)
        norm_gating_out = nn.LayerNorm(name='ln2')(gating_out)
        ffn_out = nn.Sequential([
            nn.Dense(self.d_ffc, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0), name="dense1"),
            nn.gelu,
            nn.Dense(self.d_model, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0), name="dense2")
        ], name='ffn')(norm_gating_out)

        out = GTrXLGRUGate(self.d_model, self.gru_bias, name='gru2')(gating_out, ffn_out)
        return out, intrinsic_bonus, new_carry


class UnbatchedSdLGApproxADAM(nn.Module):
    n_layers: int
    d_model: int
    d_k: int
    d_v: int
    d_ffc: int
    r: int
    window_size: int
    variant: str = "adam"
    use_softplus_denominator: bool = True

    @nn.compact
    def __call__(self, inputs, terminations, carry, deterministic=True):
        x = nn.Dense(self.d_model)(inputs)
        x = nn.relu(x)

        new_carry = []
        for l in range(self.n_layers):
            x, _, next_carry_layer = SdLGApproxADAMBlock(
                d_model=self.d_model, d_k=self.d_k, d_v=self.d_v, d_ffc=self.d_ffc,
                r=self.r, window_size=self.window_size, layer_idx=l, variant=self.variant,
                gru_bias=2.0, name=f"block_{l}"
            )(x, terminations, carry[l], deterministic)
            new_carry.append(next_carry_layer)

        return x, tuple(new_carry)
