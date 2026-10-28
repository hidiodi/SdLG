# src/models/common/adam_gla.py

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple, NamedTuple, Dict
from flax.linen.initializers import zeros, orthogonal, constant


class ADAMMultiScaleGLA(nn.Module):
    d_model: int
    n_heads: int
    d_k: int
    d_v: int
    variant: str = "adam"
    # Variants:
    #   "additive"     — Family A baseline: scalar decay + additive outer-product write
    #   "delta"        — Family A: scalar decay + delta-rule (error correction) write
    #   "channel_decay"— Family B: per-key-channel erase gate + additive write (isolates retention)
    #   "gated"        — scalar decay + additive write + per-channel erase (legacy name)
    #   "adam"         — Family A: per-channel erase gate + delta-rule write (full ADAM)

    def setup(self):
        # Base timescales for exponential decay (per-head multi-scale)
        base_timescales = jnp.array([16.0, 64.0, 256.0, 1024.0, 4096.0, 16384.0])
        if self.n_heads > len(base_timescales):
            pad_size = self.n_heads - len(base_timescales)
            extra_scales = base_timescales[-1] * (4.0 ** jnp.arange(1, pad_size + 1))
            timescales = jnp.concatenate([base_timescales, extra_scales])
        else:
            timescales = base_timescales[:self.n_heads]

        lambda_h = jnp.log(2.0) / timescales
        self.gamma_bias = jnp.log(jnp.exp(lambda_h) - 1.0)

        # Projections
        self.q_proj = nn.Dense(self.n_heads * self.d_k, use_bias=False, name="q_proj")
        self.k_proj = nn.Dense(self.n_heads * self.d_k, use_bias=False, name="k_proj")
        self.v_proj = nn.Dense(self.n_heads * self.d_v, use_bias=False, name="v_proj")
        self.out_proj = nn.Dense(self.d_model, use_bias=False, name="out_proj")

        self.delta_lambda_proj = nn.Dense(self.n_heads, use_bias=False, kernel_init=zeros, name="gamma_gate")
        self.beta_proj = nn.Dense(self.n_heads, use_bias=False, kernel_init=zeros, name="beta_gate")

        if self.variant in ("gated", "adam", "channel_decay"):
            # Family B / ADAM: Factorized channel-wise erase gate: d_t^k in R^{n_heads x d_k}
            # Positive bias (+3.0) ensures sigmoid(3.0) ≈ 0.95 (preserving memory by default at init)
            self.erase_proj = nn.Dense(self.n_heads * self.d_k, use_bias=True, kernel_init=zeros, bias_init=constant(3.0), name="erase_gate")


    def __call__(self, inputs: jnp.ndarray, terminations: jnp.ndarray, carry: Tuple[jnp.ndarray, jnp.ndarray]) -> Tuple[
        jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray], Dict[str, jnp.ndarray]]:
        T, _ = inputs.shape
        M_prev, z_prev = carry

        # Projections
        q = self.q_proj(inputs).reshape((T, self.n_heads, self.d_k))
        k = self.k_proj(inputs).reshape((T, self.n_heads, self.d_k))
        v = self.v_proj(inputs).reshape((T, self.n_heads, self.d_v))

        delta_lambda = self.delta_lambda_proj(inputs)
        beta_raw = self.beta_proj(inputs)

        # Feature normalization
        q_rms = q / (jnp.sqrt(jnp.mean(q**2, axis=-1, keepdims=True)) + 1e-5) * 2.0
        k_rms = k / (jnp.sqrt(jnp.mean(k**2, axis=-1, keepdims=True)) + 1e-5) * 2.0

        if self.variant in ("delta", "adam"):
            # Delta-rule feedback loop requires zero-centered keys/queries to prevent exponential feedback explosion
            phi_q = q_rms
            phi_k = k_rms
        else:
            # Additive GLA uses non-negative ELU+1 feature map
            phi_q = jax.nn.elu(q_rms) + 1.0
            phi_k = jax.nn.elu(k_rms) + 1.0

        gamma = jnp.exp(-jax.nn.softplus(delta_lambda + self.gamma_bias)) # [T, n_heads]
        beta = jax.nn.sigmoid(beta_raw) # [T, n_heads]

        if self.variant in ("gated", "adam", "channel_decay"):
            erase_raw = self.erase_proj(inputs).reshape((T, self.n_heads, self.d_k))
            erase_gate = jax.nn.sigmoid(erase_raw) * gamma[:, :, None]  # [T, n_heads, d_k]
        else:
            erase_gate = None

        # Sequential scan for Delta-rule / ADAM updates across time dimension T
        def step_fn(c, elem):
            M_t, z_t = c
            q_i, k_i, v_i, gam_i, bet_i, term_i, e_gate_i = elem

            # Clear carry if episode ended
            reset_mask = (1.0 - term_i)
            M_t = M_t * reset_mask
            z_t = z_t * reset_mask

            # 1. Read prediction
            # M_t shape: [n_heads, d_k, d_v], k_i shape: [n_heads, d_k]
            v_hat = jnp.einsum("h k v, h k -> h v", M_t, k_i)

            # 2. Compute error signal — Family A write rule selection
            if self.variant in ("delta", "adam"):
                import jax
                target_value = v_i - jax.lax.stop_gradient(v_hat)   # Delta rule: error correction
                # Spectral Radius Stabilization: Normalize beta by key squared norm to bound rho < 1.0
                k_sq_norm = jnp.sum(k_i ** 2, axis=-1) + 1e-5  # [n_heads]
                bet_eff = bet_i / k_sq_norm                     # [n_heads]
            else:
                target_value = v_i           # Additive: standard outer-product
                bet_eff = bet_i

            # Prediction error magnitude (for telemetry — meaningful for delta/adam only)
            pred_error = jnp.mean(jnp.linalg.norm(target_value, axis=-1))

            # 3. Update memory matrix — Family B erase gate selection
            if e_gate_i is not None:
                # Factorized channel-wise erase (Families B / ADAM):  diag(d_t^k) M + write
                M_next = e_gate_i[:, :, None] * M_t + bet_eff[:, None, None] * (k_i[:, :, None] * target_value[:, None, :])
            else:
                # Scalar decay (Families A-additive / A-delta)
                M_next = gam_i[:, None, None] * M_t + bet_eff[:, None, None] * (k_i[:, :, None] * target_value[:, None, :])

            z_next = gam_i[:, None] * z_t + bet_eff[:, None] * k_i

            # 4. Output retrieval for query q_i
            y_i = jnp.einsum("h k v, h k -> h v", M_next, q_i)

            return (M_next, z_next), (y_i, pred_error)

        elems = (phi_q, phi_k, v, gamma, beta, terminations,
                 erase_gate if erase_gate is not None else jnp.zeros((T, self.n_heads, self.d_k)))

        (final_M, final_z), (y_seq, pred_errors) = jax.lax.scan(step_fn, (M_prev, z_prev), elems)

        y_flat = y_seq.reshape((T, self.n_heads * self.d_v))
        out = self.out_proj(y_flat)

        telemetry = {
            "beta_mean": jnp.mean(beta),
            "gamma_mean": jnp.mean(gamma),
            "pred_error_mean": jnp.mean(pred_errors),  # E[||v_t - v_hat_t||_2]
        }

        return out, (final_M, final_z), telemetry
