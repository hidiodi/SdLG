# src/models/common/sdlg_hybrid.py
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import NamedTuple
from flax.linen.initializers import constant, orthogonal

RECORD_DEBUG = False
DEBUG_REGISTRY = []


def record_activation(layer_idx, name, val):
    if RECORD_DEBUG:
        DEBUG_REGISTRY.append((int(layer_idx), name, val))


class SdLGHybridCarry(NamedTuple):
    # GLA (Gated Linear Attention) Carry
    tilde_k: jnp.ndarray  # (r, d_k)
    tilde_v: jnp.ndarray  # (r, d_v)
    theta: jnp.ndarray  # (r,) -> Stationary phase state

    # SWA (Sliding-Window Attention) Carry
    swa_k: jnp.ndarray  # (W, d_k)
    swa_v: jnp.ndarray  # (W, d_v)

    # Intrinsic tracking
    step_count: jnp.ndarray  # () -> scalar step count


def rope_rotate(x, theta):
    """
    Apply block-wise complex rotation R(x_t, theta_{k,t}) to 2D pairs of x_t.
    Guarantees magnitude preservation: ||R(x_t, theta)|| = ||x_t||.
    x shape: (..., 1, d_k)
    theta shape: (..., r)
    """
    assert x.shape[-2] == 1, f"rope_rotate: Second-to-last dimension of x must be 1, got shape {x.shape}"
    d_k = x.shape[-1]
    assert d_k % 2 == 0, f"rope_rotate: d_k must be even for block-wise pairing, got d_k={d_k}"

    x_paired = x.reshape(x.shape[:-1] + (d_k // 2, 2))

    cos_theta = jnp.cos(theta)[..., None]
    sin_theta = jnp.sin(theta)[..., None]

    x0 = x_paired[..., 0]
    x1 = x_paired[..., 1]

    y0 = x0 * cos_theta - x1 * sin_theta
    y1 = x0 * sin_theta + x1 * cos_theta

    y = jnp.stack([y0, y1], axis=-1)
    return y.reshape(y.shape[:-2] + (d_k,))


def parallel_recurrent_scan(carry, writes, discounts):
    """
    Associative scan for parallel recurrent evaluation.
    y_t = discount_t * y_{t-1} + write_t
    """

    def combine(a, b):
        w_a, d_a = a
        w_b, d_b = b
        return w_b + d_b * w_a, d_a * d_b

    w_0 = writes[0] + discounts[0] * carry
    writes = writes.at[0].set(w_0)
    out_writes, _ = jax.lax.associative_scan(combine, (writes, discounts), axis=0)
    return out_writes


class GTrXLGRUGate(nn.Module):
    d_model: int
    bg: float = 2.0

    @nn.compact
    def __call__(self, x, y):
        # x: residual input
        # y: candidate update
        gate_proj_y = nn.Dense(2 * self.d_model, bias_init=constant(0.0), name="gate_y")(y)
        gate_proj_x = nn.Dense(2 * self.d_model, bias_init=constant(0.0), name="gate_x")(x)

        y_z, y_r = jnp.split(gate_proj_y, 2, axis=-1)
        x_z, x_r = jnp.split(gate_proj_x, 2, axis=-1)

        # CRITICAL FIX: Load-bearing subtraction of bg to initialize near identity (See PDF Sec. 5)
        # Without this minus sign, the GRU overwrites 88% of memory at step 0, causing amnesia.
        z_t = jax.nn.sigmoid(y_z + x_z - self.bg)
        r_t = jax.nn.sigmoid(y_r + x_r)

        cand_proj_y = nn.Dense(self.d_model, bias_init=constant(0.0), name="cand_y")(y)
        cand_proj_x = nn.Dense(self.d_model, bias_init=constant(0.0), name="cand_x")(r_t * x)
        h_hat = jnp.tanh(cand_proj_y + cand_proj_x)

        return (1.0 - z_t) * x + z_t * h_hat


class GatedSlidingWindowAttention(nn.Module):
    d_model: int
    d_k: int
    d_v: int
    window_size: int

    @nn.compact
    def __call__(self, inputs, terminations, swa_carry_k, swa_carry_v, carry_step_count, deterministic=True):
        T, _ = inputs.shape
        W = self.window_size

        total_swa_dims = 2 * self.d_k + self.d_v + self.d_model
        joint_swa = nn.Dense(total_swa_dims, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0),
                             name="joint_swa_proj")(inputs)

        split_idx = [self.d_k, 2 * self.d_k, 2 * self.d_k + self.d_v]
        q, k, v, g_raw = jnp.split(joint_swa, split_idx, axis=-1)

        g = jax.nn.sigmoid(g_raw)

        if deterministic:
            W_active = W
        else:
            rng = self.make_rng('random')
            W_active = jax.random.randint(rng, (), minval=1, maxval=W + 1)

        if T > 1:
            carry_k = swa_carry_k * (1.0 - terminations[0])
            carry_v = swa_carry_v * (1.0 - terminations[0])

            full_K = jnp.concatenate([carry_k, k], axis=0)
            full_V = jnp.concatenate([carry_v, v], axis=0)

            logits = jnp.einsum('t d, s d -> t s', q, full_K) / jnp.sqrt(self.d_k + 1e-8)

            t_idx = jnp.arange(T)[:, None]
            s_idx = jnp.arange(W + T)[None, :]
            valid_window = (s_idx <= t_idx + W) & (s_idx > t_idx + W - W_active)

            cum_terms = jnp.cumsum(terminations, axis=0)
            cum_terms_full = jnp.concatenate([jnp.zeros(W), cum_terms], axis=0)
            same_episode = (cum_terms[:, None] == cum_terms_full[None, :]) & (
                    (s_idx >= W - carry_step_count) | (s_idx >= W)
            )

            combined_mask = valid_window & same_episode
            logits = jnp.where(combined_mask, logits, -jnp.inf)

            attn_weights = jax.nn.softmax(logits, axis=-1)
            attn_out = jnp.einsum('t s, s d -> t d', attn_weights, full_V)

            new_swa_k = full_K[-W:]
            new_swa_v = full_V[-W:]

            valid_carry = (cum_terms_full[-W:] == cum_terms[-1])
            new_swa_k = new_swa_k * valid_carry[:, None]
            new_swa_v = new_swa_v * valid_carry[:, None]

        else:
            term = terminations[0]
            carry_k = swa_carry_k * (1.0 - term)
            carry_v = swa_carry_v * (1.0 - term)

            resetted_step_count = carry_step_count * (1.0 - term)
            write_idx = jnp.mod(resetted_step_count, W).astype(jnp.int32)

            new_swa_k = carry_k.at[write_idx].set(k[0])
            new_swa_v = carry_v.at[write_idx].set(v[0])

            logits = jnp.einsum('t d, s d -> t s', q, new_swa_k) / jnp.sqrt(self.d_k + 1e-8)

            s_idx = jnp.arange(W)
            valid_count = jnp.minimum(resetted_step_count + 1, W)
            valid_mask = s_idx < valid_count
            logits = jnp.where(valid_mask[None, :], logits, -jnp.inf)

            attn_weights = jax.nn.softmax(logits, axis=-1)
            attn_out = jnp.einsum('t s, s d -> t d', attn_weights, new_swa_v)

        attn_out = nn.Dense(self.d_model, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0),
                            name="out_proj")(attn_out)
        out = g * attn_out

        return out, new_swa_k, new_swa_v


class GatedAttentionGLALayer(nn.Module):
    d_model: int
    d_k: int
    d_v: int
    r: int
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
        joint_gla = nn.Dense(total_gla_dims, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0),
                             name="joint_gla_proj")(inputs)

        split_idx = [self.d_k, 2 * self.d_k, 2 * self.d_k + self.d_v, 2 * self.d_k + 2 * self.d_v]
        q, k, v, beta_raw, gamma_raw = jnp.split(joint_gla, split_idx, axis=-1)

        beta = jax.nn.sigmoid(beta_raw)
        gamma = jax.nn.sigmoid(gamma_raw)

        if self.use_rope_phase:
            q_rot = rope_rotate(q[..., None, :], theta)
            k_rot = rope_rotate(k[..., None, :], theta)
            write_v = jnp.cos(theta)[..., None] * (beta * v)[..., None, :]
            write_k = gamma[..., None, :] * k_rot
        else:
            # Absolute cosine scaling fallback
            abs_theta = step_counts[..., None] * omegas
            q_rot = jnp.cos(abs_theta)[..., None] * q[..., None, :]
            k_rot = jnp.cos(abs_theta)[..., None] * k[..., None, :]
            write_v = jnp.cos(abs_theta)[..., None] * (beta * v)[..., None, :]
            write_k = gamma[..., None, :] * k_rot

        discount_v = (1.0 - beta[..., None, :]) * (1.0 - terminations[..., None, None])
        discount_k = (1.0 - gamma[..., None, :]) * (1.0 - terminations[..., None, None])

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

        attn_out = nn.Dense(self.d_model, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0),
                            name="out_proj")(attn_out)
        g = jax.nn.sigmoid(
            nn.Dense(self.d_model, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0), name="g_proj")(inputs))
        out = g * attn_out

        new_tilde_k = final_tilde_k[-1]
        new_tilde_v = final_tilde_v[-1]
        new_theta = theta[-1]
        new_step_count = step_counts[-1]

        return out, intrinsic_bonus, (new_tilde_k, new_tilde_v, new_theta), new_step_count


class SdLGHybridBlock(nn.Module):
    d_model: int
    d_k: int
    d_v: int
    d_ffc: int
    r: int
    window_size: int
    layer_idx: int
    eta: float = 0.05
    gru_bias: float = 2.0
    use_rope_phase: bool = True
    use_softplus_denominator: bool = True
    use_per_rank_norm: bool = True

    @nn.compact
    def __call__(self, inputs, terminations, carry, deterministic=True):
        T, _ = inputs.shape

        norm_inputs = nn.LayerNorm(name='ln1')(inputs)
        intrinsic_bonus = jnp.zeros_like(inputs[..., 0])

        if self.layer_idx % 2 == 0:
            attn_out, new_swa_k, new_swa_v = GatedSlidingWindowAttention(
                d_model=self.d_model,
                d_k=self.d_k,
                d_v=self.d_v,
                window_size=self.window_size,
                name="swa"
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
                tilde_k=carry.tilde_k,
                tilde_v=carry.tilde_v,
                theta=carry.theta,
                swa_k=new_swa_k,
                swa_v=new_swa_v,
                step_count=new_step_count
            )
        else:
            attn_out, intrinsic_bonus, (new_tilde_k, new_tilde_v, new_theta), new_step_count = GatedAttentionGLALayer(
                d_model=self.d_model,
                d_k=self.d_k,
                d_v=self.d_v,
                r=self.r,
                eta=self.eta,
                use_rope_phase=self.use_rope_phase,
                use_softplus_denominator=self.use_softplus_denominator,
                use_per_rank_norm=self.use_per_rank_norm,
                name="gla"
            )(norm_inputs, terminations, (carry.tilde_k, carry.tilde_v, carry.theta), carry.step_count)

            new_carry = SdLGHybridCarry(
                tilde_k=new_tilde_k,
                tilde_v=new_tilde_v,
                theta=new_theta,
                swa_k=carry.swa_k,
                swa_v=carry.swa_v,
                step_count=new_step_count
            )

        # Uses the local correctly-signed GRU Gate
        gating_out = GTrXLGRUGate(self.d_model, self.gru_bias, name='gru1')(inputs, attn_out)

        norm_gating_out = nn.LayerNorm(name='ln2')(gating_out)
        ffn_out = nn.Sequential([
            nn.Dense(self.d_ffc, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0), name="dense1"),
            nn.gelu,
            nn.Dense(self.d_model, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0), name="dense2")
        ], name='ffn')(norm_gating_out)

        out = GTrXLGRUGate(self.d_model, self.gru_bias, name='gru2')(gating_out, ffn_out)

        if RECORD_DEBUG:
            jax.debug.callback(record_activation, self.layer_idx, "attn_out", attn_out)
            jax.debug.callback(record_activation, self.layer_idx, "gating_out", gating_out)
            jax.debug.callback(record_activation, self.layer_idx, "ffn_out", ffn_out)
            jax.debug.callback(record_activation, self.layer_idx, "out", out)

        return out, intrinsic_bonus, new_carry


class SdLGHybridFullBlock(nn.Module):
    """One local--factorized-memory block used by matched execution studies.

    The historical hybrid stack alternates local and factorized-memory layers.
    This opt-in block instead applies both operations inside every layer, matching
    the Exact stack's local -> associative -> gated-FFN ordering while retaining
    the existing factorized recurrence unchanged.
    """
    d_model: int
    d_k: int
    d_v: int
    d_ffc: int
    r: int
    window_size: int
    eta: float = 0.05
    gru_bias: float = 2.0
    use_rope_phase: bool = True
    use_softplus_denominator: bool = True
    use_per_rank_norm: bool = True

    @nn.compact
    def __call__(self, inputs, terminations, carry, deterministic=True):
        norm_swa = nn.LayerNorm(name="ln_swa")(inputs)
        swa_out, new_swa_k, new_swa_v = GatedSlidingWindowAttention(
            d_model=self.d_model,
            d_k=self.d_k,
            d_v=self.d_v,
            window_size=self.window_size,
            name="swa",
        )(
            norm_swa, terminations, carry.swa_k, carry.swa_v, carry.step_count,
            deterministic=deterministic,
        )
        x_swa = GTrXLGRUGate(self.d_model, self.gru_bias, name="gru_swa")(inputs, swa_out)

        if inputs.shape[0] > 1:
            def increment_step(carry_step, termination):
                next_step = (carry_step + 1.0) * (1.0 - termination)
                return next_step, next_step

            _, step_counts = jax.lax.scan(increment_step, carry.step_count, terminations)
            new_step_count = step_counts[-1]
        else:
            new_step_count = (carry.step_count + 1.0) * (1.0 - terminations[0])

        norm_gla = nn.LayerNorm(name="ln_gla")(x_swa)
        gla_out, intrinsic_bonus, (new_tilde_k, new_tilde_v, new_theta), _ = GatedAttentionGLALayer(
            d_model=self.d_model,
            d_k=self.d_k,
            d_v=self.d_v,
            r=self.r,
            eta=self.eta,
            use_rope_phase=self.use_rope_phase,
            use_softplus_denominator=self.use_softplus_denominator,
            use_per_rank_norm=self.use_per_rank_norm,
            name="gla",
        )(
            norm_gla,
            terminations,
            (carry.tilde_k, carry.tilde_v, carry.theta),
            carry.step_count,
        )
        x_gla = GTrXLGRUGate(self.d_model, self.gru_bias, name="gru_gla")(x_swa, gla_out)

        norm_ffn = nn.LayerNorm(name="ln_ffn")(x_gla)
        ffn_out = nn.Sequential([
            nn.Dense(self.d_ffc, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0), name="dense1"),
            nn.gelu,
            nn.Dense(self.d_model, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0), name="dense2"),
        ], name="ffn")(norm_ffn)
        out = GTrXLGRUGate(self.d_model, self.gru_bias, name="gru_ffn")(x_gla, ffn_out)

        new_carry = SdLGHybridCarry(
            tilde_k=new_tilde_k,
            tilde_v=new_tilde_v,
            theta=new_theta,
            swa_k=new_swa_k,
            swa_v=new_swa_v,
            step_count=new_step_count,
        )
        return out, intrinsic_bonus, new_carry


class BatchedSdLGHybrid(nn.Module):
    n_layers: int
    d_model: int
    d_k: int
    d_v: int
    d_ffc: int
    r: int
    window_size: int
    alpha: float = 0.1
    eta: float = 0.05
    return_split: bool = True
    use_rope_phase: bool = True
    use_softplus_denominator: bool = True
    use_per_rank_norm: bool = True
    topology: str = "alternating"
    gru_bias: float = 2.0

    @nn.compact
    def __call__(self, inputs, terminations, carry, deterministic=True):
        # 0. Episode reset applied before state update (See PDF Sec. 7)
        if inputs.shape[0] == 1:
            # During rollout (T=1), manually clear the boundary here
            reset_mask = 1.0 - terminations[0]
            carry = jax.tree_util.tree_map(lambda x: x * reset_mask, carry)

        if inputs.shape[-1] != self.d_model:
            x = nn.Dense(self.d_model, name='emb_layer', kernel_init=orthogonal(jnp.sqrt(2)),
                         bias_init=constant(0.0))(inputs)
            x = nn.relu(x)
        else:
            x = inputs

        new_carry = []
        bonuses = []
        for layer_idx in range(self.n_layers):
            layer_carry = carry[layer_idx]

            block_kwargs = dict(
                d_model=self.d_model,
                d_k=self.d_k,
                d_v=self.d_v,
                d_ffc=self.d_ffc,
                r=self.r,
                window_size=self.window_size,
                eta=self.eta,
                use_rope_phase=self.use_rope_phase,
                use_softplus_denominator=self.use_softplus_denominator,
                use_per_rank_norm=self.use_per_rank_norm,
                gru_bias=self.gru_bias,
                name=f'layer_{layer_idx}',
            )
            if self.topology == "full":
                block = SdLGHybridFullBlock(**block_kwargs)
            elif self.topology == "alternating":
                block = SdLGHybridBlock(layer_idx=layer_idx, **block_kwargs)
            else:
                raise ValueError(f"Unsupported SdLG topology: {self.topology}")
            x, intrinsic_bonus, updated_layer_carry = block(
                x, terminations, layer_carry, deterministic=deterministic
            )

            new_carry.append(updated_layer_carry)
            bonuses.append(intrinsic_bonus)

        new_carry_tuple = tuple(new_carry)
        total_intrinsic_bonus = jnp.mean(jnp.stack(bonuses, axis=0), axis=0)

        if self.return_split:
            X_shared = x
            X_critic = X_shared
            X_actor = jax.lax.stop_gradient(X_shared) * (1.0 - self.alpha) + X_shared * self.alpha
            return ((X_actor, X_critic), total_intrinsic_bonus), new_carry_tuple
        else:
            return (x, total_intrinsic_bonus), new_carry_tuple

    @staticmethod
    def initialize_carry(batch_size, n_layers, d_k, d_v, r, window_size):
        carry_list = []
        for _ in range(n_layers):
            tilde_k = jnp.zeros((r, d_k))
            tilde_v = jnp.zeros((r, d_v))
            theta = jnp.zeros((r,))
            step_count = jnp.zeros(())

            swa_k = jnp.zeros((window_size, d_k))
            swa_v = jnp.zeros((window_size, d_v))

            carry_list.append(SdLGHybridCarry(
                tilde_k=tilde_k,
                tilde_v=tilde_v,
                theta=theta,
                swa_k=swa_k,
                swa_v=swa_v,
                step_count=step_count
            ))
        return tuple(carry_list)
