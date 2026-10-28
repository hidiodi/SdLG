# src/models/common/sdlg_v2.py
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import NamedTuple, Tuple, Any
from flax.linen.initializers import constant, orthogonal
import distrax

from src.models.common.sdlg_hybrid import (
    rope_rotate,
    parallel_recurrent_scan,
    GTrXLGRUGate,
    GatedSlidingWindowAttention,
)
from src.models.purejaxrl.wrappers import RepresentationNetwork

class SdLGV2Carry(NamedTuple):
    tilde_k: jnp.ndarray   # (r, d_k)
    tilde_v: jnp.ndarray   # (r, d_v)
    theta: jnp.ndarray     # (r,)
    swa_k: jnp.ndarray     # (W, d_k)
    swa_v: jnp.ndarray     # (W, d_v)
    step_count: jnp.ndarray # ()


class GatedAttentionGLALayerV2(nn.Module):
    d_model: int
    d_k: int
    d_v: int
    r: int
    use_vector_decay: bool = False
    use_cross_gate: bool = False
    use_feature_gate: bool = True
    use_critic_read: bool = False
    eta: float = 0.0
    eps: float = 1e-5
    use_rope_phase: bool = True
    use_softplus_denominator: bool = True
    use_per_rank_norm: bool = True

    @nn.compact
    def __call__(self, inputs, terminations, last_memory, carry_step_count, swa_features=None):
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

        # Feature Flag 2: Cross-Gating from SWA output to Recurrence Gating
        gating_input = swa_features if (self.use_cross_gate and swa_features is not None) else inputs

        # Determine projection shapes based on Feature Flags 1 & 3
        d_beta = self.d_v if self.use_feature_gate else 1
        d_gamma = self.d_k if self.use_vector_decay else 1

        total_gla_dims = 2 * self.d_k + self.d_v + d_beta + d_gamma
        if self.use_critic_read:
            total_gla_dims += self.d_k  # Extra query vector for Critic

        joint_gla = nn.Dense(total_gla_dims, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0),
                             name="joint_gla_proj")(gating_input)

        if self.use_critic_read:
            split_idx = [self.d_k, 2 * self.d_k, 3 * self.d_k, 3 * self.d_k + self.d_v, 3 * self.d_k + self.d_v + d_beta]
            q_actor, q_critic, k, v, beta_raw, gamma_raw = jnp.split(joint_gla, split_idx, axis=-1)
        else:
            split_idx = [self.d_k, 2 * self.d_k, 2 * self.d_k + self.d_v, 2 * self.d_k + self.d_v + d_beta]
            q_actor, k, v, beta_raw, gamma_raw = jnp.split(joint_gla, split_idx, axis=-1)
            q_critic = None

        beta = jax.nn.sigmoid(beta_raw - 4.5)
        gamma = jax.nn.sigmoid(gamma_raw - 4.5)

        if self.use_rope_phase:
            q_rot_actor = rope_rotate(q_actor[..., None, :], theta)
            q_rot_critic = rope_rotate(q_critic[..., None, :], theta) if q_critic is not None else None
            k_rot = rope_rotate(k[..., None, :], theta)
            write_v = jnp.cos(theta)[..., None] * (beta * v)[..., None, :]
            write_k = gamma[..., None, :] * k_rot
        else:
            abs_theta = step_counts[..., None] * omegas
            q_rot_actor = jnp.cos(abs_theta)[..., None] * q_actor[..., None, :]
            q_rot_critic = jnp.cos(abs_theta)[..., None] * q_critic[..., None, :] if q_critic is not None else None
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

        def read_memory(q_rot):
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
            return attn_out, den

        out_actor, den_actor = read_memory(q_rot_actor)
        out_actor = nn.Dense(self.d_model, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0), name="out_proj_actor")(out_actor)

        if self.use_critic_read:
            out_critic, _ = read_memory(q_rot_critic)
            out_critic = nn.Dense(self.d_model, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0), name="out_proj_critic")(out_critic)
        else:
            out_critic = out_actor

        g = jax.nn.sigmoid(
            nn.Dense(self.d_model, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0), name="g_proj")(inputs))

        out_actor = g * out_actor
        out_critic = g * out_critic if self.use_critic_read else out_actor

        step_weight = jnp.clip((step_counts - 32) / 32, 0.0, 1.0)
        intrinsic_bonus = step_weight * (self.eta / jax.lax.stop_gradient(den_actor))

        new_tilde_k = final_tilde_k[-1]
        new_tilde_v = final_tilde_v[-1]
        new_theta = theta[-1]
        new_step_count = step_counts[-1]

        return (out_actor, out_critic), intrinsic_bonus, (new_tilde_k, new_tilde_v, new_theta), new_step_count


class SdLGV2Block(nn.Module):
    d_model: int
    d_k: int
    d_v: int
    d_ffc: int
    r: int
    window_size: int
    layer_idx: int
    use_vector_decay: bool = False
    use_cross_gate: bool = False
    use_feature_gate: bool = True
    use_critic_read: bool = False
    eta: float = 0.0
    gru_bias: float = 2.0

    @nn.compact
    def __call__(self, inputs, terminations, carry: SdLGV2Carry):
        norm_swa = nn.LayerNorm()(inputs)
        swa_features, new_swa_k, new_swa_v = GatedSlidingWindowAttention(
            d_model=self.d_model, d_k=self.d_k, d_v=self.d_v, window_size=self.window_size
        )(norm_swa, terminations, carry.swa_k, carry.swa_v, carry.step_count)

        x_swa = GTrXLGRUGate(self.d_model, self.gru_bias)(inputs, swa_features)

        norm_gla = nn.LayerNorm()(x_swa)
        (gla_actor, gla_critic), intrinsic_bonus, (new_tilde_k, new_tilde_v, new_theta), new_step_count = GatedAttentionGLALayerV2(
            d_model=self.d_model, d_k=self.d_k, d_v=self.d_v, r=self.r,
            use_vector_decay=self.use_vector_decay,
            use_cross_gate=self.use_cross_gate,
            use_feature_gate=self.use_feature_gate,
            use_critic_read=self.use_critic_read,
            eta=self.eta
        )(norm_gla, terminations, (carry.tilde_k, carry.tilde_v, carry.theta), carry.step_count, swa_features=swa_features)

        x_gla_actor = GTrXLGRUGate(self.d_model, self.gru_bias)(x_swa, gla_actor)
        x_gla_critic = GTrXLGRUGate(self.d_model, self.gru_bias)(x_swa, gla_critic) if self.use_critic_read else x_gla_actor

        norm_ffn_actor = nn.LayerNorm()(x_gla_actor)
        ffn_out_actor = nn.Sequential([
            nn.Dense(self.d_ffc, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)), nn.gelu,
            nn.Dense(self.d_model, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))
        ])(norm_ffn_actor)
        out_actor = GTrXLGRUGate(self.d_model, self.gru_bias)(x_gla_actor, ffn_out_actor)

        if self.use_critic_read:
            norm_ffn_critic = nn.LayerNorm()(x_gla_critic)
            ffn_out_critic = nn.Sequential([
                nn.Dense(self.d_ffc, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)), nn.gelu,
                nn.Dense(self.d_model, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))
            ])(norm_ffn_critic)
            out_critic = GTrXLGRUGate(self.d_model, self.gru_bias)(x_gla_critic, ffn_out_critic)
        else:
            out_critic = out_actor

        new_carry = SdLGV2Carry(
            tilde_k=new_tilde_k, tilde_v=new_tilde_v, theta=new_theta,
            swa_k=new_swa_k, swa_v=new_swa_v, step_count=new_step_count
        )

        return (out_actor, out_critic), intrinsic_bonus, new_carry


class BatchedSdLGV2(nn.Module):
    n_layers: int
    d_model: int
    d_k: int
    d_v: int
    d_ffc: int
    r: int
    window_size: int
    use_vector_decay: bool = False
    use_cross_gate: bool = False
    use_feature_gate: bool = True
    use_critic_read: bool = False
    eta: float = 0.0
    gru_bias: float = 2.0

    @nn.compact
    def __call__(self, inputs, terminations, carry_tuple: Tuple[SdLGV2Carry, ...]):
        x = inputs

        new_carries = []
        total_intrinsic_bonus = 0.0

        x_actor, x_critic = x, x

        for l in range(self.n_layers):
            (x_actor, x_critic), intrinsic_bonus, next_carry = SdLGV2Block(
                d_model=self.d_model, d_k=self.d_k, d_v=self.d_v, d_ffc=self.d_ffc,
                r=self.r, window_size=self.window_size, layer_idx=l,
                use_vector_decay=self.use_vector_decay,
                use_cross_gate=self.use_cross_gate,
                use_feature_gate=self.use_feature_gate,
                use_critic_read=self.use_critic_read,
                eta=self.eta, gru_bias=self.gru_bias
            )(x_actor, terminations, carry_tuple[l])

            new_carries.append(next_carry)
            total_intrinsic_bonus = total_intrinsic_bonus + intrinsic_bonus

        return (x_actor, x_critic), total_intrinsic_bonus, tuple(new_carries)

    @staticmethod
    def initialize_carry(batch_size: int, n_layers: int, d_k: int, d_v: int, r: int, window_size: int) -> Tuple[SdLGV2Carry, ...]:
        carries = []
        for _ in range(n_layers):
            tilde_k = jnp.zeros((r, d_k), dtype=jnp.float32)
            tilde_v = jnp.zeros((r, d_v), dtype=jnp.float32)
            theta = jnp.zeros((r,), dtype=jnp.float32)
            swa_k = jnp.zeros((window_size, d_k), dtype=jnp.float32)
            swa_v = jnp.zeros((window_size, d_v), dtype=jnp.float32)
            step_count = jnp.zeros((), dtype=jnp.float32)
            carries.append(SdLGV2Carry(
                tilde_k=tilde_k, tilde_v=tilde_v, theta=theta,
                swa_k=swa_k, swa_v=swa_v, step_count=step_count
            ))
        return tuple(carries)


class ActorCriticSdLGV2(nn.Module):
    rnn_module: Any
    action_dim: int
    config: dict

    @nn.compact
    def __call__(self, carry, x):
        obs, dones = x
        emb = RepresentationNetwork(d_model=self.config["RNN"]["D_MODEL"], name="repr_network")(obs)

        # Transpose to Batch-First (B, T, d_model) for sequence model vmap
        emb_trans = jnp.transpose(emb, (1, 0, 2))
        dones_trans = jnp.transpose(dones, (1, 0))

        VmappedSdLGV2 = nn.vmap(
            BatchedSdLGV2,
            in_axes=(0, 0, 0),
            out_axes=0,
            variable_axes={'params': None},
            split_rngs={'params': False}
        )

        sdlg_v2 = VmappedSdLGV2(
            n_layers=self.config["RNN"].get("N_LAYERS", 4),
            d_model=self.config["RNN"]["D_MODEL"],
            d_k=self.config["RNN"].get("D_K", 32),
            d_v=self.config["RNN"].get("D_V", 32),
            d_ffc=self.config["RNN"].get("D_FFC", 1024),
            r=self.config["RNN"].get("R", 4),
            window_size=self.config["RNN"].get("WINDOW_SIZE", 8),
            use_vector_decay=self.config["RNN"].get("USE_VECTOR_DECAY", False),
            use_cross_gate=self.config["RNN"].get("USE_CROSS_GATE", False),
            use_feature_gate=self.config["RNN"].get("USE_FEATURE_GATE", True),
            use_critic_read=self.config["RNN"].get("USE_CRITIC_READ", False),
            eta=self.config["RNN"].get("ETA", 0.0)
        )

        (outs_actor_trans, outs_critic_trans), total_intrinsic_bonus_trans, new_carry = sdlg_v2(emb_trans, dones_trans, carry)

        outs_actor = jnp.transpose(outs_actor_trans, (1, 0, 2))
        outs_critic = jnp.transpose(outs_critic_trans, (1, 0, 2))
        total_intrinsic_bonus = jnp.transpose(total_intrinsic_bonus_trans, (1, 0))

        action_logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(outs_actor)
        action_logits = 5.0 * jnp.tanh(jnp.nan_to_num(action_logits, nan=0.0, posinf=5.0, neginf=-5.0))

        critic_raw = nn.Dense(1, kernel_init=orthogonal(1.0))(outs_critic)
        critic_values = jnp.nan_to_num(jnp.squeeze(critic_raw, axis=-1), nan=0.0, posinf=10.0, neginf=-10.0)

        return action_logits, critic_values, total_intrinsic_bonus, new_carry

