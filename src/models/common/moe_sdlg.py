import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import NamedTuple, Any
from flax.linen.initializers import constant, orthogonal
from src.models.common.sdlg_hybrid import SdLGHybridCarry, rope_rotate, parallel_recurrent_scan, GTrXLGRUGate, GatedSlidingWindowAttention, BatchedSdLGHybrid
from src.models.purejaxrl.wrappers import RepresentationNetwork
import distrax


class SparseMoESwiGLU(nn.Module):
    d_model: int
    d_ffc: int
    num_experts: int = 4
    top_k: int = 2

    @nn.compact
    def __call__(self, x):
        # Router
        router_logits = nn.Dense(
            self.num_experts,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
            name="router"
        )(x)
        router_probs = jax.nn.softmax(router_logits, axis=-1)

        # Get top-k probabilities and indices
        top_k_probs, top_k_indices = jax.lax.top_k(router_probs, self.top_k)

        # Normalize top-k probabilities
        prob_sum = jnp.sum(top_k_probs, axis=-1, keepdims=True)
        top_k_probs_normalized = top_k_probs / (prob_sum + 1e-8)

        # Create a mask for top-k experts
        # One-hot encode indices: shape (..., top_k, num_experts)
        one_hot_indices = jax.nn.one_hot(top_k_indices, self.num_experts)
        # Multiply one-hot by normalized probs and sum over top_k to get a mask: shape (..., num_experts)
        expert_weights = jnp.sum(one_hot_indices * top_k_probs_normalized[..., None], axis=-2)

        # Execute all experts
        expert_outputs = []
        for i in range(self.num_experts):
            # SwiGLU expert: FFN(x) = (xW1 * SiLU(xW3))W2
            # Let's project x to 2 * d_ffc and split to W1 and W3
            dense_w1_w3 = nn.Dense(
                2 * self.d_ffc,
                kernel_init=orthogonal(jnp.sqrt(2)),
                bias_init=constant(0.0),
                name=f"expert_{i}_w1_w3"
            )(x)
            x_w1, x_w3 = jnp.split(dense_w1_w3, 2, axis=-1)
            hidden = x_w1 * jax.nn.silu(x_w3)
            expert_out = nn.Dense(
                self.d_model,
                kernel_init=orthogonal(jnp.sqrt(2)),
                bias_init=constant(0.0),
                name=f"expert_{i}_w2"
            )(hidden)
            expert_outputs.append(expert_out)

        # Stack outputs: shape (..., num_experts, d_model)
        stacked_expert_outputs = jnp.stack(expert_outputs, axis=-2)

        # Multiply by routing weights and sum
        out = jnp.sum(stacked_expert_outputs * expert_weights[..., None], axis=-2)
        return out


class GatedAttentionGLAELULayer(nn.Module):
    d_model: int
    d_k: int
    d_v: int
    r: int
    eta: float = 0.05
    eps: float = 1e-5

    @nn.compact
    def __call__(self, inputs, terminations, last_memory, carry_step_count):
        T = inputs.shape[0]
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

        # ELU + 1 activation for bounded non-negativity
        q_pos = jax.nn.elu(q) + 1.0
        k_pos = jax.nn.elu(k) + 1.0

        # No RoPE phase rotation here, just use pos features
        write_v = (beta * v)[..., None, :]
        write_k = gamma[..., None, :] * k_pos[..., None, :]
        q_feat = q_pos[..., None, :]

        discount_v = (1.0 - beta[..., None, :]) * (1.0 - terminations[..., None, None])
        discount_k = (1.0 - gamma[..., None, :]) * (1.0 - terminations[..., None, None])

        if T > 1:
            final_tilde_k = parallel_recurrent_scan(tilde_k_prev, write_k, discount_k)
            final_tilde_v = parallel_recurrent_scan(tilde_v_prev, write_v, discount_v)
        else:
            final_tilde_k = jnp.expand_dims(discount_k[0] * tilde_k_prev + write_k[0], axis=0)
            final_tilde_v = jnp.expand_dims(discount_v[0] * tilde_v_prev + write_v[0], axis=0)

        dot_prod = jnp.sum(q_feat * final_tilde_k, axis=-1)
        num = jnp.sum(final_tilde_v * dot_prod[..., None], axis=-2)
        den = jnp.sum(final_tilde_k * q_feat, axis=(-2, -1))

        # No softplus needed since q and k are strictly positive
        den = jnp.maximum(den, self.eps)

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


class MoESdLGHybridBlock(nn.Module):
    d_model: int
    d_k: int
    d_v: int
    d_ffc: int
    r: int
    window_size: int
    layer_idx: int
    eta: float = 0.05
    gru_bias: float = 2.0

    @nn.compact
    def __call__(self, inputs, terminations, carry, deterministic=True):
        T = inputs.shape[0]

        norm_inputs = nn.RMSNorm(epsilon=1e-8, name='ln1')(inputs)
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
            attn_out, intrinsic_bonus, (new_tilde_k, new_tilde_v, new_theta), new_step_count = GatedAttentionGLAELULayer(
                d_model=self.d_model,
                d_k=self.d_k,
                d_v=self.d_v,
                r=self.r,
                eta=self.eta,
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

        gating_out = GTrXLGRUGate(self.d_model, self.gru_bias, name='gru1')(inputs, attn_out)

        norm_gating_out = nn.RMSNorm(epsilon=1e-8, name='ln2')(gating_out)

        ffn_out = SparseMoESwiGLU(
            d_model=self.d_model,
            d_ffc=self.d_ffc,
            name='ffn'
        )(norm_gating_out)

        out = GTrXLGRUGate(self.d_model, self.gru_bias, name='gru2')(gating_out, ffn_out)

        return out, intrinsic_bonus, new_carry


class BatchedMoESdLG(nn.Module):
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

    @nn.compact
    def __call__(self, inputs, terminations, carry, deterministic=True):
        if inputs.shape[0] == 1:
            reset_mask = 1.0 - terminations[0]
            carry = jax.tree_util.tree_map(lambda x: (x * reset_mask).astype(x.dtype), carry)

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

            x, intrinsic_bonus, updated_layer_carry = MoESdLGHybridBlock(
                d_model=self.d_model,
                d_k=self.d_k,
                d_v=self.d_v,
                d_ffc=self.d_ffc,
                r=self.r,
                window_size=self.window_size,
                layer_idx=layer_idx,
                eta=self.eta,
                name=f'layer_{layer_idx}'
            )(x, terminations, layer_carry, deterministic=deterministic)

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
        # Create a batch of 1 and squeeze it out to return an unbatched carry for vmap!
        carry = BatchedSdLGHybrid.initialize_carry(1, n_layers, d_k, d_v, r, window_size)
        return jax.tree_util.tree_map(lambda x: jnp.squeeze(x, axis=0), carry)


class ActorCriticMoESdLG(nn.Module):
    action_dim: int
    config: dict

    @nn.compact
    def __call__(self, carry, x, deterministic=True):
        obs, dones = x

        emb = RepresentationNetwork(
            d_model=self.config["RNN"]["D_MODEL"], name="repr_network"
        )(obs)

        # Transpose to (B, T, D) for vmap
        emb_trans = jnp.transpose(emb, (1, 0, 2))
        dones_trans = jnp.transpose(dones, (1, 0))

        VmappedMoESdLG = nn.vmap(
            BatchedMoESdLG,
            in_axes=(0, 0, 0, None),
            out_axes=0,
            variable_axes={'params': None},
            split_rngs={'params': False, 'random': True}
        )

        rnn = VmappedMoESdLG(
            n_layers=self.config["RNN"]["N_LAYERS"],
            d_model=self.config["RNN"]["D_MODEL"],
            d_k=self.config["RNN"]["D_K"],
            d_v=self.config["RNN"]["D_V"],
            d_ffc=self.config["RNN"]["D_FFC"],
            r=self.config["RNN"]["R"],
            window_size=self.config["RNN"]["WINDOW_SIZE"],
            alpha=self.config["RNN"]["ALPHA"],
            eta=self.config["RNN"]["ETA"],
            return_split=True,
            name="moe_sdlg"
        )

        ((x_actor_trans, x_critic_trans), intrinsic_bonus_trans), new_carry = rnn(emb_trans, dones_trans, carry, deterministic)

        # Transpose back to (T, B, D)
        x_actor = jnp.transpose(x_actor_trans, (1, 0, 2))
        x_critic = jnp.transpose(x_critic_trans, (1, 0, 2))
        intrinsic_bonus = jnp.transpose(intrinsic_bonus_trans, (1, 0))

        action_logits = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), name="actor_head"
        )(x_actor)

        action_logits = 5.0 * jnp.tanh(jnp.nan_to_num(action_logits, nan=0.0, posinf=5.0, neginf=-5.0))

        critic_values = nn.Dense(
            1, kernel_init=orthogonal(1.0), name="critic_head"
        )(x_critic)
        critic_values = jnp.squeeze(critic_values, axis=-1)
        critic_values = jnp.nan_to_num(critic_values, nan=0.0, posinf=10.0, neginf=-10.0)

        return action_logits, critic_values, intrinsic_bonus, new_carry
