import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import NamedTuple
from flax.linen.initializers import constant, orthogonal

class MoESdLGv2Carry(NamedTuple):
    # GLA Exact State Carry
    S: jnp.ndarray  # (n_heads, d_k, d_v)
    z: jnp.ndarray  # (n_heads, d_k)

    # SWA Carry
    swa_k: jnp.ndarray  # (W, d_k)
    swa_v: jnp.ndarray  # (W, d_v)

    # Intrinsic tracking
    step_count: jnp.ndarray  # () -> scalar step count

class RMSNorm(nn.Module):
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x):
        norm = x * jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + self.eps)
        scale = self.param('scale', constant(1.0), (x.shape[-1],))
        return norm * scale

def rope_rotate(x, theta):
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

class GatedSlidingWindowAttention(nn.Module):
    d_model: int
    d_k: int
    d_v: int
    window_size: int

    @nn.compact
    def __call__(self, inputs, terminations, swa_carry_k, swa_carry_v, carry_step_count, deterministic=True):
        T = inputs.shape[0]
        W = self.window_size

        total_swa_dims = 2 * self.d_k + self.d_v + self.d_model
        joint_swa = nn.Dense(total_swa_dims, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0),
                             name="joint_swa_proj")(inputs)

        split_idx = [self.d_k, 2 * self.d_k, 2 * self.d_k + self.d_v]
        q, k, v, g_raw = jnp.split(joint_swa, split_idx, axis=-1)

        g = jax.nn.sigmoid(g_raw)

        def compute_step_counts(carry, term):
            next_step = (carry + 1.0) * (1.0 - term)
            return next_step, next_step

        if T > 1:
            _, step_counts = jax.lax.scan(compute_step_counts, carry_step_count, terminations)
        else:
            step_counts = jnp.expand_dims((carry_step_count + 1.0) * (1.0 - terminations[0]), axis=0)

        r_rope = self.d_k // 2
        omegas = 10000 ** (-2 * jnp.arange(r_rope) / self.d_k)

        theta_q = step_counts[..., None, None] * omegas
        q = rope_rotate(q[..., None, :], theta_q).squeeze(-2)

        if deterministic:
            W_active = W
        else:
            rng = self.make_rng('random')
            W_active = jax.random.randint(rng, (), minval=1, maxval=W + 1)

        theta_k = step_counts[..., None, None] * omegas
        k_rot = rope_rotate(k[..., None, :], theta_k).squeeze(-2)

        if T > 1:
            carry_k = swa_carry_k * (1.0 - terminations[0])
            carry_v = swa_carry_v * (1.0 - terminations[0])

            full_K = jnp.concatenate([carry_k, k_rot], axis=0)
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

            new_swa_k = carry_k.at[write_idx].set(k_rot[0])
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

class GLA_KV_Layer(nn.Module):
    d_model: int
    n_heads: int
    d_k: int
    d_v: int
    eps: float = 1e-5

    @nn.compact
    def __call__(self, inputs, terminations, S_prev, z_prev, carry_step_count):
        T = inputs.shape[0]

        q_raw = nn.Dense(self.n_heads * self.d_k, use_bias=False, name="q_proj")(inputs).reshape(T, self.n_heads, self.d_k)
        k_raw = nn.Dense(self.n_heads * self.d_k, use_bias=False, name="k_proj")(inputs).reshape(T, self.n_heads, self.d_k)
        v_raw = nn.Dense(self.n_heads * self.d_v, use_bias=False, name="v_proj")(inputs).reshape(T, self.n_heads, self.d_v)

        q = jax.nn.elu(q_raw) + 1.0
        k = jax.nn.elu(k_raw) + 1.0
        v = v_raw

        gamma = jax.nn.sigmoid(nn.Dense(self.n_heads, bias_init=constant(-4.0), name="gamma_dense")(inputs)) # (T, n_heads)

        S_prev = S_prev.astype(jnp.float32)
        z_prev = z_prev.astype(jnp.float32)

        def scan_fn(carry, xs):
            S, z, step_c = carry
            q_t, k_t, v_t, gamma_t, term_t = xs

            S = S * (1.0 - term_t)
            z = z * (1.0 - term_t)
            step_c = (step_c + 1.0) * (1.0 - term_t)

            gamma_t_exp = gamma_t[:, None, None]

            S_next = (1.0 - gamma_t_exp) * S + gamma_t_exp * (k_t[:, :, None] * v_t[:, None, :])
            z_next = (1.0 - gamma_t[:, None]) * z + gamma_t[:, None] * k_t

            y_t_num = jnp.einsum('h d v, h d -> h v', S_next, q_t)
            y_t_den = jnp.einsum('h d, h d -> h', z_next, q_t) + self.eps

            y_t = y_t_num / y_t_den[:, None]

            return (S_next.astype(jnp.float32), z_next.astype(jnp.float32), step_c), y_t

        if T > 1:
            (S_final, z_final, step_count_final), y = jax.lax.scan(
                scan_fn, (S_prev, z_prev, carry_step_count), (q, k, v, gamma, terminations)
            )
            y = y.reshape(T, self.n_heads * self.d_v)
        else:
            (S_final, z_final, step_count_final), y = scan_fn(
                (S_prev, z_prev, carry_step_count),
                (q[0], k[0], v[0], gamma[0], terminations[0])
            )
            y = y.reshape(1, self.n_heads * self.d_v)

        attn_out = nn.Dense(self.d_model, use_bias=False, name="out_proj")(y)
        g = jax.nn.sigmoid(nn.Dense(self.d_model, use_bias=True, name="g_proj")(inputs))
        out = g * attn_out

        return out, (S_final, z_final), step_count_final

class GTrXLGRUGate(nn.Module):
    d_model: int
    bg: float = 2.0

    @nn.compact
    def __call__(self, x, y):
        gate_proj_y = nn.Dense(2 * self.d_model, bias_init=constant(0.0), name="gate_y")(y)
        gate_proj_x = nn.Dense(2 * self.d_model, bias_init=constant(0.0), name="gate_x")(x)

        y_z, y_r = jnp.split(gate_proj_y, 2, axis=-1)
        x_z, x_r = jnp.split(gate_proj_x, 2, axis=-1)

        z_t = jax.nn.sigmoid(y_z + x_z - self.bg)
        r_t = jax.nn.sigmoid(y_r + x_r)

        cand_proj_y = nn.Dense(self.d_model, bias_init=constant(0.0), name="cand_y")(y)
        cand_proj_x = nn.Dense(self.d_model, bias_init=constant(0.0), name="cand_x")(r_t * x)
        h_hat = jnp.tanh(cand_proj_y + cand_proj_x)

        return (1.0 - z_t) * x + z_t * h_hat

class SwiGLU(nn.Module):
    d_model: int
    d_ffc: int

    @nn.compact
    def __call__(self, x):
        gate = nn.Dense(self.d_ffc, use_bias=False)(x)
        up = nn.Dense(self.d_ffc, use_bias=False)(x)
        intermediate = jax.nn.silu(gate) * up
        down = nn.Dense(self.d_model, use_bias=False)(intermediate)
        return down

class SparseMoESwiGLU(nn.Module):
    d_model: int
    d_ffc: int
    num_experts: int = 4
    top_k: int = 2

    @nn.compact
    def __call__(self, x):
        prefix_shape = x.shape[:-1]
        gate_logits = nn.Dense(self.num_experts, use_bias=False, name="router")(x)
        routing_probs = jax.nn.softmax(gate_logits, axis=-1)

        # Top-K (using jax.lax.top_k instead of approx_max_k)
        topk_probs, topk_indices = jax.lax.top_k(routing_probs, k=self.top_k)

        # Load balancing loss: N * sum(f_i * p_i)
        # f_i = fraction of tokens routed to expert i
        # p_i = mean routing prob for expert i
        mask = jax.nn.one_hot(topk_indices, self.num_experts).sum(axis=-2)
        axes_to_mean = tuple(range(len(prefix_shape)))
        f_i = jnp.mean(mask, axis=axes_to_mean) # (num_experts,)
        p_i = jnp.mean(routing_probs, axis=axes_to_mean) # (num_experts,)
        l_bal = self.num_experts * jnp.sum(f_i * p_i)

        expert_weights = topk_probs / (jnp.sum(topk_probs, axis=-1, keepdims=True) + 1e-9)

        # Compute expert outputs in a list and stack to avoid mutation
        expert_outs = []
        for e in range(self.num_experts):
            out = SwiGLU(self.d_model, self.d_ffc, name=f"expert_{e}")(x)
            expert_outs.append(out)
        expert_outputs = jnp.stack(expert_outs, axis=-2)

        combined_output = jnp.zeros_like(x)

        # We need a robust gather over arbitrary batch dims.
        # expert_outputs is (*prefix, num_experts, d_model)
        # topk_indices is (*prefix, top_k)

        for k in range(self.top_k):
            indices = topk_indices[..., k]
            weights = expert_weights[..., k]

            # Use jnp.take_along_axis to extract the correct expert output
            # indices has shape (*prefix) -> expand to (*prefix, 1, d_model) to broadcast
            expanded_indices = jnp.broadcast_to(indices[..., None, None], (*prefix_shape, 1, self.d_model))
            selected_outputs = jnp.take_along_axis(expert_outputs, expanded_indices, axis=-2).squeeze(-2)

            combined_output += weights[..., None] * selected_outputs

        expert_usages = jnp.mean(mask, axis=axes_to_mean)
        return combined_output, l_bal, expert_usages

class MoESdLGv2Block(nn.Module):
    d_model: int
    n_heads: int
    d_k: int
    d_v: int
    d_ffc: int
    window_size: int
    layer_idx: int
    gru_bias: float = 2.0

    @nn.compact
    def __call__(self, inputs, terminations, carry, deterministic=True):
        T = inputs.shape[0]
        norm_inputs = RMSNorm(name='ln1')(inputs)

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

            new_carry = MoESdLGv2Carry(
                S=carry.S,
                z=carry.z,
                swa_k=new_swa_k,
                swa_v=new_swa_v,
                step_count=new_step_count
            )
        else:
            attn_out, (new_S, new_z), new_step_count = GLA_KV_Layer(
                d_model=self.d_model,
                n_heads=self.n_heads,
                d_k=self.d_k,
                d_v=self.d_v,
                name="gla"
            )(norm_inputs, terminations, carry.S, carry.z, carry.step_count)

            new_carry = MoESdLGv2Carry(
                S=new_S,
                z=new_z,
                swa_k=carry.swa_k,
                swa_v=carry.swa_v,
                step_count=new_step_count
            )

        gating_out = GTrXLGRUGate(self.d_model, self.gru_bias, name='gru1')(inputs, attn_out)

        norm_gating_out = RMSNorm(name='ln2')(gating_out)
        ffn_out = SwiGLU(self.d_model, self.d_ffc, name='ffn')(norm_gating_out)

        out = GTrXLGRUGate(self.d_model, self.gru_bias, name='gru2')(gating_out, ffn_out)

        return out, new_carry

class BatchedMoESdLGv2(nn.Module):
    n_layers: int
    d_model: int
    n_heads: int
    d_k: int
    d_v: int
    d_ffc: int
    window_size: int

    @nn.compact
    def __call__(self, inputs, terminations, carry, deterministic=True):
        if inputs.shape[0] == 1:
            reset_mask = 1.0 - terminations[0]
            carry = jax.tree_util.tree_map(lambda x: (x * reset_mask).astype(x.dtype), carry)

        if inputs.shape[-1] != self.d_model:
            x = nn.Dense(self.d_model, name='emb_layer', kernel_init=orthogonal(jnp.sqrt(2)),
                         bias_init=constant(0.0))(inputs)
            x = jax.nn.elu(x)
        else:
            x = inputs

        new_carry = []
        for layer_idx in range(self.n_layers):
            layer_carry = carry[layer_idx]
            x, updated_layer_carry = MoESdLGv2Block(
                d_model=self.d_model,
                n_heads=self.n_heads,
                d_k=self.d_k,
                d_v=self.d_v,
                d_ffc=self.d_ffc,
                window_size=self.window_size,
                layer_idx=layer_idx,
                name=f'layer_{layer_idx}'
            )(x, terminations, layer_carry, deterministic=deterministic)
            new_carry.append(updated_layer_carry)

        return x, tuple(new_carry)

    @staticmethod
    def initialize_carry(batch_size, n_layers, n_heads, d_k, d_v, window_size):
        carry_list = []
        for _ in range(n_layers):
            S = jnp.zeros((n_heads, d_k, d_v), dtype=jnp.float32)
            z = jnp.zeros((n_heads, d_k), dtype=jnp.float32)
            step_count = jnp.zeros(())
            swa_k = jnp.zeros((window_size, d_k))
            swa_v = jnp.zeros((window_size, d_v))

            carry_list.append(MoESdLGv2Carry(
                S=S, z=z, swa_k=swa_k, swa_v=swa_v, step_count=step_count
            ))
        return tuple(carry_list)

class ActorCriticMoESdLGv2(nn.Module):
    action_dim: int
    config: dict

    @nn.compact
    def __call__(self, carry, x, deterministic=True):
        obs, dones = x

        from src.models.purejaxrl.wrappers import RepresentationNetwork
        emb = RepresentationNetwork(
            d_model=self.config["RNN"]["D_MODEL"], name="repr_network"
        )(obs)

        emb_trans = jnp.transpose(emb, (1, 0, 2))
        dones_trans = jnp.transpose(dones, (1, 0))

        VmappedMoESdLGv2 = nn.vmap(
            BatchedMoESdLGv2,
            in_axes=(0, 0, 0, None),
            out_axes=0,
            variable_axes={'params': None},
            split_rngs={'params': False, 'random': True}
        )

        rnn = VmappedMoESdLGv2(
            n_layers=self.config["RNN"].get("N_LAYERS", 4),
            d_model=self.config["RNN"]["D_MODEL"],
            n_heads=self.config["RNN"].get("N_HEADS", 8),
            d_k=self.config["RNN"].get("D_K", 32),
            d_v=self.config["RNN"].get("D_V", 32),
            d_ffc=self.config["RNN"].get("D_FFC", 1024),
            window_size=self.config["RNN"].get("WINDOW_SIZE", 8),
        )

        h_t_trans, new_carry = rnn(emb_trans, dones_trans, carry, deterministic)

        # Transpose back to (T, B, D)
        h_t = jnp.transpose(h_t_trans, (1, 0, 2))

        # Actor MoE
        actor_moe_out, actor_l_bal, actor_exp_w = SparseMoESwiGLU(
            d_model=self.config["RNN"]["D_MODEL"],
            d_ffc=self.config["RNN"]["D_FFC"],
            num_experts=4, top_k=2, name="actor_moe"
        )(h_t)

        # Critic MoE
        critic_moe_out, critic_l_bal, critic_exp_w = SparseMoESwiGLU(
            d_model=self.config["RNN"]["D_MODEL"],
            d_ffc=self.config["RNN"]["D_FFC"],
            num_experts=4, top_k=2, name="critic_moe"
        )(h_t)

        action_logits = nn.Dense(
            self.action_dim, kernel_init=flax.linen.initializers.orthogonal(0.01), name="actor_head"
        )(actor_moe_out)
        action_logits = 5.0 * jnp.tanh(action_logits)

        critic_values = nn.Dense(
            1, kernel_init=flax.linen.initializers.orthogonal(1.0), name="critic_head"
        )(critic_moe_out)
        critic_values = jnp.squeeze(critic_values, axis=-1)

        return action_logits, critic_values, (actor_l_bal, critic_l_bal), (actor_exp_w, critic_exp_w), new_carry
