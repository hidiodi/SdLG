import functools
from collections.abc import Sequence

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal
from flax import struct

# --- TRANSITION TUPLE ---
@struct.dataclass
class Transition:
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: dict

# --- HYBRID SEQUENCE BLOCK (LAYERSCALE OPTIMIZED) ---
class HybridBlock(nn.Module):
    embed_dim: int
    num_heads: int
    head_dim_k: int
    head_dim_v: int
    window_size: int
    approx_rank: int = 1
    use_swa: bool = True

    def setup(self):
        # SWA Projections
        if self.use_swa:
            self.q_proj = nn.Dense(self.num_heads * self.head_dim_k, use_bias=False)
            self.k_proj = nn.Dense(self.num_heads * self.head_dim_k, use_bias=False)
            self.v_proj = nn.Dense(self.num_heads * self.head_dim_v, use_bias=False)
            self.o_proj = nn.Dense(self.embed_dim, use_bias=False)

        # GLA Projections
        self.gla_q_proj = nn.Dense(self.num_heads * self.head_dim_k, use_bias=False)
        self.gla_k_proj = nn.Dense(self.num_heads * self.head_dim_k, use_bias=False)
        self.gla_v_proj = nn.Dense(self.num_heads * self.head_dim_v, use_bias=False)
        self.gla_g_proj = nn.Dense(self.num_heads * self.head_dim_v, use_bias=True) # Gate
        self.gla_o_proj = nn.Dense(self.embed_dim, use_bias=False)

        # Data-Dependent Gated Decay Projection (Phase 2)
        self.gla_beta_proj = nn.Dense(self.num_heads * self.head_dim_v, bias_init=constant(-5.0))
        self.gla_gamma_proj = nn.Dense(self.num_heads * self.head_dim_k, bias_init=constant(-5.0))

        # FiLM Projection Layer (Outputs gamma & beta)
        self.film_proj = nn.Dense(
            2 * self.embed_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0)
        )

        # LayerScale Vectors (Phase 4 Optimization)
        # Initializing scales to 1e-4 forces the layer to act as an identity block at step 0
        self.scale_actor = self.param('scale_actor', jax.nn.initializers.constant(1.0), (self.embed_dim,))
        self.scale_critic = self.param('scale_critic', jax.nn.initializers.constant(1.0), (self.embed_dim,))

        self.ln_actor = nn.LayerNorm()
        self.ln_critic = nn.LayerNorm()

    def _swa_attention(self, q, k_cache, v_cache, valid_cache):
        scores = jnp.einsum("bhd,bhwd->bhw", q, k_cache) / np.sqrt(self.head_dim_k + 1e-8)
        scores = jnp.where(valid_cache, scores, -1e9)
        attn = jax.nn.softmax(scores, axis=-1)
        out = jnp.einsum("bhw,bhwd->bhd", attn, v_cache)
        out = out.reshape((q.shape[0], -1))
        return self.o_proj(out)

    def _gla_step(self, x, state, time_step):
        q = self.gla_q_proj(x).reshape((x.shape[0], self.num_heads, self.head_dim_k))
        k = self.gla_k_proj(x).reshape((x.shape[0], self.num_heads, self.head_dim_k))
        v = self.gla_v_proj(x).reshape((x.shape[0], self.num_heads, self.head_dim_v))
        g = self.gla_g_proj(x).reshape((x.shape[0], self.num_heads, self.head_dim_v))
        q = jax.nn.elu(q) + 1.0
        k = jax.nn.elu(k) + 1.0

        # Phase 2: Data-Dependent Gated Decay
        beta_t = jax.nn.sigmoid(self.gla_beta_proj(x)).reshape((x.shape[0], self.num_heads, self.head_dim_v))
        gamma_t = jax.nn.sigmoid(self.gla_gamma_proj(x)).reshape((x.shape[0], self.num_heads, self.head_dim_k))

        # Phase 3: Low-Rank Vector Recurrence (AGaLiTe)
        v_state, k_state, s_state = state
        # v_state: (batch, approx_rank, num_heads, head_dim_v)
        # k_state: (batch, approx_rank, num_heads, head_dim_k)
        # s_state: (batch, num_heads, head_dim_k)

        k_indices = jnp.arange(self.approx_rank)
        w_k = 2 * jnp.pi * k_indices / self.approx_rank
        # time_step is (batch,). We need (batch, approx_rank)
        cos_wt = jnp.cos(time_step[:, None] * w_k[None, :])[:, :, None, None] # (batch, approx_rank, 1, 1)

        v_t_mod = beta_t * v
        k_t_mod = gamma_t * k

        new_v_state = (1 - beta_t[:, None, :, :]) * v_state + cos_wt * v_t_mod[:, None, :, :]
        new_k_state = (1 - gamma_t[:, None, :, :]) * k_state + cos_wt * k_t_mod[:, None, :, :]
        new_s_state = (1 - gamma_t) * s_state + k_t_mod

        # Memory-Free Retrieval
        # a_t = sum_k ( v_k * (k_k^T q) ) / (2r * (s_t^T q))

        # k_k^T q: (batch, approx_rank, num_heads)
        k_q = jnp.einsum("brhd,bhd->brh", new_k_state, q)

        # sum_k ( v_k * (k_k^T q) ): (batch, num_heads, head_dim_v)
        num_v = jnp.einsum("brhd,brh->bhd", new_v_state, k_q)

        # s_t^T q: (batch, num_heads)
        den_q = jnp.maximum(jnp.einsum("bhd,bhd->bh", new_s_state, q), 1e-6) # prevent division by zero

        a_t = num_v / (2 * self.approx_rank * den_q[..., None])

        out = a_t * jax.nn.silu(g)
        out = out.reshape((x.shape[0], -1))

        return self.gla_o_proj(out), (new_v_state, new_k_state, new_s_state)

    def block_forward_eval(self, x, swa_k, swa_v, swa_valid, swa_idx, gla_state, last_done, time_step):
        # 1. Pre-Forward Masking
        done_mask = last_done[:, None, None, None]
        done_mask_state = last_done[:, None, None]

        if self.use_swa:
            swa_k = jnp.where(done_mask, jnp.zeros_like(swa_k), swa_k)
            swa_v = jnp.where(done_mask, jnp.zeros_like(swa_v), swa_v)
            swa_valid = jnp.where(last_done[:, None, None], jnp.zeros_like(swa_valid), swa_valid)

        v_state, k_state, s_state = gla_state
        v_state = jnp.where(done_mask, jnp.zeros_like(v_state), v_state)
        k_state = jnp.where(done_mask, jnp.zeros_like(k_state), k_state)
        s_state = jnp.where(done_mask_state, jnp.zeros_like(s_state), s_state)
        gla_state = (v_state, k_state, s_state)

        if self.use_swa:
            q_t = self.q_proj(x).reshape((x.shape[0], self.num_heads, self.head_dim_k))
            k_t = self.k_proj(x).reshape((x.shape[0], self.num_heads, self.head_dim_k))
            v_t = self.v_proj(x).reshape((x.shape[0], self.num_heads, self.head_dim_v))

            new_swa_k = swa_k.at[:, :, swa_idx].set(k_t)
            new_swa_v = swa_v.at[:, :, swa_idx].set(v_t)
            new_swa_valid = swa_valid.at[:, :, swa_idx].set(True)

            swa_out = self._swa_attention(q_t, new_swa_k, new_swa_v, new_swa_valid)
        else:
            new_swa_k, new_swa_v, new_swa_valid = swa_k, swa_v, swa_valid
            swa_out = jnp.zeros_like(x)

        gla_out, new_gla_state = self._gla_step(x, gla_state, time_step)
        gla_out_actor = jax.lax.stop_gradient(gla_out)

        if self.use_swa:
            # Latent Goal FiLM Gating
            film_params = self.film_proj(gla_out_actor)
            gamma, beta = jnp.split(film_params, 2, axis=-1)
            gamma = gamma + 1.0
            swa_out_conditioned = swa_out * gamma + beta

            # LayerScale Gating
            actor_features = self.ln_actor(x + self.scale_actor * (swa_out_conditioned + gla_out_actor))
            critic_features = self.ln_critic(x + self.scale_critic * (swa_out_conditioned + gla_out))
        else:
            actor_features = self.ln_actor(x + self.scale_actor * gla_out_actor)
            critic_features = self.ln_critic(x + self.scale_critic * gla_out)

        # --- FIX: Ensure new_swa_valid is included in the returned tuple (6 values) ---
        return actor_features, critic_features, new_swa_k, new_swa_v, new_swa_valid, new_gla_state


    def block_forward_train(self, x, swa_mask, init_gla_state, last_done_mask, time_steps_batch):
        if self.use_swa:
            # Parallel SWA
            q_swa = self.q_proj(x).reshape((x.shape[0], x.shape[1], self.num_heads, self.head_dim_k))
            k_swa = self.k_proj(x).reshape((x.shape[0], x.shape[1], self.num_heads, self.head_dim_k))
            v_swa = self.v_proj(x).reshape((x.shape[0], x.shape[1], self.num_heads, self.head_dim_v))

            scores = jnp.einsum("bthd,bshd->bhts", q_swa, k_swa) / np.sqrt(self.head_dim_k + 1e-8)
            scores = jnp.where(swa_mask[:, None, :, :], scores, -1e9)
            attn = jax.nn.softmax(scores, axis=-1)

            swa_out = jnp.einsum("bhts,bshd->bthd", attn, v_swa)
            swa_out = swa_out.reshape((x.shape[0], x.shape[1], -1))
            swa_out = self.o_proj(swa_out)
        else:
            swa_out = jnp.zeros_like(x)

        # Parallel GLA
        q = self.gla_q_proj(x).reshape((x.shape[0], x.shape[1], self.num_heads, self.head_dim_k))
        k = self.gla_k_proj(x).reshape((x.shape[0], x.shape[1], self.num_heads, self.head_dim_k))
        v = self.gla_v_proj(x).reshape((x.shape[0], x.shape[1], self.num_heads, self.head_dim_v))
        g = self.gla_g_proj(x).reshape((x.shape[0], x.shape[1], self.num_heads, self.head_dim_v))
        q = jax.nn.elu(q) + 1.0
        k = jax.nn.elu(k) + 1.0

        # Phase 2: Element-Wise Gated Decay
        beta_t = jax.nn.sigmoid(self.gla_beta_proj(x)).reshape((x.shape[0], x.shape[1], self.num_heads, self.head_dim_v))
        gamma_t = jax.nn.sigmoid(self.gla_gamma_proj(x)).reshape((x.shape[0], x.shape[1], self.num_heads, self.head_dim_k))

        k_indices = jnp.arange(self.approx_rank)
        w_k = 2 * jnp.pi * k_indices / self.approx_rank
        # cos_wt: time_steps_batch is (batch, seq_len)
        # w_k is (approx_rank,)
        # cos_wt -> (batch, seq_len, approx_rank) -> (batch, seq_len, approx_rank, 1, 1)
        cos_wt = jnp.cos(time_steps_batch[..., None] * w_k[None, None, :])
        cos_wt = cos_wt[:, :, :, None, None]

        v_t_mod = beta_t * v
        k_t_mod = gamma_t * k

        reset_mask = (1.0 - last_done_mask[:, :, None, None, None])
        reset_mask_s = (1.0 - last_done_mask[:, :, None, None])

        A_t_v = (1 - beta_t[:, :, None, :, :]) * reset_mask # (batch, seq, 1, heads, dim) * mask
        A_t_v = jnp.tile(A_t_v, (1, 1, self.approx_rank, 1, 1))
        B_t_v = cos_wt * v_t_mod[:, :, None, :, :]

        A_t_k = (1 - gamma_t[:, :, None, :, :]) * reset_mask
        A_t_k = jnp.tile(A_t_k, (1, 1, self.approx_rank, 1, 1))
        B_t_k = cos_wt * k_t_mod[:, :, None, :, :]

        A_t_s = (1 - gamma_t) * reset_mask_s
        B_t_s = k_t_mod

        def _scan_op(carry, x_scan):
            A1, B1 = carry
            A2, B2 = x_scan
            return (A2 * A1, A2 * B1 + B2)

        A_t_v_scan = jnp.swapaxes(A_t_v, 0, 1)
        B_t_v_scan = jnp.swapaxes(B_t_v, 0, 1)
        A_cumprod_v, V_t_scan = jax.lax.associative_scan(_scan_op, (A_t_v_scan, B_t_v_scan))
        V_t_scan = V_t_scan + A_cumprod_v * init_gla_state[0][None, ...]
        V_t = jnp.swapaxes(V_t_scan, 0, 1)

        A_t_k_scan = jnp.swapaxes(A_t_k, 0, 1)
        B_t_k_scan = jnp.swapaxes(B_t_k, 0, 1)
        A_cumprod_k, K_t_scan = jax.lax.associative_scan(_scan_op, (A_t_k_scan, B_t_k_scan))
        K_t_scan = K_t_scan + A_cumprod_k * init_gla_state[1][None, ...]
        K_t = jnp.swapaxes(K_t_scan, 0, 1)

        A_t_s_scan = jnp.swapaxes(A_t_s, 0, 1)
        B_t_s_scan = jnp.swapaxes(B_t_s, 0, 1)
        A_cumprod_s, S_t_scan = jax.lax.associative_scan(_scan_op, (A_t_s_scan, B_t_s_scan))
        S_t_scan = S_t_scan + A_cumprod_s * init_gla_state[2][None, ...]
        S_t = jnp.swapaxes(S_t_scan, 0, 1)

        # k_k^T q
        k_q = jnp.einsum("btrhd,bthd->btrh", K_t, q)

        # sum_k ( v_k * (k_k^T q) )
        num_v = jnp.einsum("btrhd,btrh->bthd", V_t, k_q)

        # s_t^T q
        den_q = jnp.maximum(jnp.einsum("bthd,bthd->bth", S_t, q), 1e-6)

        a_t = num_v / (2 * self.approx_rank * den_q[..., None])

        gla_out = a_t * jax.nn.silu(g)
        gla_out = gla_out.reshape((x.shape[0], x.shape[1], -1))
        gla_out = self.gla_o_proj(gla_out)

        # --- FIX: Stop actor gradient to protect memory weights, but pass representation to both ---
        gla_out_actor = jax.lax.stop_gradient(gla_out)

        if self.use_swa:
            # Latent Goal FiLM Gating
            film_params = self.film_proj(gla_out_actor)
            gamma_film, beta_film = jnp.split(film_params, 2, axis=-1)
            gamma_film = gamma_film + 1.0
            swa_out_conditioned = swa_out * gamma_film + beta_film

            # LayerScale Gating: Actor receives SWA + GLA (protected), Critic receives SWA + GLA (trainable)
            actor_features = self.ln_actor(x + self.scale_actor * (swa_out_conditioned + gla_out_actor))
            critic_features = self.ln_critic(x + self.scale_critic * (swa_out_conditioned + gla_out))
        else:
            actor_features = self.ln_actor(x + self.scale_actor * gla_out_actor)
            critic_features = self.ln_critic(x + self.scale_critic * gla_out)

        return actor_features, critic_features

# --- MULTI-LAYER HYBRID AC ---
class ActorCriticHybrid(nn.Module):
    action_dim: int
    activation: str
    hidden_layers: int
    embed_dim: int
    num_heads: int
    head_dim_k: int
    head_dim_v: int
    window_size: int
    num_layers: int = 2
    approx_rank: int = 1
    gla_decay: float = 0.99
    use_swa: bool = True

    def setup(self):
        if self.activation == "relu":
            jax.nn.silu = nn.relu
        else:
            jax.nn.silu = nn.tanh

        # Basic embeddings
        self.embed = nn.Dense(
            self.embed_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )

        # Stacked sequence layers
        self.layers = [
            HybridBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                head_dim_k=self.head_dim_k,
                head_dim_v=self.head_dim_v,
                window_size=self.window_size,

                approx_rank=self.approx_rank,
                use_swa=self.use_swa,
                name=f"layer_{i}"
            )
            for i in range(self.num_layers)
        ]

        # Actor heads
        self.actor_ln1 = nn.Dense(
            self.hidden_layers, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )
        self.actor_ln2 = nn.Dense(
            self.hidden_layers, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )
        self.actor_out = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )

        # Feasibility Head
        self.feasibility_ln1 = nn.Dense(
            self.hidden_layers, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )
        self.feasibility_out = nn.Dense(
            self.action_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0)
        )

        # Critic heads
        self.critic_ln1 = nn.Dense(
            self.hidden_layers, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )
        self.critic_ln2 = nn.Dense(
            self.hidden_layers, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )
        self.critic_out = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))

    def model_forward_eval(self, swa_k_cache, swa_v_cache, swa_valid_cache, swa_idx, gla_state, obs, last_done, action_mask, masking_type):
        """Sequential rollout timestep."""
        x = jax.nn.silu(self.embed(obs))

        new_swa_k_list = []
        new_swa_v_list = []
        new_swa_valid_list = []
        new_gla_state_v_list = []
        new_gla_state_k_list = []
        new_gla_state_s_list = []

        h_act = x
        h_crit = x

        v_state_all, k_state_all, s_state_all, t_state = gla_state

        # t_state is shape (num_envs,)
        # It needs to reset if last_done is true
        current_t = jnp.where(last_done, 0, t_state)
        new_t_state = current_t + 1

        for l in range(self.num_layers):
            layer_gla_state = (v_state_all[l], k_state_all[l], s_state_all[l])
            h_act_new, h_crit_new, n_k, n_v, n_valid, n_gla = self.layers[l].block_forward_eval(
                h_act, swa_k_cache[l], swa_v_cache[l], swa_valid_cache[l], swa_idx, layer_gla_state, last_done, current_t
            )
            h_act = h_act_new
            h_crit = h_crit + h_crit_new

            new_swa_k_list.append(n_k)
            new_swa_v_list.append(n_v)
            new_swa_valid_list.append(n_valid)
            new_gla_state_v_list.append(n_gla[0])
            new_gla_state_k_list.append(n_gla[1])
            new_gla_state_s_list.append(n_gla[2])

        new_swa_k_cache = jnp.stack(new_swa_k_list, axis=0)
        new_swa_v_cache = jnp.stack(new_swa_v_list, axis=0)
        new_swa_valid_cache = jnp.stack(new_swa_valid_list, axis=0)

        new_gla_state_v = jnp.stack(new_gla_state_v_list, axis=0)
        new_gla_state_k = jnp.stack(new_gla_state_k_list, axis=0)
        new_gla_state_s = jnp.stack(new_gla_state_s_list, axis=0)
        new_gla_state = (new_gla_state_v, new_gla_state_k, new_gla_state_s, new_t_state)

        # --- Policy Head ---
        actor_mean = self.actor_ln1(h_act)
        actor_mean = jax.nn.silu(actor_mean)
        actor_mean = self.actor_ln2(actor_mean)
        actor_mean = jax.nn.silu(actor_mean)
        actor_mean = self.actor_out(actor_mean)
        # Clip logits to prevent softmax probabilities from exactly reaching 0.0 (which causes NaN entropy)
        actor_mean = 5.0 * jax.nn.tanh(actor_mean)

        if masking_type == "oracle":
            active_mask = action_mask
        elif masking_type == "learned":
            feas_logits = self.feasibility_out(jax.nn.silu(self.feasibility_ln1(h_act)))
            feas_probs = jax.nn.sigmoid(feas_logits)
            active_mask = (feas_probs > 0.5) | (jnp.arange(self.action_dim) == 0)
        else:
            active_mask = jnp.ones_like(action_mask)

        masked_actor_mean = jnp.where(active_mask, actor_mean, -1e9)
        pi = distrax.Categorical(logits=masked_actor_mean)

        # --- Value Head ---
        critic = self.critic_ln1(h_crit)
        critic = jax.nn.silu(critic)
        critic = self.critic_ln2(critic)
        critic = jax.nn.silu(critic)
        critic = self.critic_out(critic)

        return pi, jnp.squeeze(critic, axis=-1), new_swa_k_cache, new_swa_v_cache, new_swa_valid_cache, new_gla_state

    def model_forward_train(self, obs, swa_mask, init_gla_state, last_done_mask, action_mask, masking_type):
        """Parallel training over sequence T."""
        x = jax.nn.silu(self.embed(obs))

        h_act = x
        h_crit = x

        v_state_all, k_state_all, s_state_all, t_state = init_gla_state

        # Calculate absolute time steps incorporating resets
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        # t_state is (batch,) initial time
        def _time_scan(carry, is_done):
            # if previous step was done, time is 0. Else time is carry
            t = jnp.where(is_done, 0, carry)
            next_t = t + 1
            return next_t, t

        _, time_steps_batch = jax.lax.scan(_time_scan, t_state, jnp.swapaxes(last_done_mask, 0, 1))
        time_steps_batch = jnp.swapaxes(time_steps_batch, 0, 1) # (batch, seq_len)

        for l in range(self.num_layers):
            layer_gla_state = (v_state_all[l], k_state_all[l], s_state_all[l])
            h_act_new, h_crit_new = self.layers[l].block_forward_train(
                h_act, swa_mask, layer_gla_state, last_done_mask, time_steps_batch
            )
            h_act = h_act_new
            h_crit = h_crit + h_crit_new

        # --- Policy Head ---
        actor_mean = self.actor_ln1(h_act)
        actor_mean = jax.nn.silu(actor_mean)
        actor_mean = self.actor_ln2(actor_mean)
        actor_mean = jax.nn.silu(actor_mean)
        actor_mean = self.actor_out(actor_mean)
        # Clip logits to prevent softmax probabilities from exactly reaching 0.0 (which causes NaN entropy)
        actor_mean = 5.0 * jax.nn.tanh(actor_mean)

        feasibility_logits = jnp.zeros_like(actor_mean)
        if masking_type == "oracle":
            active_mask = action_mask
        elif masking_type == "learned":
            feasibility_logits = self.feasibility_out(jax.nn.silu(self.feasibility_ln1(h_act)))
            feas_probs = jax.nn.sigmoid(feasibility_logits)
            active_mask = (feas_probs > 0.5) | (jnp.arange(self.action_dim) == 0)
        else:
            active_mask = jnp.ones_like(action_mask)

        masked_actor_mean = jnp.where(active_mask, actor_mean, -1e9)
        pi = distrax.Categorical(logits=masked_actor_mean)

        # --- Value Head ---
        critic = self.critic_ln1(h_crit)
        critic = jax.nn.silu(critic)
        critic = self.critic_ln2(critic)
        critic = jax.nn.silu(critic)
        critic = self.critic_out(critic)

        return pi, jnp.squeeze(critic, axis=-1), feasibility_logits
class BatchedHybrid(nn.Module):
    embed_dim: int
    num_heads: int
    head_dim_k: int
    head_dim_v: int
    window_size: int
    num_layers: int
    approx_rank: int = 1
    use_swa: bool = True
    action_dim: int = 1

    @nn.compact
    def __call__(self, carry, x):
        swa_k_cache, swa_v_cache, swa_valid_cache, gla_state = carry
        ins, resets = x

        ins = jnp.transpose(ins, (1, 0, 2))

        # we will extract embeddings from the inputs
        # The activation_fn cannot be passed to vmap cleanly inside Flax dataclasses
        # so we will use jax.nn.silu directly inside the model wrapper instead
        batch_hybrid = nn.vmap(ActorCriticHybrid,
            in_axes=0, out_axes=0,
            variable_axes={'params': None},
            split_rngs={'params': False})(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                head_dim_k=self.head_dim_k,
                head_dim_v=self.head_dim_v,
                window_size=self.window_size,

                num_layers=self.num_layers,
                approx_rank=self.approx_rank,
                use_swa=self.use_swa,
                action_dim=self.action_dim, activation="silu", hidden_layers=256
            )

        return carry, ins

    @staticmethod
    def initialize_carry(batch_size, num_layers, num_heads, head_dim_k, head_dim_v, approx_rank, window_size):
        swa_k_cache = jnp.zeros((batch_size, num_layers, window_size, num_heads, head_dim_k))
        swa_v_cache = jnp.zeros((batch_size, num_layers, window_size, num_heads, head_dim_v))
        swa_valid_cache = jnp.zeros((batch_size, num_layers, window_size), dtype=jnp.bool_)

        v_state = jnp.zeros((batch_size, num_layers, approx_rank, num_heads, head_dim_v))
        k_state = jnp.zeros((batch_size, num_layers, approx_rank, num_heads, head_dim_k))
        s_state = jnp.zeros((batch_size, num_layers, approx_rank, num_heads, head_dim_k))
        t_state = jnp.zeros((batch_size,))
        gla_state = (v_state, k_state, s_state, t_state)

        return (swa_k_cache, swa_v_cache, swa_valid_cache, gla_state)
