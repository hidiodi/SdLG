# scripts/diagnostics/exp18/exact_sdlg_tmaze_ablations.py

import sys
import os
import time
import json
import csv
import functools
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import wandb
import distrax
import gc
from flax import struct
from typing import Any, Tuple, Callable, NamedTuple
from flax.linen.initializers import orthogonal, zeros, constant
# Path fixing
current_file_path = os.path.abspath(__file__)
project_root = current_file_path
while not os.path.exists(os.path.join(project_root, "src")):
    project_root = os.path.dirname(project_root)
    if project_root == "/": break
if project_root not in sys.path: sys.path.append(project_root)

import math
from src.models.baselines.mamba import MambaCell
import argparse

from src.envs.tmaze.jax_tmaze import PassiveTMaze
from src.trainers.purejax_trainer import make_update
from gymnax.wrappers.purerl import LogWrapper

# =====================================================================
# --- FACTORIAL ABLATION CONFIGURATIONS (Rank vs RoPE) ---
# =====================================================================
ABLATIONS = {
    "Exact_FastRoPE": {"IS_BASELINE": False, "USE_EXACT": True, "GAMMA_BIAS": "log", "ROPE_VARIANT": "fast_head"},
}

MODELS_TO_RUN = list(ABLATIONS.keys())
TRAINING_SEEDS = [42, 54, 99]
COMPLETED_RUNS_PATH = os.path.join(os.path.dirname(current_file_path), "completed_tmaze_factorial.json")


def load_completed_runs():
    if os.path.exists(COMPLETED_RUNS_PATH):
        try:
            with open(COMPLETED_RUNS_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_completed_run(run_id):
    completed = load_completed_runs()
    if run_id not in completed: completed.append(run_id)
    with open(COMPLETED_RUNS_PATH, "w") as f: json.dump(completed, f)


# =====================================================================
# --- BASELINE MODULES ---
# =====================================================================
class GRU_Module(nn.Module):
    config: dict
    @nn.compact
    def __call__(self, x, carry):
        hidden_dim = self.config.get("RNN", {}).get("HIDDEN_DIM", 128)
        return nn.GRUCell(features=hidden_dim)(carry, x)
    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # FIX: Pass a dummy input shape that Flax expects
        return nn.GRUCell(features=hidden_size).initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))

class LSTM_Module(nn.Module):
    config: dict
    @nn.compact
    def __call__(self, x, carry):
        hidden_dim = self.config.get("RNN", {}).get("HIDDEN_DIM", 128)
        return nn.LSTMCell(features=hidden_dim)(carry, x)
    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # FIX: Pass a dummy input shape that Flax expects
        return nn.LSTMCell(features=hidden_size).initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))

class Markovian_Module(nn.Module):
    config: dict
    @nn.compact
    def __call__(self, x, carry):
        hidden_dim = self.config.get("RNN", {}).get("HIDDEN_DIM", 128)
        out = jax.nn.relu(nn.Dense(features=hidden_dim)(x))
        out = jax.nn.relu(nn.Dense(features=hidden_dim)(out))
        return carry, out
    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        return jnp.zeros((batch_size, hidden_size))

class LTI_Linear_Module(nn.Module):
    config: dict
    @nn.compact
    def __call__(self, x, carry):
        hidden_dim = self.config.get("RNN", {}).get("HIDDEN_DIM", 128)
        A = self.param('A', nn.initializers.normal(stddev=0.1), (hidden_dim,))
        B = self.param('B', nn.initializers.normal(stddev=0.1), (x.shape[-1], hidden_dim))
        C = self.param('C', nn.initializers.normal(stddev=0.1), (hidden_dim, hidden_dim))
        D = self.param('D', nn.initializers.normal(stddev=0.1), (x.shape[-1], hidden_dim))
        A_stable = jax.nn.sigmoid(A)
        new_carry = A_stable * carry + jnp.dot(x, B)
        out = jax.nn.gelu(jnp.dot(new_carry, C) + jnp.dot(x, D))
        return new_carry, out
    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        return jnp.zeros((batch_size, hidden_size))

# =====================================================================
# --- EXACT SDLG MODULES ---
# =====================================================================
class ExactSdLGCarry(NamedTuple):
    M: jnp.ndarray
    z: jnp.ndarray
    swa_k: jnp.ndarray
    swa_v: jnp.ndarray
    step_count: jnp.ndarray
    rng: jnp.ndarray
    time_scale: jnp.ndarray


class FastExactMultiScaleGLA(nn.Module):
    d_model: int;
    n_heads: int;
    d_k: int;
    d_v: int;
    gamma_bias_type: str
    rope_variant: str = "none"

    def setup(self):
        if self.gamma_bias_type == "log":
            base_timescales = jnp.array([16.0, 64.0, 256.0, 1024.0, 4096.0, 16384.0])
            pad_size = max(0, self.n_heads - len(base_timescales))
            if pad_size > 0:
                extra = base_timescales[-1] * (4.0 ** jnp.arange(1, pad_size + 1))
                timescales = jnp.concatenate([base_timescales, extra])
            else:
                timescales = base_timescales[:self.n_heads]
            lambda_h = jnp.log(2.0) / timescales
            self.gamma_bias = jnp.log(jnp.exp(lambda_h) - 1.0)
        else:
            self.gamma_bias = float(self.gamma_bias_type)

        self.q_proj = nn.Dense(self.n_heads * self.d_k, use_bias=False)
        self.k_proj = nn.Dense(self.n_heads * self.d_k, use_bias=False)
        self.v_proj = nn.Dense(self.n_heads * self.d_v, use_bias=False)
        self.out_proj = nn.Dense(self.d_model, use_bias=False)
        self.delta_lambda_proj = nn.Dense(self.n_heads, use_bias=False, kernel_init=zeros)
        self.beta_proj = nn.Dense(self.n_heads, use_bias=False, kernel_init=zeros)

    def __call__(self, inputs, terminations, carry, step_counts=None):
        T, _ = inputs.shape
        M_prev, z_prev = carry
        q = self.q_proj(inputs).reshape((T, self.n_heads, self.d_k))
        k = self.k_proj(inputs).reshape((T, self.n_heads, self.d_k))
        v = self.v_proj(inputs).reshape((T, self.n_heads, self.d_v))

        if self.rope_variant != "none" and step_counts is not None:
            # step_counts could be scaled if time_scale is passed. We pass (step_counts, time_scale) as step_counts if time_warp
            if isinstance(step_counts, tuple):
                step_counts, time_scale = step_counts
            else:
                time_scale = 1.0

            if self.rope_variant == "fast_head" or self.rope_variant == "time_warp":
                from src.models.common.sdlg_hybrid import rope_rotate
                omegas = jnp.linspace(-jnp.pi, jnp.pi, self.n_heads)
                theta = (step_counts[..., None] * omegas * time_scale) % (2 * jnp.pi)
                theta = theta[..., None] # Shape: (T, n_heads, 1)
                q_rot = rope_rotate(q[..., None, :], theta).squeeze(-2)
                k_rot = rope_rotate(k[..., None, :], theta).squeeze(-2)
                phi_q, phi_k = jax.nn.elu(q_rot) + 1.0, jax.nn.elu(k_rot) + 1.0
            elif self.rope_variant == "slow_head":
                from src.models.common.sdlg_hybrid import rope_rotate
                omegas = 10000.0 ** (-jnp.arange(self.n_heads) / self.n_heads)
                theta = (step_counts[..., None] * omegas) % (2 * jnp.pi)
                theta = theta[..., None]
                q_rot = rope_rotate(q[..., None, :], theta).squeeze(-2)
                k_rot = rope_rotate(k[..., None, :], theta).squeeze(-2)
                phi_q, phi_k = jax.nn.elu(q_rot) + 1.0, jax.nn.elu(k_rot) + 1.0
            elif self.rope_variant == "scaled_fast_head":
                from src.models.common.sdlg_hybrid import rope_rotate
                omegas = jnp.linspace(-jnp.pi, jnp.pi, self.n_heads) * 0.25
                theta = (step_counts[..., None] * omegas) % (2 * jnp.pi)
                theta = theta[..., None]
                q_rot = rope_rotate(q[..., None, :], theta).squeeze(-2)
                k_rot = rope_rotate(k[..., None, :], theta).squeeze(-2)
                phi_q, phi_k = jax.nn.elu(q_rot) + 1.0, jax.nn.elu(k_rot) + 1.0
            elif self.rope_variant == "standard_dim":
                # Standard RoPE within each head across d_k dimensions
                # theta applied independently to dimension pairs
                d_k = self.d_k
                omegas = 10000.0 ** (-2 * jnp.arange(d_k // 2) / d_k)
                theta = (step_counts[..., None, None] * omegas) % (2 * jnp.pi) # (T, 1, d_k//2)
                
                # Reshape q, k to group pairs: (T, n_heads, d_k//2, 2)
                q_paired = q.reshape(T, self.n_heads, d_k // 2, 2)
                k_paired = k.reshape(T, self.n_heads, d_k // 2, 2)
                
                cos_t = jnp.cos(theta) # (T, 1, d_k//2)
                sin_t = jnp.sin(theta)
                
                # Apply rotation
                q_rot_0 = q_paired[..., 0] * cos_t - q_paired[..., 1] * sin_t
                q_rot_1 = q_paired[..., 1] * cos_t + q_paired[..., 0] * sin_t
                q_rot = jnp.stack([q_rot_0, q_rot_1], axis=-1).reshape(T, self.n_heads, d_k)
                
                k_rot_0 = k_paired[..., 0] * cos_t - k_paired[..., 1] * sin_t
                k_rot_1 = k_paired[..., 1] * cos_t + k_paired[..., 0] * sin_t
                k_rot = jnp.stack([k_rot_0, k_rot_1], axis=-1).reshape(T, self.n_heads, d_k)
                
                phi_q, phi_k = jax.nn.elu(q_rot) + 1.0, jax.nn.elu(k_rot) + 1.0
            else:
                phi_q, phi_k = jax.nn.elu(q) + 1.0, jax.nn.elu(k) + 1.0
        else:
            phi_q, phi_k = jax.nn.elu(q) + 1.0, jax.nn.elu(k) + 1.0
        terminations = terminations.astype(jnp.float32)

        gamma = jnp.exp(-jax.nn.softplus(self.delta_lambda_proj(inputs) + self.gamma_bias))
        beta = jax.nn.sigmoid(self.beta_proj(inputs))

        write_M = beta[..., None, None] * (phi_k[..., None] * v[..., None, :])
        write_z = beta[..., None] * phi_k

        discount_gamma = gamma * (1.0 - terminations[:, None])
        discount_M, discount_z = discount_gamma[..., None, None], discount_gamma[..., None]

        def combine(a, b): return b[0] + b[1] * a[0], a[1] * b[1]

        write_M = write_M.at[0].set(write_M[0] + discount_M[0] * M_prev)
        write_z = write_z.at[0].set(write_z[0] + discount_z[0] * z_prev)

        all_M, _ = jax.lax.associative_scan(combine, (write_M, discount_M), axis=0)
        all_z, _ = jax.lax.associative_scan(combine, (write_z, discount_z), axis=0)
        num = jnp.matmul(jnp.expand_dims(phi_q, -2), all_M).squeeze(-2)
        den = jnp.sum(phi_q * all_z, axis=-1, keepdims=True) + 1e-6
        out = self.out_proj((num / den).reshape((T, self.n_heads * self.d_v)))

        return jnp.nan_to_num(out, nan=0.0, posinf=10.0, neginf=-10.0), (all_M[-1], all_z[-1])


class FastExactSdLGHybridBlock(nn.Module):
    d_model: int;
    n_heads: int;
    d_k: int;
    d_v: int;
    d_ffc: int;
    window_size: int;
    gamma_bias_type: str;
    rope_variant: str = "none"
    gru_bias: float = 2.0

    @nn.compact
    def __call__(self, inputs, terminations, carry: ExactSdLGCarry, deterministic=True):
        from src.models.common.sdlg_hybrid import GatedSlidingWindowAttention, GTrXLGRUGate
        T = inputs.shape[0]

        norm_swa = nn.LayerNorm(epsilon=1e-5)(inputs)
        swa_out, new_swa_k, new_swa_v = GatedSlidingWindowAttention(d_model=self.d_model, d_k=self.d_k, d_v=self.d_v,
                                                                    window_size=self.window_size)(norm_swa,
                                                                                                  terminations,
                                                                                                  carry.swa_k,
                                                                                                  carry.swa_v,
                                                                                                  carry.step_count,
                                                                                                  deterministic=deterministic)
        x_swa = GTrXLGRUGate(self.d_model, self.gru_bias)(inputs, swa_out)

        if T > 1:
            def scan_fn(c, t):
                curr_step, curr_scale, curr_rng = c
                rng1, rng2 = jax.random.split(curr_rng)
                new_scale = jnp.where(t, jax.random.uniform(rng1, minval=0.25, maxval=4.0), curr_scale)
                new_step = (curr_step + 1.0) * (1.0 - t)
                return (new_step, new_scale, rng2), (new_step, new_scale)
            
            (new_step_count, new_scale, new_rng), (step_counts, scales) = jax.lax.scan(
                scan_fn, (carry.step_count, carry.time_scale, carry.rng), terminations
            )
        else:
            rng1, rng2 = jax.random.split(carry.rng)
            new_scale = jnp.where(terminations[0], jax.random.uniform(rng1, minval=0.25, maxval=4.0), carry.time_scale)
            new_step_count = (carry.step_count + 1.0) * (1.0 - terminations[0])
            step_counts = jnp.expand_dims(new_step_count, 0)
            scales = jnp.expand_dims(new_scale, 0)
            new_rng = rng2

        norm_gla = nn.LayerNorm(epsilon=1e-5)(x_swa)
        gla_out, (new_M, new_z) = FastExactMultiScaleGLA(d_model=self.d_model, n_heads=self.n_heads, d_k=self.d_k,
                                                         d_v=self.d_v, gamma_bias_type=self.gamma_bias_type,
                                                         rope_variant=self.rope_variant)(norm_gla,
                                                                                              terminations,
                                                                                              (carry.M,
                                                                                               carry.z),
                                                                                              (step_counts, scales[..., None]))
        x_gla = GTrXLGRUGate(self.d_model, self.gru_bias)(x_swa, gla_out)

        norm_ffn = nn.LayerNorm(epsilon=1e-5)(x_gla)
        ffn_out = nn.Sequential([nn.Dense(self.d_ffc, kernel_init=orthogonal(jnp.sqrt(2))), nn.gelu,
                                 nn.Dense(self.d_model, kernel_init=orthogonal(jnp.sqrt(2)))])(norm_ffn)
        out = GTrXLGRUGate(self.d_model, self.gru_bias)(x_gla, ffn_out)
        return out, ExactSdLGCarry(M=new_M, z=new_z, swa_k=new_swa_k, swa_v=new_swa_v, step_count=new_step_count, rng=new_rng, time_scale=new_scale)


class FastGLABlock(nn.Module):
    d_model: int
    n_heads: int
    d_k: int
    d_v: int
    d_ffc: int
    gamma_bias_type: str
    rope_variant: str = "none"
    gru_bias: float = 2.0

    @nn.compact
    def __call__(self, inputs, terminations, carry: ExactSdLGCarry, deterministic=True):
        from src.models.common.sdlg_hybrid import GTrXLGRUGate
        
        norm_gla = nn.LayerNorm(epsilon=1e-5)(inputs)
        gla_out, (new_M, new_z) = FastExactMultiScaleGLA(
            d_model=self.d_model, n_heads=self.n_heads, d_k=self.d_k,
            d_v=self.d_v, gamma_bias_type=self.gamma_bias_type,
            rope_variant=self.rope_variant
        )(norm_gla, terminations, (carry.M, carry.z))
        x_gla = GTrXLGRUGate(self.d_model, self.gru_bias)(inputs, gla_out)

        norm_ffn = nn.LayerNorm(epsilon=1e-5)(x_gla)
        ffn_out = nn.Sequential([
            nn.Dense(self.d_ffc, kernel_init=orthogonal(jnp.sqrt(2))), nn.gelu,
            nn.Dense(self.d_model, kernel_init=orthogonal(jnp.sqrt(2)))
        ])(norm_ffn)
        out = GTrXLGRUGate(self.d_model, self.gru_bias)(x_gla, ffn_out)
        
        # We reuse the same carry tuple to match the interface, but SWA fields stay zeroed
        return out, ExactSdLGCarry(M=new_M, z=new_z, swa_k=carry.swa_k, swa_v=carry.swa_v, step_count=carry.step_count, rng=carry.rng, time_scale=carry.time_scale)


class FastBatchedExactSdLG(nn.Module):
    config: dict;
    ablation_cfg: dict

    @nn.compact
    def __call__(self, inputs, terminations, carry, deterministic=True):
        x = inputs
        new_carry = []
        for i in range(self.config["RNN"]["N_LAYERS"]):
            x, c = FastExactSdLGHybridBlock(d_model=self.config["RNN"]["D_MODEL"],
                                            n_heads=self.config["RNN"]["N_HEADS"],
                                            d_k=self.config["RNN"]["D_MODEL"] // self.config["RNN"]["N_HEADS"],
                                            d_v=self.config["RNN"]["D_MODEL"] // self.config["RNN"]["N_HEADS"],
                                            d_ffc=self.config["RNN"]["D_FFC"], window_size=16,
                                            gamma_bias_type=self.ablation_cfg["GAMMA_BIAS"],
                                            rope_variant=self.ablation_cfg.get("ROPE_VARIANT", "none"))(x, terminations, carry[i],
                                                                                                     deterministic)
            new_carry.append(c)
        return x, tuple(new_carry)


class FastBatchedGLA(nn.Module):
    config: dict
    ablation_cfg: dict

    @nn.compact
    def __call__(self, inputs, terminations, carry, deterministic=True):
        x = inputs
        new_carry = []
        for l in range(self.config["RNN"]["N_LAYERS"]):
            x, next_carry = FastGLABlock(
                d_model=self.config["RNN"]["D_MODEL"],
                n_heads=self.config["RNN"]["N_HEADS"], d_k=32, d_v=32,
                d_ffc=self.config["RNN"]["D_FFC"],
                gamma_bias_type=self.ablation_cfg.get("GAMMA_BIAS", "log"),
                rope_variant=self.ablation_cfg.get("ROPE_VARIANT", "none")
            )(x, terminations, carry[l], deterministic)
            new_carry.append(next_carry)
        return x, tuple(new_carry)


class ActorCriticAblation(nn.Module):
    action_dim: int;
    config: dict

    @nn.compact
    def __call__(self, carry, x):
        obs, dones = x
        x_flat = obs.reshape(obs.shape[0], obs.shape[1], -1)
        emb = nn.relu(nn.Dense(self.config["RNN"]["D_MODEL"], kernel_init=orthogonal(jnp.sqrt(2)))(x_flat))
        emb = nn.relu(nn.Dense(self.config["RNN"]["D_MODEL"], kernel_init=orthogonal(jnp.sqrt(2)))(emb))

        emb_trans, dones_trans = jnp.transpose(emb, (1, 0, 2)), jnp.transpose(dones, (1, 0))
        ablation_cfg = self.config["ABLATION"]

        if ablation_cfg.get("IS_BASELINE", False):
            # --- RUNNING BASELINES OVER TIME ---
            rnn_type = ablation_cfg["RNN_TYPE"]
            cfg = self.config

            if rnn_type == "gtrxl":
                from src.models.common.gtrxl import BatchedGTrXL
                gtrxl = BatchedGTrXL(
                    head_dim=32, embedding_dim=cfg["RNN"]["D_MODEL"], head_num=4,
                    mlp_num=2, layer_num=3, memory_len=ablation_cfg.get("MEMORY_LEN", cfg.get("NUM_STEPS", 16)),
                    dropout_ratio=0.0, gru_gating=True, use_embedding_layer=False
                )
                new_carry, rnn_out = gtrxl(carry, (emb, dones))
            elif rnn_type == "gla":
                VmappedGLA = nn.vmap(FastBatchedGLA, in_axes=(0, 0, 0, None), out_axes=(0, 0),
                                     variable_axes={'params': None}, split_rngs={'params': False})
                rnn_out_trans, new_carry = VmappedGLA(self.config, ablation_cfg)(emb_trans, dones_trans, carry, True)
                rnn_out = jnp.transpose(rnn_out_trans, (1, 0, 2))
            else:
                class SingleStepRNN(nn.Module):
                    @nn.compact
                    def __call__(self, c, inputs):
                        obs_t, done_t = inputs
                        done_mask = done_t[:, None]  # Format for batched multiplication

                        if rnn_type == "lstm":
                            c = jax.tree_util.tree_map(lambda x: x * (1.0 - done_mask), c)
                            c, out = LSTM_Module(cfg)(obs_t, c)
                        elif rnn_type == "gru":
                            c = c * (1.0 - done_mask)
                            c, out = GRU_Module(cfg)(obs_t, c)
                        elif rnn_type == "lti_linear":
                            c = c * (1.0 - done_mask)
                            c, out = LTI_Linear_Module(cfg)(obs_t, c)
                        elif rnn_type == "markovian":
                            c, out = Markovian_Module(cfg)(obs_t, c)
                        elif rnn_type == "mamba":
                            c = jax.tree_util.tree_map(lambda x: x * (1.0 - done_mask[:, None]), c)
                            c, out = MambaCell(d_model=cfg["RNN"]["D_MODEL"])(obs_t, c)
                        else:
                            out = obs_t
                        return c, out

                # in_axes=1 forces the scan to run across the Time dimension
                ScanRNN = nn.scan(SingleStepRNN, variable_broadcast="params", split_rngs={"params": False}, in_axes=1,
                                  out_axes=1)
                new_carry, rnn_out_trans = ScanRNN(name="rnn_baseline")(carry, (emb_trans, dones_trans))
                rnn_out = jnp.transpose(rnn_out_trans, (1, 0, 2))
        else:
            # --- RUNNING SDLG ---
            if ablation_cfg["USE_EXACT"]:
                VmappedExact = nn.vmap(FastBatchedExactSdLG, in_axes=(0, 0, 0, None), out_axes=(0, 0),
                                       variable_axes={'params': None}, split_rngs={'params': False})
                rnn_out_trans, new_carry = VmappedExact(self.config, ablation_cfg)(emb_trans, dones_trans, carry, True)
                rnn_out = jnp.transpose(rnn_out_trans, (1, 0, 2))
            else:
                from src.models.common.sdlg_hybrid import BatchedSdLGHybrid
                VmappedSdLG = nn.vmap(BatchedSdLGHybrid, in_axes=(0, 0, 0, None), out_axes=0,
                                      variable_axes={'params': None}, split_rngs={'params': False})
                ((X_actor_trans, X_critic_trans), _), new_carry = VmappedSdLG(n_layers=self.config["RNN"]["N_LAYERS"],
                                                                              d_model=self.config["RNN"]["D_MODEL"],
                                                                              d_k=32, d_v=32,
                                                                              d_ffc=self.config["RNN"]["D_FFC"], 
                                                                              r=self.config["RNN"].get("R", 4),
                                                                              window_size=16, alpha=0.1, eta=0.0,
                                                                              use_rope_phase=ablation_cfg.get("USE_ROPE", True),
                                                                              use_softplus_denominator=True,
                                                                              use_per_rank_norm=True,
                                                                              return_split=True)(emb_trans, dones_trans,
                                                                                                 carry, True)
                rnn_out = jnp.transpose(X_actor_trans, (1, 0, 2))

        # No MoE here. Keep it simple and fast.
        x_actor = x_critic = rnn_out

        action_logits = 5.0 * jnp.tanh(
            jnp.nan_to_num(nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(x_actor), nan=0.0, posinf=5.0,
                           neginf=-5.0))
        critic_values = jnp.nan_to_num(jnp.squeeze(nn.Dense(1, kernel_init=orthogonal(1.0))(x_critic), axis=-1),
                                       nan=0.0, posinf=10.0, neginf=-10.0)

        return action_logits, critic_values, jnp.zeros_like(critic_values), new_carry


# =====================================================================
# --- MECHANISTIC BOTTLENECK PROBE ---
# =====================================================================
@functools.partial(jax.jit, static_argnums=(3,))
def get_max_local_grad(params, probe_obs, probe_carry, network_apply_fn):
    T = probe_obs.shape[0]
    probe_dones = jnp.zeros((T, 1))

    def bottleneck_loss_fn(obs):
        _, critic_values, _, _ = network_apply_fn(params, probe_carry, (obs, probe_dones))
        return jnp.mean(critic_values)

    local_grad = jax.grad(bottleneck_loss_fn)(probe_obs)
    return jnp.max(jnp.abs(local_grad))


# =====================================================================
# --- EVALUATION ENGINE ---
# =====================================================================
def run_corridor_eval(rng, params, network, rnn_carry_init_fn, H):
    eval_env = PassiveTMaze(corridor_length=H)
    num_eval_envs = 256
    max_steps = H + 5
    rng, _rng = jax.random.split(rng)
    keys = jax.random.split(_rng, num_eval_envs)
    obs, env_state = jax.vmap(eval_env.reset, in_axes=(0, None))(keys, eval_env.default_params)
    init_carry_state = rnn_carry_init_fn(num_eval_envs)

    def step_fn(carry, _):
        obs, env_state, hstate, reward_sum, done, rng_step = carry
        logits, _, _, next_hstate = network.apply(params, hstate,
                                                  (jnp.expand_dims(obs, axis=0), jnp.expand_dims(done, axis=0)))
        pi = distrax.Categorical(logits=logits.squeeze(0))
        rng_step, _rng = jax.random.split(rng_step)
        next_obs, next_env_state, reward, next_done, _ = jax.vmap(eval_env.step, in_axes=(0, 0, 0, None))(
            jax.random.split(_rng, num_eval_envs), env_state, pi.mode(), eval_env.default_params)

        def flush(c_in, d_in):
            return jax.tree_util.tree_map(
                lambda x: (x * (1.0 - jnp.expand_dims(d_in, axis=tuple(range(1, 1 + x.ndim - d_in.ndim))))).astype(
                    x.dtype) if getattr(x, "ndim", 0) > 0 else x, c_in)

        return (next_obs, next_env_state, flush(next_hstate, next_done), reward_sum + reward * (1.0 - done), next_done,
                rng_step), None

    rng, _rng = jax.random.split(rng)
    final_carry, _ = jax.lax.scan(step_fn, (obs, env_state, init_carry_state, jnp.zeros(num_eval_envs),
                                            jnp.zeros(num_eval_envs, dtype=bool), _rng), None, max_steps)
    return float(jnp.mean(final_carry[3] >= 0.99))


# =====================================================================
# --- PLOTTING GENERATOR (ROPE VS COSINE) ---
# =====================================================================
# =====================================================================
# --- PLOTTING GENERATOR (CHAPTER 5: MEMORY HORIZONS) ---
# =====================================================================
def generate_rope_vs_cosine_plot(training_csv, zs_csv):
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np
    except ImportError:
        print("Missing libraries (pandas/matplotlib). Skipping plot generation.")
        return

    print("\nGenerating Publication-Ready Plots for Memory Horizons...")
    os.makedirs("plots", exist_ok=True)
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman']})

    # Load the data
    df_train = pd.read_csv(training_csv)
    df_zs = pd.read_csv(zs_csv)

    target_models = ["Exact_LogBias_RoPE", "Approx_LogBias_RoPE", "Baseline_Mamba", "Baseline_GRU"]

    colors = {
        "Exact_LogBias_RoPE": "#2c7bb6",  # Blue
        "Approx_LogBias_RoPE": "#d7191c",  # Red
        "Baseline_Mamba": "#984ea3",  # Purple (New!)
        "Baseline_GRU": "#fdae61"  # Orange
    }

    labels = {
        "Exact_LogBias_RoPE": "Exact SdLG",
        "Approx_LogBias_RoPE": "Approx SdLG",
        "Baseline_Mamba": "Mamba",  # New!
        "Baseline_GRU": "GRU"
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # =================================================================
    # Panel A: The BPTT Gradient Wall (Bar Chart)
    # =================================================================
    bar_data_50, bar_data_100 = [], []

    for m in target_models:
        m_df = df_train[df_train["model"] == m]

        m50 = m_df[m_df["H_train"] == 50]
        score50 = m50.groupby("seed")["mean_return"].max().mean() * 100 if not m50.empty else 0.0

        m100 = m_df[m_df["H_train"] == 100]
        score100 = m100.groupby("seed")["mean_return"].max().mean() * 100 if not m100.empty else 0.0

        bar_data_50.append(score50)
        bar_data_100.append(score100)

    x = np.arange(len(target_models))
    width = 0.35
    axes[0].bar(x - width / 2, bar_data_50, width, label="H=50 (Inside BPTT Window)", color="#a6cee3",
                edgecolor="black")
    axes[0].bar(x + width / 2, bar_data_100, width, label="H=100 (Crossing BPTT Wall)", color="#1f78b4",
                edgecolor="black")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([labels[m] for m in target_models], fontweight='bold', fontsize=11)
    axes[0].set_ylabel("Max Training Success Rate (%)", fontsize=11)
    axes[0].set_title("Overcoming Truncated BPTT (Window = 64)", fontweight='bold', fontsize=12)
    axes[0].set_ylim(0, 105)
    axes[0].legend(frameon=True)

    # =================================================================
    # Panel B: Training Convergence on H=100
    # =================================================================
    df100 = df_train[df_train["H_train"] == 100]

    for m in target_models:
        m_df = df100[df100["model"] == m]
        if m_df.empty: continue
        grp = m_df.groupby("step")
        mean_vals = grp["mean_return"].mean()
        std_vals = grp["mean_return"].std()
        steps = mean_vals.index.values / 1e6

        axes[1].plot(steps, mean_vals, label=labels[m], color=colors[m], linewidth=2.5)
        axes[1].fill_between(steps, mean_vals - std_vals, mean_vals + std_vals, color=colors[m], alpha=0.15)

    axes[1].set_title("Training Convergence (H=100)", fontweight='bold', fontsize=12)
    axes[1].set_xlabel("Environment Steps (Millions)", fontsize=11)
    axes[1].set_ylabel("Mean Episode Return", fontsize=11)
    axes[1].legend(frameon=True)

    # =================================================================
    # Panel C: Zero-Shot Extrapolation
    # =================================================================
    for m in target_models:
        m_zs = df_zs[(df_zs["model"] == m) & (df_zs["H_train"] == 100)]
        if not m_zs.empty:
            eval_lengths = [100, 150, 200]
            scores = [m_zs["val_H100"].mean() * 100, m_zs["val_H150"].mean() * 100, m_zs["val_H200"].mean() * 100]
            errs = [m_zs["val_H100"].std() * 100, m_zs["val_H150"].std() * 100, m_zs["val_H200"].std() * 100]

            axes[2].plot(eval_lengths, scores, label=labels[m], color=colors[m], linewidth=2.5, marker='o',
                         markersize=8)
            axes[2].fill_between(eval_lengths, np.array(scores) - np.array(errs), np.array(scores) + np.array(errs),
                                 color=colors[m], alpha=0.15)

    axes[2].set_xticks([100, 150, 200])
    axes[2].set_title("Zero-Shot Extrapolation", fontweight='bold', fontsize=12)
    axes[2].set_xlabel("Evaluation Corridor Length (H)", fontsize=11)
    axes[2].set_ylabel("Success Rate (%)", fontsize=11)
    axes[2].set_ylim(0, 105)
    axes[2].legend(frameon=True)

    fig.tight_layout()
    plot_path = "plots/rope_variants_tmaze.pdf"
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Successfully generated {plot_path}\n")


# =====================================================================
# --- INDEPENDENT RUNNER LOOP ---
# =====================================================================
def run_tmaze_ablations():
    print("====================================================================")
    print("=== EXP 18: UNIFIED EXACT SDLG & ROPE/COSINE ABLATION SUITE      ===")
    print("====================================================================")

    NUM_ENVS = 1024
    TOTAL_STEPS = 50_000_000
    LOG_INTERVAL = 10
    EARLY_STOP_THRESHOLD = 0.98
    CONSECUTIVE_STEPS = 3

    WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "sdlgHybrid")
    completed = load_completed_runs()

    training_csv = os.path.join(os.path.dirname(current_file_path), "rope_variants_training.csv")
    zs_csv = os.path.join(os.path.dirname(current_file_path), "rope_variants_zeroshot.csv")

    if not os.path.exists(training_csv):
        with open(training_csv, "w", newline="") as f: csv.writer(f).writerow(
            ["H_train", "model", "seed", "step", "mean_return", "train_len", "train_ent", "local_grad_norm"])
    if not os.path.exists(zs_csv):
        with open(zs_csv, "w", newline="") as f: csv.writer(f).writerow(
            ["H_train", "model", "seed", "val_H100", "val_H150", "val_H200", "val_H250"])

    import src.trainers.purejax_trainer
    def patched_flush_carry_state(hstate, last_done):
        return jax.tree_util.tree_map(lambda x: (
                    x * (1.0 - jnp.expand_dims(last_done, axis=tuple(range(1, 1 + x.ndim - last_done.ndim))))).astype(
            x.dtype) if getattr(x, "ndim", 0) > 0 else x, hstate)

    src.trainers.purejax_trainer.flush_carry_state = patched_flush_carry_state

    import src.models.purejaxrl.wrappers
    src.models.purejaxrl.wrappers.ActorCriticSdLGHybrid = ActorCriticAblation

    for H_train in [100]:
        for model_name in MODELS_TO_RUN:
            for seed in TRAINING_SEEDS:
                run_id = f"{model_name}_H{H_train}_seed_{seed}"

                if run_id in completed:
                    print(f"⏩ Skipping {run_id} (Already completed).")
                    continue

                jax.clear_caches()
                gc.collect()
                print(f"\n🚀 LAUNCHING TMAZE ABLATION: [{run_id}] | SEED = {seed}")

                ablation_cfg = ABLATIONS[model_name]
                NUM_STEPS = ablation_cfg.get("NUM_STEPS", 16) # Wall chunk size is 16
                STEPS_PER_UPDATE = NUM_ENVS * NUM_STEPS
                TOTAL_JIT_CALLS = (TOTAL_STEPS // STEPS_PER_UPDATE) // LOG_INTERVAL

                config = {
                    "NUM_ENVS": NUM_ENVS, "NUM_STEPS": NUM_STEPS, "TOTAL_TIMESTEPS": TOTAL_STEPS,
                    "UPDATE_EPOCHS": 4, "NUM_MINIBATCHES": 8, "GAMMA": 0.99, "GAE_LAMBDA": 0.95,
                    "CLIP_EPS": 0.2, "ENT_COEF": 0.01, "VF_COEF": 0.5, "MAX_GRAD_NORM": 0.5, "LR": 0.0005,
                    "ANNEAL_LR": True, "OPTIMISTIC_RESETS": False, "LOG_INTERVAL": LOG_INTERVAL,
                    "ABLATION": ablation_cfg,
                    "RNN": {"RNN_TYPE": "sdlg_hybrid", "D_MODEL": 128, "N_HEADS": 4, "N_LAYERS": 3, "D_FFC": 512,
                            "HIDDEN_DIM": 128, "R": 16},
                    "SEED": seed
                }

                wandb.init(project=WANDB_PROJECT, group="unified_exp18_tmaze_ablations",
                           name=f"{model_name}-H{H_train}-seed-{seed}", config=config, reinit=True)
                network = ActorCriticAblation(2, config)

                def rnn_carry_init(batch_size):
                    if ablation_cfg.get("IS_BASELINE", False):
                        hidden_dim = config["RNN"]["HIDDEN_DIM"]
                        if ablation_cfg["RNN_TYPE"] == "lstm":
                            return LSTM_Module.initialize_carry(batch_size, hidden_dim)
                        elif ablation_cfg["RNN_TYPE"] == "gru":
                            return GRU_Module.initialize_carry(batch_size, hidden_dim)
                        elif ablation_cfg["RNN_TYPE"] == "lti_linear":
                            return LTI_Linear_Module.initialize_carry(batch_size, hidden_dim)
                        elif ablation_cfg["RNN_TYPE"] == "markovian":
                            return Markovian_Module.initialize_carry(batch_size, hidden_dim)
                        elif ablation_cfg["RNN_TYPE"] == "mamba":
                            return MambaCell.initialize_carry(batch_size, hidden_dim)
                        elif ablation_cfg["RNN_TYPE"] == "gtrxl":
                            from src.models.common.gtrxl import BatchedGTrXL
                            return BatchedGTrXL.initialize_carry(
                                batch_size, memory_len=ablation_cfg.get("MEMORY_LEN", config.get("NUM_STEPS", 16)),
                                embedding_dim=config["RNN"]["D_MODEL"], layer_num=3
                            )
                        elif ablation_cfg["RNN_TYPE"] == "gla":
                            carry_list = []
                            for _ in range(config["RNN"]["N_LAYERS"]):
                                carry_list.append(
                                    ExactSdLGCarry(M=jnp.zeros((config["RNN"]["N_HEADS"], 32, 32), dtype=jnp.float32),
                                                   z=jnp.zeros((config["RNN"]["N_HEADS"], 32), dtype=jnp.float32),
                                                   swa_k=jnp.zeros((1, 32), dtype=jnp.float32),
                                                   swa_v=jnp.zeros((1, 32), dtype=jnp.float32),
                                                   step_count=jnp.zeros((), dtype=jnp.float32)))
                            return jax.tree_util.tree_map(
                                lambda x: jnp.repeat(jnp.expand_dims(x, 0), batch_size, axis=0), tuple(carry_list))
                    else:
                        if ablation_cfg.get("USE_EXACT", False):
                            carry_list = []
                            for i in range(config["RNN"]["N_LAYERS"]):
                                carry_list.append(
                                    ExactSdLGCarry(M=jnp.zeros((config["RNN"]["N_HEADS"], 32, 32), dtype=jnp.float32),
                                                   z=jnp.zeros((config["RNN"]["N_HEADS"], 32), dtype=jnp.float32),
                                                   swa_k=jnp.zeros((16, 32), dtype=jnp.float32),
                                                   swa_v=jnp.zeros((16, 32), dtype=jnp.float32),
                                                   step_count=jnp.zeros((), dtype=jnp.float32),
                                                   rng=jax.random.fold_in(jax.random.PRNGKey(0), i),
                                                   time_scale=jnp.ones((), dtype=jnp.float32)))
                            
                            def init_per_batch(c):
                                # Fold in batch index to RNG so each batch element gets unique scales
                                def fold_batch(x):
                                    if hasattr(x, 'shape') and x.shape == (2,) and x.dtype == jnp.uint32: # It's an RNG key
                                        keys = jax.vmap(lambda b: jax.random.fold_in(x, b))(jnp.arange(batch_size))
                                        return keys
                                    return jnp.repeat(jnp.expand_dims(x, 0), batch_size, axis=0)
                                return jax.tree_util.tree_map(fold_batch, c)
                            
                            return jax.tree_util.tree_map(init_per_batch, tuple(carry_list))
                        else:
                            from src.models.common.sdlg_hybrid import BatchedSdLGHybrid
                            return jax.tree_util.tree_map(
                                lambda x: jnp.repeat(jnp.expand_dims(x, 0), batch_size, axis=0),
                                BatchedSdLGHybrid.initialize_carry(batch_size, config["RNN"]["N_LAYERS"], 32, 32, config["RNN"]["R"],
                                                                   16))

                rng = jax.random.PRNGKey(seed)
                env = LogWrapper(PassiveTMaze(corridor_length=H_train))
                raw_update_fn, runner_state = make_update(rng, config, env, env.default_params, lambda: network,
                                                          rnn_carry_init)
                obs_shape = env.observation_space(env.default_params).shape

                @jax.jit
                def compiled_update(r_state):
                    out = raw_update_fn(r_state)
                    m = out["metric"]
                    d_mask = m.get("returned_episode", jnp.ones_like(m.get("entropy", jnp.zeros(1))))

                    def mean_d(x): return jnp.sum(x * d_mask) / jnp.maximum(1.0, jnp.sum(d_mask))

                    return out["runner_state"], {
                        "train_ret": mean_d(m.get("returned_episode_returns", jnp.zeros_like(d_mask))),
                        "train_len": mean_d(m.get("returned_episode_lengths", jnp.zeros_like(d_mask))),
                        "train_ent": jnp.mean(m.get("entropy", jnp.zeros(1))),
                        "train_vloss": jnp.mean(m.get("value_loss", jnp.zeros(1)))}

                steps_processed = 0
                start_wall_time = time.time()
                early_stop_counter = 0

                for jit_idx in range(TOTAL_JIT_CALLS):
                    runner_state, metrics = compiled_update(runner_state)
                    runner_state = jax.block_until_ready(runner_state)
                    metrics = jax.device_get(metrics)

                    steps_processed += STEPS_PER_UPDATE * LOG_INTERVAL
                    sps = steps_processed / max(time.time() - start_wall_time, 1.0)
                    train_ret = float(metrics["train_ret"])

                    probe_obs = jax.random.uniform(jax.random.split(rng)[1], (100, 1, *obs_shape))
                    local_grad_norm = float(
                        get_max_local_grad(runner_state[0].params, probe_obs, rnn_carry_init(1), network.apply))

                    wandb.log({"System/SPS": sps, "Train/Mean_Return": train_ret,
                               "Train/Mean_Length": float(metrics["train_len"]),
                               "Train/Policy_Entropy": float(metrics["train_ent"]),
                               "Train/Value_Loss": float(metrics["train_vloss"]),
                               "Diagnostics/Local_Grad_Norm": local_grad_norm}, step=steps_processed)

                    with open(training_csv, "a", newline="") as f:
                        csv.writer(f).writerow(
                            [H_train, model_name, seed, steps_processed, train_ret, float(metrics["train_len"]),
                             float(metrics["train_ent"]), local_grad_norm])

                    print(
                        f"  [JIT {jit_idx + 1:>3}/{TOTAL_JIT_CALLS}] Step: {steps_processed / 1e6:>5.2f}M | Ret: {train_ret:>5.2f} | Grad: {local_grad_norm:>5.3f} | SPS: {int(sps):>5}")

                    if train_ret >= EARLY_STOP_THRESHOLD:
                        early_stop_counter += 1
                        if early_stop_counter >= CONSECUTIVE_STEPS:
                            print(
                                f"⏹️  [EARLY STOP] Solved H={H_train}! Proceeding to Zero-Shot Generalization Evaluation.")
                            break
                    else:
                        early_stop_counter = 0

                print("\n📏 Probing Zero-Shot Generalization Memory Horizons...")
                eval_rng = jax.random.PRNGKey(99)
                zs_results = {}

                for H_eval in [100, 150, 200, 250]:
                    sr = run_corridor_eval(eval_rng, runner_state[0].params, network, rnn_carry_init, H_eval)
                    zs_results[H_eval] = sr
                    print(f"   - Corridor Length H = {H_eval:<3} | Success Rate = {sr * 100:>5.1f}%")
                    wandb.log({f"Eval/Success_H{H_eval}": sr}, step=steps_processed)

                with open(zs_csv, "a", newline="") as f:
                    csv.writer(f).writerow(
                        [H_train, model_name, seed, zs_results[100], zs_results[150], zs_results[200], zs_results[250]])

                print(f"✅ [{run_id}] Run & Evaluation Complete!\n")
                wandb.finish()
                save_completed_run(run_id)
                jax.clear_caches()
                gc.collect()

    generate_rope_vs_cosine_plot(training_csv, zs_csv)


if __name__ == "__main__":
    os.environ["HSA_ENABLE_SDMA"] = "0"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    run_tmaze_ablations()