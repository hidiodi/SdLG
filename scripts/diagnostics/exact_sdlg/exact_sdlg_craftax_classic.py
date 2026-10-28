# scripts/diagnostics/exp18/exp18_craftax_classic.py

import sys
import os
import time
import json
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import wandb
import distrax
import functools
import numpy as np
import gc
from flax import struct
from typing import Any, Tuple, Callable, NamedTuple
from flax.linen.initializers import orthogonal, zeros, constant

current_file_path = os.path.abspath(__file__)
project_root = current_file_path
while not os.path.exists(os.path.join(project_root, "src")):
    project_root = os.path.dirname(project_root)
    if project_root == "/": break
if project_root not in sys.path: sys.path.append(project_root)

from src.trainers.purejax_trainer import make_update
from src.envs.builder import build_purejaxrl_env
from src.utils.config_schema import StaticEnvConfig, DynamicTrainingConfig

# Dynamically load the correct Achievement constants for Craftax-Classic
try:
    from craftax.craftax_classic.constants import Achievement
except ImportError:
    from craftax.craftax.constants import Achievement

ACH_KEYS = [ach.name for ach in Achievement]

# =====================================================================
# --- MASSIVE ABLATION DICTIONARY (CRAFTAX-CLASSIC) ---
# =====================================================================
ABLATIONS = {
    # --- THE CONTROL MODEL (Baseline for comparison) ---
    "Control_Exact_Dense": {"USE_EXACT": True, "USE_MOE": False, "GAMMA_BIAS": "log", "W_SIZE": 32, "D_FFC": 1024, "N_HEADS": 8, "GRU_BIAS": 2.0},

    # --- 6.1 MATHEMATICAL PRECISION ---
    "Ablation_Approx_Dense": {"USE_EXACT": False, "USE_MOE": False, "GAMMA_BIAS": "log", "W_SIZE": 32, "D_FFC": 1024, "N_HEADS": 8, "GRU_BIAS": 2.0},

    # --- 6.4 THE GAMMA PRIOR (TIMESCALES) ---
    "Ablation_GammaZero": {"USE_EXACT": True, "USE_MOE": False, "GAMMA_BIAS": "0.0", "W_SIZE": 32, "D_FFC": 1024, "N_HEADS": 8, "GRU_BIAS": 2.0},

    # --- 6.5 RESIDUAL SKIP CONNECTIONS (GRU BIAS) ---
    # We skip Bias=4.0 since it crashes XLA, we compare 0.0 vs the control (2.0)
    "Ablation_GRUBias_0": {"USE_EXACT": True, "USE_MOE": False, "GAMMA_BIAS": "log", "W_SIZE": 32, "D_FFC": 1024, "N_HEADS": 8, "GRU_BIAS": 0.0},

    # --- 6.6 NETWORK CAPACITY SCALING ---
    "Capacity_WideFFN_2048": {"USE_EXACT": True, "USE_MOE": False, "GAMMA_BIAS": "log", "W_SIZE": 32, "D_FFC": 2048, "N_HEADS": 8, "GRU_BIAS": 2.0},
    "Capacity_NarrowFFN_512": {"USE_EXACT": True, "USE_MOE": False, "GAMMA_BIAS": "log", "W_SIZE": 32, "D_FFC": 512, "N_HEADS": 8, "GRU_BIAS": 2.0},
    "Capacity_16Heads": {"USE_EXACT": True, "USE_MOE": False, "GAMMA_BIAS": "log", "W_SIZE": 32, "D_FFC": 1024, "N_HEADS": 16, "GRU_BIAS": 2.0},
    "Capacity_2Heads": {"USE_EXACT": True, "USE_MOE": False, "GAMMA_BIAS": "log", "W_SIZE": 32, "D_FFC": 1024, "N_HEADS": 2, "GRU_BIAS": 2.0},
}

# Only run active keys
MODELS_TO_RUN = ["Ablation_Approx_Dense"]
# Rule 1: Sweeping over 1 seed
TRAINING_SEEDS = [42]
COMPLETED_RUNS_PATH = os.path.join(os.path.dirname(current_file_path), "completed_craftax_plethora.json")


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
# --- EXACT GLA MULTI-SCALE MEMORY CORE ---
# =====================================================================
class ExactSdLGCarry(NamedTuple):
    M: jnp.ndarray
    z: jnp.ndarray
    swa_k: jnp.ndarray
    swa_v: jnp.ndarray
    step_count: jnp.ndarray


class FastExactMultiScaleGLA(nn.Module):
    d_model: int;
    n_heads: int;
    d_k: int;
    d_v: int;
    gamma_bias_type: str

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

    def __call__(self, inputs, terminations, carry):
        T, _ = inputs.shape
        M_prev, z_prev = carry
        q = self.q_proj(inputs).reshape((T, self.n_heads, self.d_k))
        k = self.k_proj(inputs).reshape((T, self.n_heads, self.d_k))
        v = self.v_proj(inputs).reshape((T, self.n_heads, self.d_v))

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
    gamma_bias_type: str
    gru_bias: float = 2.0

    @nn.compact
    def __call__(self, inputs, terminations, carry: ExactSdLGCarry, deterministic=True):
        from src.models.common.sdlg_hybrid import GatedSlidingWindowAttention, GTrXLGRUGate
        T = inputs.shape[0]

        # 1. Local SWA (Bypass if W_SIZE == 0)
        if self.window_size > 0:
            norm_swa = nn.LayerNorm(epsilon=1e-5)(inputs)
            swa_out, new_swa_k, new_swa_v = GatedSlidingWindowAttention(
                d_model=self.d_model, d_k=self.d_k, d_v=self.d_v, window_size=self.window_size
            )(norm_swa, terminations, carry.swa_k, carry.swa_v, carry.step_count, deterministic=deterministic)
            x_swa = GTrXLGRUGate(self.d_model, self.gru_bias)(inputs, swa_out)
        else:
            x_swa = inputs
            new_swa_k, new_swa_v = carry.swa_k, carry.swa_v

        # Update step count
        if T > 1:
            _, step_counts = jax.lax.scan(lambda c, t: ((c + 1.0) * (1.0 - t),) * 2, carry.step_count, terminations)
            new_step_count = step_counts[-1]
        else:
            new_step_count = (carry.step_count + 1.0) * (1.0 - terminations[0])

        # 2. Global Exact GLA
        norm_gla = nn.LayerNorm(epsilon=1e-5)(x_swa)
        gla_out, (new_M, new_z) = FastExactMultiScaleGLA(
            d_model=self.d_model, n_heads=self.n_heads, d_k=self.d_k, d_v=self.d_v, gamma_bias_type=self.gamma_bias_type
        )(norm_gla, terminations, (carry.M, carry.z))
        x_gla = GTrXLGRUGate(self.d_model, self.gru_bias)(x_swa, gla_out)

        # 3. FFN
        norm_ffn = nn.LayerNorm(epsilon=1e-5)(x_gla)
        ffn_out = nn.Sequential([nn.Dense(self.d_ffc, kernel_init=orthogonal(jnp.sqrt(2))), nn.gelu,
                                 nn.Dense(self.d_model, kernel_init=orthogonal(jnp.sqrt(2)))])(norm_ffn)
        out = GTrXLGRUGate(self.d_model, self.gru_bias)(x_gla, ffn_out)

        return out, ExactSdLGCarry(M=new_M, z=new_z, swa_k=new_swa_k, swa_v=new_swa_v, step_count=new_step_count)


class FastBatchedExactSdLG(nn.Module):
    config: dict;
    ablation_cfg: dict

    @nn.compact
    def __call__(self, inputs, terminations, carry, deterministic=True):
        x = inputs
        new_carry = []
        gru_b = self.ablation_cfg.get("GRU_BIAS", 2.0)

        for l in range(self.config["RNN"]["N_LAYERS"]):
            x, next_carry = FastExactSdLGHybridBlock(
                d_model=self.config["RNN"]["D_MODEL"], n_heads=self.config["RNN"]["N_HEADS"],
                d_k=self.config["RNN"]["D_K"], d_v=self.config["RNN"]["D_V"],
                d_ffc=self.config["RNN"]["D_FFC"], window_size=self.ablation_cfg["W_SIZE"],
                gamma_bias_type=self.ablation_cfg["GAMMA_BIAS"], gru_bias=gru_b
            )(x, terminations, carry[l], deterministic)
            new_carry.append(next_carry)
        return x, tuple(new_carry)


# =====================================================================
# --- MOE HEAD WRAPPERS ---
# =====================================================================
class SparseMoESwiGLU(nn.Module):
    d_model: int;
    d_ffc: int;
    num_experts: int = 4;
    top_k: int = 2

    @nn.compact
    def __call__(self, x):
        prefix_shape = x.shape[:-1]
        routing_probs = jax.nn.softmax(nn.Dense(self.num_experts, use_bias=False)(x), axis=-1)
        topk_probs, topk_indices = jax.lax.top_k(routing_probs, k=self.top_k)
        expert_weights = topk_probs / (jnp.sum(topk_probs, axis=-1, keepdims=True) + 1e-9)

        expert_outs = []
        for e in range(self.num_experts):
            gate = nn.Dense(self.d_ffc, use_bias=False)(x)
            up = nn.Dense(self.d_ffc, use_bias=False)(x)
            expert_outs.append(nn.Dense(self.d_model, use_bias=False)(jax.nn.silu(gate) * up))

        expert_outputs = jnp.nan_to_num(jnp.stack(expert_outs, axis=-2), nan=0.0)
        combined = jnp.zeros_like(x)
        for k in range(self.top_k):
            indices = jnp.broadcast_to(topk_indices[..., k, None, None], (*prefix_shape, 1, self.d_model))
            selected = jnp.take_along_axis(expert_outputs, indices, axis=-2).squeeze(-2)
            combined += expert_weights[..., k, None] * selected
        return combined


# --- CRAFTAX CNN ---
class CraftaxPatchedRepresentationNetwork(nn.Module):
    d_model: int

    @nn.compact
    def __call__(self, obs):
        T, B = obs.shape[0], obs.shape[1]
        obs_flat = obs.reshape(T * B, *obs.shape[2:])
        if obs.ndim > 3:
            x = nn.Conv(32, kernel_size=(3, 3), strides=(1, 1), padding="SAME")(obs_flat)
            x = nn.relu(x)
            x = nn.Conv(64, kernel_size=(3, 3), strides=(1, 1), padding="SAME")(x)
            x = nn.relu(x)
            flat = x.reshape(x.shape[0], -1)
        else:
            flat = obs_flat.reshape(obs_flat.shape[0], -1)

        x = nn.Dense(self.d_model, kernel_init=orthogonal(jnp.sqrt(2)))(flat)
        x = nn.relu(x)
        out = nn.Dense(self.d_model, kernel_init=orthogonal(jnp.sqrt(2)))(x)
        return nn.relu(out).reshape(T, B, self.d_model)


# --- ABLATION SHELL ---
class ActorCriticAblation(nn.Module):
    action_dim: int;
    config: dict

    @nn.compact
    def __call__(self, carry, x):
        obs, dones = x
        emb = CraftaxPatchedRepresentationNetwork(d_model=self.config["RNN"]["D_MODEL"])(obs)

        emb_trans, dones_trans = jnp.transpose(emb, (1, 0, 2)), jnp.transpose(dones, (1, 0))
        ablation_cfg = self.config["ABLATION"]

        if ablation_cfg["USE_EXACT"]:
            VmappedExact = nn.vmap(FastBatchedExactSdLG, in_axes=(0, 0, 0, None), out_axes=(0, 0),
                                   variable_axes={'params': None}, split_rngs={'params': False})
            rnn_out_trans, new_carry = VmappedExact(self.config, ablation_cfg)(emb_trans, dones_trans, carry, True)
        else:
            from src.models.common.sdlg_hybrid import BatchedSdLGHybrid
            VmappedSdLG = nn.vmap(BatchedSdLGHybrid, in_axes=(0, 0, 0, None), out_axes=0,
                                  variable_axes={'params': None}, split_rngs={'params': False})
            ((rnn_out_trans, _), _), new_carry = VmappedSdLG(
                n_layers=self.config["RNN"]["N_LAYERS"], d_model=self.config["RNN"]["D_MODEL"],
                d_k=self.config["RNN"]["D_K"], d_v=self.config["RNN"]["D_V"], d_ffc=self.config["RNN"]["D_FFC"], r=4,
                window_size=ablation_cfg["W_SIZE"], alpha=0.1, return_split=True
            )(emb_trans, dones_trans, carry, True)

        rnn_out = jnp.transpose(rnn_out_trans, (1, 0, 2))
        x_critic, x_actor = rnn_out, jax.lax.stop_gradient(rnn_out) * 0.9 + rnn_out * 0.1

        if ablation_cfg["USE_MOE"]:
            k = ablation_cfg.get("TOP_K", 2)
            n_exp = ablation_cfg.get("NUM_EXPERTS", 4)
            actor_rep = SparseMoESwiGLU(self.config["RNN"]["D_MODEL"], self.config["RNN"]["D_FFC"], num_experts=n_exp,
                                        top_k=k)(x_actor)
            critic_rep = SparseMoESwiGLU(self.config["RNN"]["D_MODEL"], self.config["RNN"]["D_FFC"], num_experts=n_exp,
                                         top_k=k)(x_critic)
        else:
            actor_rep, critic_rep = x_actor, x_critic

        action_logits = 5.0 * jnp.tanh(
            jnp.nan_to_num(nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(actor_rep), nan=0.0, posinf=5.0,
                           neginf=-5.0))
        critic_values = jnp.nan_to_num(jnp.squeeze(nn.Dense(1, kernel_init=orthogonal(1.0))(critic_rep), axis=-1),
                                       nan=0.0, posinf=10.0, neginf=-10.0)

        return action_logits, critic_values, jnp.zeros_like(critic_values), new_carry


# =====================================================================
# --- RUNNER LOOP ---
# =====================================================================
def run_craftax_classic_plethora():
    print("====================================================================")
    print("=== EXP 18: PLETHORA ABLATION SWEEP (CRAFTAX CLASSIC) ===")
    print("====================================================================")

    ENV_NAME = "Craftax-Classic-Symbolic-v1"
    NUM_ENVS = 256
    # Rule 3: strictly hardcode NUM_STEPS = 64
    NUM_STEPS = 64
    STEPS_PER_UPDATE = NUM_ENVS * NUM_STEPS
    TOTAL_STEPS = 1_000_000_000
    LOG_INTERVAL_UPDATES = 50
    TOTAL_JIT_CALLS = (TOTAL_STEPS // STEPS_PER_UPDATE) // LOG_INTERVAL_UPDATES

    WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "sdlgHybrid")
    completed = load_completed_runs()

    # Monkey patch standard trainer to handle bool flushes
    import src.trainers.purejax_trainer
    def patched_flush_carry_state(hstate, last_done):
        def flush_leaf(x):
            dims_to_add = x.ndim - last_done.ndim
            broadcast_done = jnp.expand_dims(last_done, axis=tuple(range(1, 1 + dims_to_add)))
            return (x * (1.0 - broadcast_done)).astype(x.dtype)

        return jax.tree_util.tree_map(flush_leaf, hstate)

    src.trainers.purejax_trainer.flush_carry_state = patched_flush_carry_state

    import src.models.purejaxrl.wrappers
    src.models.purejaxrl.wrappers.ActorCriticSdLGHybrid = ActorCriticAblation

    for model_name in MODELS_TO_RUN:
        for seed in TRAINING_SEEDS:
            run_id = f"{model_name}_{ENV_NAME}_seed_{seed}_1B_run"

            if run_id in completed:
                print(f"⏩ Skipping {run_id} (Already completed).")
                continue

            # Rule 1: Clean caches & collect garbage
            jax.clear_caches()
            gc.collect()
            print(f"\n🚀 LAUNCHING PLETHORA ABLATION: [{run_id}] | SEED = {seed}")

            ablation_cfg = ABLATIONS[model_name]

            # Dynamic Hyperparameter Override
            W_SIZE = ablation_cfg["W_SIZE"]
            D_FFC = ablation_cfg.get("D_FFC", 1024)
            N_HEADS = ablation_cfg.get("N_HEADS", 8)
            # To maintain parameter parity when changing heads, scale D_K so N_HEADS * D_K = 256
            D_K = 256 // N_HEADS

            config = {
                "NUM_ENVS": NUM_ENVS, "NUM_STEPS": NUM_STEPS, "TOTAL_TIMESTEPS": TOTAL_STEPS,
                "UPDATE_EPOCHS": 4, "NUM_MINIBATCHES": 8, "GAMMA": 0.99, "GAE_LAMBDA": 0.95,
                "CLIP_EPS": 0.2, "ENT_COEF": 0.01, "VF_COEF": 0.5, "MAX_GRAD_NORM": 0.5, "LR": 0.0003,
                "ANNEAL_LR": True, "OPTIMISTIC_RESETS": False, "LOG_INTERVAL": LOG_INTERVAL_UPDATES,
                "ABLATION": ablation_cfg,
                "RNN": {"RNN_TYPE": "sdlg_hybrid", "D_MODEL": 256, "N_HEADS": N_HEADS, "D_K": D_K, "D_V": D_K,
                        "N_LAYERS": 4, "D_FFC": D_FFC},
                "SEED": seed
            }

            wandb.init(
                project=WANDB_PROJECT,
                group="exp18_craftax_plethora",
                name=f"{model_name}-seed-{seed}",
                config=config,
                reinit=True
            )

            static_env = StaticEnvConfig(env_name=ENV_NAME, num_envs=NUM_ENVS, num_steps=NUM_STEPS, max_episode_steps=10000,
                                         optimistic_resets=False)
            dynamic_train = DynamicTrainingConfig(
                total_timesteps=TOTAL_STEPS, learning_rate=0.0003, gamma=0.99, gae_lambda=0.95,
                clip_eps=0.2, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, update_epochs=4,
                minibatch_size=2048, log_interval=LOG_INTERVAL_UPDATES
            )
            env, env_params = build_purejaxrl_env(static_env, dynamic_train)

            # Rule 4: Wrap environment in LogWrapper
            from src.envs.wrappers import LogWrapper
            if not isinstance(env, LogWrapper):
                env = LogWrapper(env)

            network = ActorCriticAblation(env.action_space(env_params).n, config)

            def rnn_carry_init(batch_size):
                if ablation_cfg["USE_EXACT"]:
                    carry_list = []
                    for _ in range(config["RNN"]["N_LAYERS"]):
                        M = jnp.zeros((N_HEADS, D_K, D_K), dtype=jnp.float32)
                        z = jnp.zeros((N_HEADS, D_K), dtype=jnp.float32)
                        swa_k = jnp.zeros((W_SIZE, D_K), dtype=jnp.float32) if W_SIZE > 0 else jnp.zeros((1, D_K))
                        swa_v = jnp.zeros((W_SIZE, D_K), dtype=jnp.float32) if W_SIZE > 0 else jnp.zeros((1, D_K))
                        step_count = jnp.zeros((), dtype=jnp.float32)
                        carry_list.append(ExactSdLGCarry(M=M, z=z, swa_k=swa_k, swa_v=swa_v, step_count=step_count))
                    unbatched = tuple(carry_list)
                    return jax.tree_util.tree_map(lambda x: jnp.repeat(jnp.expand_dims(x, 0), batch_size, axis=0),
                                                  unbatched)
                else:
                    from src.models.common.sdlg_hybrid import BatchedSdLGHybrid
                    c = BatchedSdLGHybrid.initialize_carry(batch_size, config["RNN"]["N_LAYERS"], D_K, D_K, 4, W_SIZE)
                    return jax.tree_util.tree_map(lambda x: jnp.repeat(jnp.expand_dims(x, 0), batch_size, axis=0), c)

            rng = jax.random.PRNGKey(seed)
            raw_update_fn, runner_state = make_update(rng, config, env, env_params, lambda: network, rnn_carry_init)

            @jax.jit
            def compiled_update(r_state):
                out = raw_update_fn(r_state)
                m = out["metric"]
                d_mask = m.get("returned_episode", jnp.ones_like(m.get("entropy", jnp.zeros(1))))

                def mean_d(x): return jnp.sum(x * d_mask) / jnp.maximum(1.0, jnp.sum(d_mask))

                reduced = {
                    "train_ret": mean_d(m.get("returned_episode_returns", jnp.zeros_like(d_mask))),
                    "train_len": mean_d(m.get("returned_episode_lengths", jnp.zeros_like(d_mask))),
                    "train_ent": jnp.mean(m.get("entropy", jnp.zeros(1))),
                    "train_vloss": jnp.mean(m.get("value_loss", jnp.zeros(1)))
                }
                return out["runner_state"], reduced

            steps_processed = 0
            start_wall_time = time.time()

            # Pure Training execution loop without eval halts
            for jit_idx in range(TOTAL_JIT_CALLS):
                runner_state, metrics = compiled_update(runner_state)
                runner_state = jax.block_until_ready(runner_state)
                metrics = jax.device_get(metrics)

                steps_processed += STEPS_PER_UPDATE * LOG_INTERVAL_UPDATES
                active_time = time.time() - start_wall_time
                sps = steps_processed / max(active_time, 1.0)
                train_ret = float(metrics["train_ret"])

                wandb.log({
                    "System/SPS": sps,
                    "Train/Mean_Return": train_ret,
                    "Train/Mean_Length": float(metrics["train_len"]),
                    "Train/Policy_Entropy": float(metrics["train_ent"]),
                    "Train/Value_Loss": float(metrics["train_vloss"])
                }, step=steps_processed)

                print(
                    f"  [JIT {jit_idx + 1:>3}/{TOTAL_JIT_CALLS}] Step: {steps_processed / 1e6:>5.2f}M | Ret: {train_ret:>5.2f} | SPS: {int(sps):>5}")
                
                if train_ret >= 20.0:
                    print(f"🎉 SOLVED! Reached return 20+ ({train_ret}) in {steps_processed / 1e6:.2f}M steps!")
                    break

            print(f"✅ [{run_id}] Execution Complete!\n")
            wandb.finish()
            save_completed_run(run_id)
            
            # Rule 1: Clean caches & collect garbage at the end of the seed run
            jax.clear_caches()
            gc.collect()


if __name__ == "__main__":
    os.environ["HSA_ENABLE_SDMA"] = "0"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
    run_craftax_classic_plethora()