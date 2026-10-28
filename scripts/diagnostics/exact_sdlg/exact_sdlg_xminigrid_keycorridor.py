# scripts/diagnostics/exp18/exp18_xminigrid_keycorridor.py

import sys
import os
import time
import json
import jax
import jax.numpy as jnp
import flax.linen as nn
import wandb
import distrax
from flax import struct
import gc
from typing import Any, Tuple, Callable, NamedTuple
from flax.linen.initializers import orthogonal, zeros, constant

current_file_path = os.path.abspath(__file__)
project_root = current_file_path
while not os.path.exists(os.path.join(project_root, "src")):
    project_root = os.path.dirname(project_root)
    if project_root == "/": break
if project_root not in sys.path: sys.path.append(project_root)

from src.trainers.purejax_trainer import make_update
import xminigrid

# =====================================================================
# --- HEAD-TO-HEAD CONFIGURATION ---
# =====================================================================
ABLATIONS = {
    "Exact_SdLG_Dense": {"USE_EXACT": False, "USE_MOE": False, "GAMMA_BIAS": 0.0, "W_SIZE": 32},
    "ExactMoEv2_W32": {"USE_EXACT": True, "USE_MOE": True, "GAMMA_BIAS": "log", "W_SIZE": 32},
}

MODELS_TO_RUN = ["Exact_SdLG_Dense"]
TRAINING_SEEDS = [42]
COMPLETED_RUNS_PATH = os.path.join(os.path.dirname(current_file_path), "completed_keycorridor.json")


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
# 1. PUREJAXRL WRAPPER FOR XMINIGRID
# =====================================================================
@struct.dataclass
class EnvStateAndMetrics:
    timestep: Any
    episode_return: jnp.ndarray
    episode_length: jnp.ndarray


class PureJaxRLXMiniGridWrapper:
    def __init__(self, env_id):
        self.env, self.default_params = xminigrid.make(env_id)

    def action_space(self, params=None):
        p = params if params is not None else self.default_params

        class Space:
            def __init__(self, n): self.n = n

        return Space(self.env.num_actions(p))

    def observation_space(self, params=None):
        p = params if params is not None else self.default_params

        class Space:
            def __init__(self, shape): self.shape = shape

        return Space(self.env.observation_shape(p))

    def reset(self, key, params=None):
        p = params if params is not None else self.default_params
        timestep = self.env.reset(p, key)
        state = EnvStateAndMetrics(timestep=timestep, episode_return=jnp.array(0.0, dtype=jnp.float32),
                                   episode_length=jnp.array(0, dtype=jnp.int32))
        return timestep.observation, state

    def step(self, key, state, action, params=None):
        p = params if params is not None else self.default_params
        timestep = self.env.step(p, state.timestep, action)

        done = (timestep.step_type == 2)
        reward = timestep.reward

        new_ep_return = state.episode_return + reward
        new_ep_length = state.episode_length + 1

        info = {"returned_episode_returns": new_ep_return, "returned_episode_lengths": new_ep_length,
                "returned_episode": done}

        reset_timestep = self.env.reset(p, key)
        next_timestep = jax.tree_util.tree_map(lambda res, cur: jnp.where(done, res, cur), reset_timestep, timestep)
        next_timestep = next_timestep.replace(reward=timestep.reward, step_type=timestep.step_type)

        next_state = EnvStateAndMetrics(timestep=next_timestep, episode_return=jnp.where(done, 0.0, new_ep_return),
                                        episode_length=jnp.where(done, 0, new_ep_length))
        return next_timestep.observation, next_state, reward, done, info


ENV_NAME = "MiniGrid-DoorKey-5x5" # Default, overridden in run_xminigrid_keycorridor
TOTAL_STEPS = 50_000_000
RESUME = False


# =====================================================================
# --- ARCHITECTURE BLOCKS ---
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

        norm_swa = nn.LayerNorm(epsilon=1e-5)(inputs)
        swa_out, new_swa_k, new_swa_v = GatedSlidingWindowAttention(
            d_model=self.d_model, d_k=self.d_k, d_v=self.d_v, window_size=self.window_size
        )(norm_swa, terminations, carry.swa_k, carry.swa_v, carry.step_count, deterministic=deterministic)
        x_swa = GTrXLGRUGate(self.d_model, self.gru_bias)(inputs, swa_out)

        if T > 1:
            _, step_counts = jax.lax.scan(lambda c, t: ((c + 1.0) * (1.0 - t),) * 2, carry.step_count, terminations)
            new_step_count = step_counts[-1]
        else:
            new_step_count = (carry.step_count + 1.0) * (1.0 - terminations[0])

        norm_gla = nn.LayerNorm(epsilon=1e-5)(x_swa)
        gla_out, (new_M, new_z) = FastExactMultiScaleGLA(
            d_model=self.d_model, n_heads=self.n_heads, d_k=self.d_k, d_v=self.d_v, gamma_bias_type=self.gamma_bias_type
        )(norm_gla, terminations, (carry.M, carry.z))
        x_gla = GTrXLGRUGate(self.d_model, self.gru_bias)(x_swa, gla_out)

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
        for l in range(self.config["RNN"]["N_LAYERS"]):
            x, next_carry = FastExactSdLGHybridBlock(
                d_model=self.config["RNN"]["D_MODEL"], n_heads=self.config["RNN"]["N_HEADS"],
                d_k=32, d_v=32, d_ffc=self.config["RNN"]["D_FFC"], window_size=self.ablation_cfg["W_SIZE"],
                gamma_bias_type=self.ablation_cfg["GAMMA_BIAS"]
            )(x, terminations, carry[l], deterministic)
            new_carry.append(next_carry)
        return x, tuple(new_carry)


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


class ActorCriticAblation(nn.Module):
    action_dim: int;
    config: dict

    @nn.compact
    def __call__(self, carry, x):
        obs, dones = x

        # --- KEEP THIS FIX TO PREVENT VLOSS EXPLOSION ---
        obs_float = obs.astype(jnp.float32) / 10.0
        x_flat = obs_float.reshape(obs_float.shape[0], obs_float.shape[1], -1)

        emb = nn.relu(nn.Dense(self.config["RNN"]["D_MODEL"], kernel_init=orthogonal(jnp.sqrt(2)))(x_flat))
        emb = nn.relu(nn.Dense(self.config["RNN"]["D_MODEL"], kernel_init=orthogonal(jnp.sqrt(2)))(emb))
        emb = nn.LayerNorm()(emb)  # Hard bound representation drift
        # ------------------------------------------------

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
                d_k=32, d_v=32, d_ffc=self.config["RNN"]["D_FFC"], r=4, window_size=ablation_cfg["W_SIZE"], alpha=0.1,
                return_split=True
            )(emb_trans, dones_trans, carry, True)

        rnn_out = jnp.transpose(rnn_out_trans, (1, 0, 2))
        x_critic, x_actor = rnn_out, jax.lax.stop_gradient(rnn_out) * 0.9 + rnn_out * 0.1

        if ablation_cfg["USE_MOE"]:
            actor_rep = SparseMoESwiGLU(self.config["RNN"]["D_MODEL"], self.config["RNN"]["D_FFC"])(x_actor)
            critic_rep = SparseMoESwiGLU(self.config["RNN"]["D_MODEL"], self.config["RNN"]["D_FFC"])(x_critic)
        else:
            actor_rep, critic_rep = x_actor, x_critic

        action_logits = 5.0 * jnp.tanh(
            jnp.nan_to_num(nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(actor_rep), nan=0.0, posinf=5.0,
                           neginf=-5.0))

        # --- KEEP THIS CLIP TO PREVENT VLOSS EXPLOSION ---
        critic_values = jnp.clip(jnp.squeeze(nn.Dense(1, kernel_init=orthogonal(1.0))(critic_rep), axis=-1), -100.0,
                                 100.0)

        return action_logits, critic_values, jnp.zeros_like(critic_values), new_carry


# =====================================================================
# --- RUNNER LOOP ---
# =====================================================================
def run_xminigrid_keycorridor():
    print("====================================================================")
    print(f"=== EXP 18: BOSS FIGHT - XMINIGRID {ENV_NAME} ===")
    print("====================================================================")

    NUM_ENVS = 512
    NUM_STEPS = 64  # Strict BPTT gradient wall
    STEPS_PER_UPDATE = NUM_ENVS * NUM_STEPS
    global TOTAL_STEPS
    LOG_INTERVAL = 10
    TOTAL_JIT_CALLS = (TOTAL_STEPS // STEPS_PER_UPDATE) // LOG_INTERVAL
    EARLY_STOP_THRESHOLD = 0.96
    CONSECUTIVE_STEPS = 3

    WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "sdlgHybrid")
    completed = load_completed_runs()

    # Monkey patch standard trainer
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
            run_id = f"{model_name}_{ENV_NAME}_seed_{seed}"

            if run_id in completed and not RESUME:
                print(f"⏩ Skipping {run_id} (Already completed).")
                continue

            jax.clear_caches()
            gc.collect()
            print(f"\n🚀 LAUNCHING SHOWDOWN: [{run_id}] | SEED = {seed}")

            ablation_cfg = ABLATIONS[model_name]
            W_SIZE = ablation_cfg["W_SIZE"]

            config = {
                "NUM_ENVS": NUM_ENVS, "NUM_STEPS": NUM_STEPS, "TOTAL_TIMESTEPS": TOTAL_STEPS,
                "UPDATE_EPOCHS": 4, "NUM_MINIBATCHES": 8, "GAMMA": 0.99, "GAE_LAMBDA": 0.95,
                "CLIP_EPS": 0.2, "ENT_COEF": 0.01, "VF_COEF": 0.5, "MAX_GRAD_NORM": 0.5, "LR": 0.0003,
                "ANNEAL_LR": True, "OPTIMISTIC_RESETS": False, "LOG_INTERVAL": LOG_INTERVAL,
                "ABLATION": ablation_cfg,
                "RNN": {"RNN_TYPE": "sdlg_hybrid", "D_MODEL": 256, "N_HEADS": 8, "N_LAYERS": 3, "D_FFC": 512},
                "SEED": seed
            }

            wandb.init(project=WANDB_PROJECT, group="exp18_keycorridor_bossfight", name=f"{model_name}-seed-{seed}", config=config,
                       reinit=True)
            env = PureJaxRLXMiniGridWrapper(ENV_NAME)
            network = ActorCriticAblation(env.action_space().n, config)

            def rnn_carry_init(batch_size):
                if ablation_cfg["USE_EXACT"]:
                    carry_list = []
                    for _ in range(config["RNN"]["N_LAYERS"]):
                        M = jnp.zeros((config["RNN"]["N_HEADS"], 32, 32), dtype=jnp.float32)
                        z = jnp.zeros((config["RNN"]["N_HEADS"], 32), dtype=jnp.float32)
                        swa_k = jnp.zeros((W_SIZE, 32), dtype=jnp.float32)
                        swa_v = jnp.zeros((W_SIZE, 32), dtype=jnp.float32)
                        step_count = jnp.zeros((), dtype=jnp.float32)
                        carry_list.append(ExactSdLGCarry(M=M, z=z, swa_k=swa_k, swa_v=swa_v, step_count=step_count))
                    unbatched = tuple(carry_list)
                    return jax.tree_util.tree_map(lambda x: jnp.repeat(jnp.expand_dims(x, 0), batch_size, axis=0),
                                                  unbatched)
                else:
                    from src.models.common.sdlg_hybrid import BatchedSdLGHybrid
                    c = BatchedSdLGHybrid.initialize_carry(batch_size, config["RNN"]["N_LAYERS"], 32, 32, 4, W_SIZE)
                    return jax.tree_util.tree_map(lambda x: jnp.repeat(jnp.expand_dims(x, 0), batch_size, axis=0), c)

            rng = jax.random.PRNGKey(seed)
            raw_update_fn, runner_state = make_update(rng, config, env, env.default_params, lambda: network, rnn_carry_init)

            if RESUME:
                ckpt_dir = os.path.abspath(f"checkpoints/{run_id}")
                if os.path.exists(ckpt_dir):
                    print(f"🔄 Resuming from {ckpt_dir}...")
                    import orbax.checkpoint as ocp
                    ckpt_mgr = ocp.CheckpointManager(ckpt_dir, ocp.Checkpointer(ocp.PyTreeCheckpointHandler()))
                    step = ckpt_mgr.latest_step()
                    if step is not None:
                        restored = ckpt_mgr.restore(step)
                        if "params" in restored:
                            ts = runner_state[0]
                            ts = ts.replace(params=restored["params"])
                            runner_state = (ts, runner_state[1], runner_state[2], runner_state[3], runner_state[4], runner_state[5])
                            print(f"✅ Loaded params from step {step}!")
                        else:
                            print("⚠️ Could not find 'params' in checkpoint.")
                else:
                    print(f"⚠️ Checkpoint directory {ckpt_dir} not found. Starting from scratch.")

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

            import orbax.checkpoint as ocp
            ckpt_mgr = ocp.CheckpointManager(
                os.path.abspath(f"checkpoints/{run_id}"),
                ocp.Checkpointer(ocp.PyTreeCheckpointHandler()),
                options=ocp.CheckpointManagerOptions(max_to_keep=5, create=True)
            )

            steps_processed = 0
            start_wall_time = time.time()
            early_stop_counter = 0

            for jit_idx in range(TOTAL_JIT_CALLS):
                runner_state, metrics = compiled_update(runner_state)
                runner_state = jax.block_until_ready(runner_state)
                metrics = jax.device_get(metrics)

                steps_processed += STEPS_PER_UPDATE * LOG_INTERVAL
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

                if train_ret >= EARLY_STOP_THRESHOLD:
                    early_stop_counter += 1
                    if early_stop_counter >= CONSECUTIVE_STEPS:
                        print(f"⏹️  [EARLY STOP] Boss Environment Conquered!")
                        break
                else:
                    early_stop_counter = 0
                    
                if train_ret >= 0.85:
                    ckpt_state = {"params": runner_state[0].params}
                    print(f"💾 High return ({train_ret})! Saving model to checkpoints/{run_id}/step_{steps_processed}")
                    ckpt_mgr.save(steps_processed, ckpt_state)

            print(f"✅ [{run_id}] Evaluation Complete!\n")
            
            ckpt_state = {
                "params": runner_state[0].params
            }
            
            print(f"💾 Saving model to checkpoints/{run_id}/step_{steps_processed}")
            ckpt_mgr.save(steps_processed, ckpt_state)
            ckpt_mgr.wait_until_finished()
            print("✅ Model saved!")

            wandb.finish()
            save_completed_run(run_id)

            jax.clear_caches()
            gc.collect()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run XMiniGrid Boss Fight")
    parser.add_argument("--env", type=str, default="MiniGrid-DoorKey-5x5", help="Target XMiniGrid environment")
    parser.add_argument("--total_steps", type=int, default=50_000_000, help="Total timesteps to run")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint if exists")
    args = parser.parse_args()
    
    ENV_NAME = args.env
    TOTAL_STEPS = args.total_steps
    RESUME = args.resume
    print(f"🎯 Target Environment: {ENV_NAME}")

    os.environ["HSA_ENABLE_SDMA"] = "0"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
    run_xminigrid_keycorridor()