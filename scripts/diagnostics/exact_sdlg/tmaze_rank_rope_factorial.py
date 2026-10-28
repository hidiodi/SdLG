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

current_file_path = os.path.abspath(__file__)
project_root = current_file_path
while not os.path.exists(os.path.join(project_root, "src")):
    project_root = os.path.dirname(project_root)
    if project_root == "/": break
if project_root not in sys.path: sys.path.append(project_root)

from src.envs.tmaze.jax_tmaze import PassiveTMaze
from src.trainers.purejax_trainer_boundary import make_update
from gymnax.wrappers.purerl import LogWrapper

ABLATIONS = {
    "SdLG-Exact": {"IS_BASELINE": False, "R": 32, "USE_ROPE": False},
    "SdLG-Approx": {"IS_BASELINE": False, "R": 16, "USE_ROPE": True},
    "SdLG-Exact-RoPE": {"IS_BASELINE": False, "R": 32, "USE_ROPE": True},
    "SdLG-Approx-NoRoPE": {"IS_BASELINE": False, "R": 16, "USE_ROPE": False},
}

MODELS_TO_RUN = list(ABLATIONS.keys())
TRAINING_SEEDS = [42]
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

        from src.models.common.sdlg_hybrid import BatchedSdLGHybrid
        VmappedSdLG = nn.vmap(BatchedSdLGHybrid, in_axes=(0, 0, 0, None), out_axes=0,
                              variable_axes={'params': None}, split_rngs={'params': False})
        
        ((X_actor_trans, X_critic_trans), _), new_carry = VmappedSdLG(
            n_layers=self.config["RNN"]["N_LAYERS"],
            d_model=self.config["RNN"]["D_MODEL"],
            d_k=32, d_v=32,
            d_ffc=self.config["RNN"]["D_FFC"], 
            r=ablation_cfg["R"],
            window_size=16, alpha=0.1, eta=0.0,
            use_rope_phase=ablation_cfg["USE_ROPE"],
            use_softplus_denominator=True,
            use_per_rank_norm=True,
            return_split=True
        )(emb_trans, dones_trans, carry, True)
        
        rnn_out = jnp.transpose(X_actor_trans, (1, 0, 2))
        x_actor = x_critic = rnn_out

        action_logits = 5.0 * jnp.tanh(
            jnp.nan_to_num(nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(x_actor), nan=0.0, posinf=5.0,
                           neginf=-5.0))
        critic_values = jnp.nan_to_num(jnp.squeeze(nn.Dense(1, kernel_init=orthogonal(1.0))(x_critic), axis=-1),
                                       nan=0.0, posinf=10.0, neginf=-10.0)

        return action_logits, critic_values, jnp.zeros_like(critic_values), new_carry

@functools.partial(jax.jit, static_argnums=(3,))
def get_max_local_grad(params, probe_obs, probe_carry, network_apply_fn):
    T = probe_obs.shape[0]
    probe_dones = jnp.zeros((T, 1))

    def bottleneck_loss_fn(obs):
        _, critic_values, _, _ = network_apply_fn(params, probe_carry, (obs, probe_dones))
        return jnp.mean(critic_values)

    local_grad = jax.grad(bottleneck_loss_fn)(probe_obs)
    return jnp.max(jnp.abs(local_grad))

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

def run_tmaze_ablations():
    print("====================================================================")
    print("=== TMAZE FACTORIAL EXPERIMENT: RANK VS ROPE                     ===")
    print("====================================================================")

    NUM_ENVS = 1024
    TOTAL_STEPS = 50_000_000
    LOG_INTERVAL = 10
    EARLY_STOP_THRESHOLD = 0.98
    CONSECUTIVE_STEPS = 3

    WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "sdlgHybrid")
    completed = load_completed_runs()

    training_csv = os.path.join(os.path.dirname(current_file_path), "tmaze_factorial_training.csv")
    zs_csv = os.path.join(os.path.dirname(current_file_path), "tmaze_factorial_zeroshot.csv")

    if not os.path.exists(training_csv):
        with open(training_csv, "w", newline="") as f: csv.writer(f).writerow(
            ["H_train", "model", "seed", "step", "mean_return", "train_len", "train_ent", "boundary_bias"])
    if not os.path.exists(zs_csv):
        with open(zs_csv, "w", newline="") as f: csv.writer(f).writerow(
            ["H_train", "model", "seed", "val_H100", "val_H150", "val_H200"])

    import src.trainers.purejax_trainer_boundary
    def patched_flush_carry_state(hstate, last_done):
        return jax.tree_util.tree_map(lambda x: (
                    x * (1.0 - jnp.expand_dims(last_done, axis=tuple(range(1, 1 + x.ndim - last_done.ndim))))).astype(
            x.dtype) if getattr(x, "ndim", 0) > 0 else x, hstate)

    src.trainers.purejax_trainer_boundary.flush_carry_state = patched_flush_carry_state

    import src.models.purejaxrl.wrappers
    src.models.purejaxrl.wrappers.ActorCriticSdLGHybrid = ActorCriticAblation

    H_train = 100
    for model_name in MODELS_TO_RUN:
        for seed in TRAINING_SEEDS:
            run_id = f"{model_name}_H{H_train}_seed_{seed}"

            if run_id in completed:
                print(f"⏩ Skipping {run_id} (Already completed).")
                continue

            jax.clear_caches()
            gc.collect()
            print(f"\n🚀 LAUNCHING TMAZE FACTORIAL: [{run_id}] | SEED = {seed}")

            ablation_cfg = ABLATIONS[model_name]
            NUM_STEPS = ablation_cfg.get("NUM_STEPS", 16)
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

            wandb.init(project=WANDB_PROJECT, group="tmaze_factorial",
                       name=f"{model_name}-H{H_train}-seed-{seed}", config=config, reinit=True)
            network = ActorCriticAblation(2, config)

            def rnn_carry_init(batch_size):
                from src.models.common.sdlg_hybrid import BatchedSdLGHybrid
                base_carry = BatchedSdLGHybrid.initialize_carry(
                    batch_size=batch_size,
                    n_layers=config["RNN"]["N_LAYERS"],
                    d_k=32, d_v=32,
                    r=ablation_cfg["R"],
                    window_size=16
                )
                return jax.tree_util.tree_map(
                    lambda x: jnp.repeat(jnp.expand_dims(x, 0), batch_size, axis=0),
                    base_carry
                )

            rng = jax.random.PRNGKey(seed)
            env = LogWrapper(PassiveTMaze(corridor_length=H_train))
            raw_update_fn, runner_state = make_update(rng, config, env, env.default_params, lambda: network, rnn_carry_init)

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
                    "train_vloss": jnp.mean(m.get("value_loss", jnp.zeros(1))),
                    "boundary_bias": jnp.mean(m.get("boundary_bias", jnp.zeros(1)))
                }

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
                boundary_bias = float(metrics["boundary_bias"])

                wandb.log({"System/SPS": sps, "Train/Mean_Return": train_ret,
                           "Train/Mean_Length": float(metrics["train_len"]),
                           "Train/Policy_Entropy": float(metrics["train_ent"]),
                           "Train/Value_Loss": float(metrics["train_vloss"]),
                           "Diagnostics/Boundary_Bias": boundary_bias}, step=steps_processed)

                with open(training_csv, "a", newline="") as f:
                    csv.writer(f).writerow(
                        [H_train, model_name, seed, steps_processed, train_ret, float(metrics["train_len"]),
                         float(metrics["train_ent"]), boundary_bias])

                print(f"  [JIT {jit_idx + 1:>3}/{TOTAL_JIT_CALLS}] Step: {steps_processed / 1e6:>5.2f}M | Ret: {train_ret:>5.2f} | BBias: {boundary_bias:>6.4f} | SPS: {int(sps):>5}")

                if train_ret >= EARLY_STOP_THRESHOLD:
                    early_stop_counter += 1
                    if early_stop_counter >= CONSECUTIVE_STEPS:
                        print(f"⏹️  [EARLY STOP] Solved H={H_train}! Proceeding to Zero-Shot Generalization Evaluation.")
                        break
                else:
                    early_stop_counter = 0

            print("\n📏 Probing Zero-Shot Generalization Memory Horizons...")
            eval_rng = jax.random.PRNGKey(99)
            zs_results = {}

            for H_eval in [100, 150, 200]:
                sr = run_corridor_eval(eval_rng, runner_state[0].params, network, rnn_carry_init, H_eval)
                zs_results[H_eval] = sr
                print(f"   - Corridor Length H = {H_eval:<3} | Success Rate = {sr * 100:>5.1f}%")
                wandb.log({f"Eval/Success_H{H_eval}": sr}, step=steps_processed)

            with open(zs_csv, "a", newline="") as f:
                csv.writer(f).writerow(
                    [H_train, model_name, seed, zs_results[100], zs_results[150], zs_results[200]])

            print(f"✅ [{run_id}] Run & Evaluation Complete!\n")
            wandb.finish()
            save_completed_run(run_id)
            jax.clear_caches()
            gc.collect()

if __name__ == "__main__":
    os.environ["HSA_ENABLE_SDMA"] = "0"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
    os.environ["WANDB_MODE"] = "offline"
    run_tmaze_ablations()
