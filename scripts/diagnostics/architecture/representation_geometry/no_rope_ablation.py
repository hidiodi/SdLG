import sys
import os
import time

# Force JAX to be gentle on Host RAM during XLA compilation
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

print("[SYSTEM] Importing libraries...", flush=True)
import jax
import jax.numpy as jnp
import flax.linen as nn
import flax
import numpy as np
import pandas as pd
import csv
import functools
import matplotlib.pyplot as plt

import gc
import optax
from typing import Any
from flax.training.train_state import TrainState



import sys
import os
current_file_path = os.path.abspath(__file__)
project_root = current_file_path
while not os.path.exists(os.path.join(project_root, "src")):
    project_root = os.path.dirname(project_root)
    if project_root == "/":
        break
if project_root not in sys.path:
    sys.path.append(project_root)

import wandb
from scripts.diagnostics.core.utils import init_wandb
from src.envs.builder import build_purejaxrl_env
from src.envs.wrappers import OptimisticResetVecEnvWrapper, BatchEnvWrapper
from src.utils.config_schema import StaticEnvConfig, DynamicTrainingConfig
import distrax

from src.models.purejaxrl.wrappers import RepresentationNetwork, ActorCriticSdLGHybrid
from src.models.common.sdlg_hybrid import BatchedSdLGHybrid
from src.trainers.purejax_trainer import Transition, flush_carry_state

from craftax.craftax.constants import Achievement

ACH_KEYS = [ach.name for ach in Achievement]

print("[SYSTEM] Imports complete.", flush=True)

# =====================================================================
# --- EXPERIMENT 12 CONFIGURATION ---
# =====================================================================
TOTAL_STEPS = 15_000_000  # 15 Million steps (Strategic Milestone window)
TRAIN_LOG_INTERVAL_STEPS = 500_000
EVAL_INTERVAL_STEPS = TOTAL_STEPS  # Evaluate ONLY at the very end

MODELS_TO_RUN = ["SdLGHybrid-NoRoPE", "SdLGHybrid-Tikhonov", "SdLGHybrid-RoPE"]
TRAINING_SEEDS = [42, 54, 99, 123, 456]

# Centralized WandB Configuration from run_all.py
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "SdLGHybrid-Exp12-Representation")


# =====================================================================

# =====================================================================
# 1. CUSTOM PPO TRAINER (WITH TIKHONOV INJECTION)
# =====================================================================
def make_custom_update(rng, config, env, env_params, network, rnn_carry_init):
    """
    A standalone PPO trainer that intercepts the GLA memory states (\tilde{k})
    and applies a Tikhonov Regularization penalty to prevent rank collapse algebraically.
    """
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    config["MINIBATCH_SIZE"] = config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]

    if config.get('OPTIMISTIC_RESETS', False):
        env = OptimisticResetVecEnvWrapper(env, config["NUM_ENVS"], reset_ratio=min(16, config["NUM_ENVS"]))
    else:
        env = BatchEnvWrapper(env, config["NUM_ENVS"])

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    rng, _rng = jax.random.split(rng)
    init_x = (jnp.zeros((1, config["NUM_ENVS"], *env.observation_space(env_params).shape)),
              jnp.zeros((1, config["NUM_ENVS"])))
    init_hstate = rnn_carry_init(config["NUM_ENVS"])
    network_params = network.init(_rng, init_hstate, init_x)

    if config.get("ANNEAL_LR", True):
        tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                         optax.adam(learning_rate=linear_schedule, eps=1e-5))
    else:
        tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(config["LR"], eps=1e-5))

    train_state = TrainState.create(apply_fn=network.apply, params=network_params, tx=tx)

    rng, _rng = jax.random.split(rng)
    obsv, env_state = env.reset(_rng, env_params)
    init_hstate = rnn_carry_init(config["NUM_ENVS"])

    runner_state = (train_state, env_state, obsv, jnp.zeros((config["NUM_ENVS"]), dtype=bool), init_hstate, rng)

    def _update_step(runner_state, unused):
        def _env_step(runner_state, unused):
            train_state, env_state, last_obs, last_done, hstate, rng = runner_state
            rng, _rng = jax.random.split(rng)

            hstate = flush_carry_state(hstate, last_done)

            ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
            action_logits, value, bonus, hstate = network.apply(train_state.params, hstate, ac_in)
            pi = distrax.Categorical(logits=action_logits)
            intrinsic_bonus = bonus.squeeze(0)

            action = pi.sample(seed=_rng)
            log_prob = pi.log_prob(action)
            value, action, log_prob = value.squeeze(0), action.squeeze(0), log_prob.squeeze(0)

            rng, _rng = jax.random.split(rng)
            obsv, env_state, reward, done, info = env.step(_rng, env_state, action, env_params)

            total_reward = reward + intrinsic_bonus
            transition = Transition(last_done, action, value, total_reward, log_prob, last_obs, info)
            return (train_state, env_state, obsv, done, hstate, rng), transition

        initial_hstate = runner_state[4]
        runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, config["NUM_STEPS"])

        train_state, env_state, last_obs, last_done, hstate, rng = runner_state
        ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
        _, last_val, _, _ = network.apply(train_state.params, hstate, ac_in)
        last_val = last_val.squeeze(0)

        def _calculate_gae(traj_batch, last_val, last_done):
            def _get_advantages(carry, transition):
                gae, next_value, next_done = carry
                done, value, reward = transition.done, transition.value, transition.reward
                delta = reward + config["GAMMA"] * next_value * (1 - next_done) - value
                gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - next_done) * gae
                return (gae, value, done), gae

            _, advantages = jax.lax.scan(_get_advantages, (jnp.zeros_like(last_val), last_val, last_done), traj_batch,
                                         reverse=True, unroll=16)
            return advantages, advantages + traj_batch.value

        advantages, targets = _calculate_gae(traj_batch, last_val, last_done)

        def _update_epoch(update_state, unused):
            def _scan_minibatch(train_state, batch_info):
                init_hstate, traj_batch, advantages, targets = batch_info

                def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                    first_state = jax.tree_util.tree_map(lambda x: x[0], init_hstate)

                    # Native extraction of hstate_out
                    action_logits, value, _, hstate_out = network.apply(params, first_state,
                                                                        (traj_batch.obs, traj_batch.done))
                    pi = distrax.Categorical(logits=action_logits)
                    log_prob = pi.log_prob(traj_batch.action)

                    value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(-config["CLIP_EPS"],
                                                                                            config["CLIP_EPS"])
                    value_losses = jnp.square(value - targets)
                    value_losses_clipped = jnp.square(value_pred_clipped - targets)
                    value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                    ratio = jnp.exp(log_prob - traj_batch.log_prob)
                    gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                    loss_actor1 = ratio * gae
                    loss_actor2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * gae
                    loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
                    entropy = pi.entropy().mean()

                    # =================================================================
                    # --- TIKHONOV REGULARIZATION PENALTY ---
                    # =================================================================
                    tikhonov_loss = 0.0
                    if config.get("USE_TIKHONOV", False):
                        for layer_carry in hstate_out:
                            tilde_k = layer_carry.tilde_k
                            r = tilde_k.shape[-2]
                            # Covariance: C = k * k^T
                            C = jnp.einsum('...ri,...ji->...rj', tilde_k, tilde_k)
                            I = jnp.eye(r)
                            tikhonov_loss += jnp.mean(jnp.sum(jnp.square(C - I), axis=(-1, -2)))
                        tikhonov_loss = tikhonov_loss * config.get("TIKHONOV_COEF", 0.01)

                    total_loss = loss_actor + config["VF_COEF"] * value_loss - config[
                        "ENT_COEF"] * entropy + tikhonov_loss
                    return total_loss, (value_loss, loss_actor, entropy, tikhonov_loss)

                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                (total_loss, (value_loss, loss_actor, entropy, tikhonov_loss)), grads = grad_fn(train_state.params,
                                                                                                init_hstate, traj_batch,
                                                                                                advantages, targets)
                train_state = train_state.apply_gradients(grads=grads)
                return train_state, (total_loss, value_loss, loss_actor, entropy, tikhonov_loss)

            (train_state, init_hstate, traj_batch, advantages, targets, rng) = update_state
            rng, _rng = jax.random.split(rng)
            permutation = jax.random.permutation(_rng, config["NUM_ENVS"])
            batch = (init_hstate, traj_batch, advantages, targets)
            shuffled_batch = jax.tree_util.tree_map(lambda x: jnp.take(x, permutation, axis=1), batch)
            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.swapaxes(jnp.reshape(x, [x.shape[0], config["NUM_MINIBATCHES"], -1] + list(x.shape[2:])),
                                       1, 0),
                shuffled_batch
            )

            train_state, metrics = jax.lax.scan(_scan_minibatch, train_state, minibatches)
            return (train_state, init_hstate, traj_batch, advantages, targets, rng), metrics

        init_hstate = jax.tree_util.tree_map(lambda x: x[None, :], initial_hstate)
        update_state = (train_state, init_hstate, traj_batch, advantages, targets, rng)
        update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, config["UPDATE_EPOCHS"])

        train_state = update_state[0]
        rng = update_state[-1]

        metric = traj_batch.info
        metric = dict(metric)
        metric["value_loss"] = loss_info[1]
        metric["entropy"] = loss_info[3]
        metric["tikhonov_loss"] = loss_info[4]

        runner_state = (train_state, env_state, last_obs, last_done, hstate, rng)
        return runner_state, metric

    def update_fn(runner_state):
        runner_state, metric = jax.lax.scan(_update_step, runner_state, None, config["LOG_INTERVAL"])
        return {"runner_state": runner_state, "metric": metric}

    return update_fn, runner_state


# =====================================================================
# 2. VRAM-SAFE CHUNKED DETERMINISTIC EVALUATOR
# =====================================================================
def build_chunked_evaluator(env, env_params, network, config):
    eval_steps = 10000

    @functools.partial(jax.jit, static_argnums=(3,))
    def eval_chunk(params, start_seed, base_rng, chunk_size):
        keys = jax.random.split(jax.random.PRNGKey(start_seed), chunk_size)
        reset_fn = jax.vmap(env.reset, in_axes=(0, None))
        step_fn = jax.vmap(env.step, in_axes=(0, 0, 0, None))

        obs, env_state = reset_fn(keys, env_params)

        unbatched = BatchedSdLGHybrid.initialize_carry(
            batch_size=chunk_size, n_layers=config["RNN"]["N_LAYERS"],
            d_k=config["RNN"]["D_K"], d_v=config["RNN"]["D_V"],
            r=config["RNN"]["R"], window_size=config["RNN"]["WINDOW_SIZE"]
        )
        hstate = jax.tree_util.tree_map(lambda x: jnp.repeat(jnp.expand_dims(x, 0), chunk_size, axis=0), unbatched)

        dones = jnp.zeros((chunk_size,), dtype=bool)
        init_max_ach = {k: jnp.zeros((chunk_size,), dtype=jnp.int32) for k in ACH_KEYS}
        init_max_ret = jnp.full((chunk_size,), -jnp.inf)

        def step_fn_scan(carry, _):
            obs, env_state, hstate, dones, max_ach, max_ret, step_rng = carry
            ac_in = (obs[np.newaxis, :], dones[np.newaxis, :])

            step_rng, action_rng = jax.random.split(step_rng)

            action_logits, _, _, next_hstate = network.apply(params, hstate, ac_in)
            pi = distrax.Categorical(logits=action_logits.squeeze(0))
            action = pi.sample(seed=action_rng)

            rngs = jax.random.split(jax.random.PRNGKey(0), chunk_size)
            next_obs, next_env_state, reward, next_dones, info = step_fn(rngs, env_state, action, env_params)

            next_hstate = flush_carry_state(next_hstate, next_dones)

            curr_ach = next_env_state.env_state.achievements
            new_max_ach = {
                name: jnp.maximum(max_ach[name], curr_ach[:, i].astype(jnp.int32))
                for i, name in enumerate(ACH_KEYS)
            }

            current_returns = jnp.where(next_dones,
                                        info.get("returned_episode_returns", next_env_state.episode_returns),
                                        next_env_state.episode_returns)
            new_max_ret = jnp.maximum(max_ret, current_returns)

            next_carry = (next_obs, next_env_state, next_hstate, next_dones, new_max_ach, new_max_ret, step_rng)
            return next_carry, None

        rng_scan = jax.random.PRNGKey(start_seed)
        init_carry = (obs, env_state, hstate, dones, init_max_ach, init_max_ret, rng_scan)

        final_carry = jax.lax.stop_gradient(jax.lax.scan(step_fn_scan, init_carry, None, length=eval_steps)[0])
        _, _, _, _, final_max_ach, final_max_ret, _ = final_carry

        return final_max_ret, final_max_ach

    return eval_chunk


# =====================================================================
# 3. ABLATION RUNNER
# =====================================================================
def run_experiment12():
    print("====================================================================", flush=True)
    print("=== EXPERIMENT 12: REPRESENTATION GEOMETRY SHOWDOWN              ===", flush=True)
    print("====================================================================", flush=True)

    ENV_NAME = "Craftax-Symbolic-v1"
    NUM_ENVS = 256
    NUM_STEPS = 64

    STEPS_PER_UPDATE = NUM_ENVS * NUM_STEPS
    TOTAL_UPDATES = TOTAL_STEPS // STEPS_PER_UPDATE
    LOG_INTERVAL_UPDATES = TRAIN_LOG_INTERVAL_STEPS // STEPS_PER_UPDATE
    EVAL_EVERY_N_LOGS = max(1, EVAL_INTERVAL_STEPS // TRAIN_LOG_INTERVAL_STEPS)

    # Dynamic path logic: Force CSV and Plot to save alongside the .py file
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    csv_path = "scripts/diagnostics/architecture/representation_geometry/no_rope_ablation.csv"

    print("[SYSTEM] Initializing CSV file...", flush=True)
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    header = [
                 "Architecture", "Seed", "Total_Steps", "Wall_Clock_Time", "SPS",
                 "Train_Return", "Train_Entropy", "Train_VLoss_Var", "Train_Tikhonov_Loss", "Eval_Return",
                 "Eval_React_Achv", "Eval_Strat_Achv"
             ] + [f"Train_{k}" for k in ACH_KEYS] + [f"Eval_{k}" for k in ACH_KEYS]
    csv_writer.writerow(header)
    csv_file.flush()

    for model_name in MODELS_TO_RUN:
        use_rope = (model_name == "SdLGHybrid-RoPE")
        use_tikhonov = (model_name == "SdLGHybrid-Tikhonov")

        print(f"\n[SYSTEM] Configuring Model: {model_name} (RoPE={use_rope}, Tikhonov={use_tikhonov})", flush=True)

        config = {
            "NUM_ENVS": NUM_ENVS, "NUM_STEPS": NUM_STEPS, "TOTAL_TIMESTEPS": TOTAL_STEPS,
            "UPDATE_EPOCHS": 4, "NUM_MINIBATCHES": 8, "GAMMA": 0.99, "GAE_LAMBDA": 0.95,
            "CLIP_EPS": 0.2, "ENT_COEF": 0.01, "VF_COEF": 0.5, "MAX_GRAD_NORM": 0.5,
            "LR": 0.0002, "OPTIMISTIC_RESETS": False, "ANNEAL_LR": True,
            "LOG_INTERVAL": LOG_INTERVAL_UPDATES,
            "USE_TIKHONOV": use_tikhonov, "TIKHONOV_COEF": 0.01,
            "RNN": {
                "RNN_TYPE": "sdlg_hybrid",
                "N_LAYERS": 4, "D_MODEL": 256, "D_HEAD": 32, "N_HEADS": 8,
                "MEMORY_LEN": 128, "D_K": 32, "D_V": 32, "D_FFC": 1024,
                "R": 4, "WINDOW_SIZE": 8, "ALPHA": 0.1, "ETA": 0.0,
                "USE_ROPE_PHASE": use_rope,
                "USE_SOFTPLUS_DENOMINATOR": True,
                "USE_PER_RANK_NORM": True
            }
        }

        static_env = StaticEnvConfig(env_name=ENV_NAME, num_envs=NUM_ENVS, num_steps=NUM_STEPS, max_episode_steps=10000,
                                     optimistic_resets=False)
        dynamic_train = DynamicTrainingConfig(
            total_timesteps=TOTAL_STEPS, learning_rate=0.0002, gamma=0.99, gae_lambda=0.95,
            clip_eps=0.2, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, update_epochs=4,
            minibatch_size=2048, log_interval=LOG_INTERVAL_UPDATES
        )

        env, env_params = build_purejaxrl_env(static_env, dynamic_train)
        from src.envs.wrappers import LogWrapper
        if not isinstance(env, LogWrapper):
            env = LogWrapper(env)
        action_dim = env.action_space(env_params).n

        network = ActorCriticSdLGHybrid(action_dim=action_dim, config=config)

        def rnn_carry_init(batch_size):
            unbatched = BatchedSdLGHybrid.initialize_carry(
                batch_size=batch_size, n_layers=4, d_k=32, d_v=32, r=4, window_size=8)
            return jax.tree_util.tree_map(lambda x: jnp.repeat(jnp.expand_dims(x, 0), batch_size, axis=0), unbatched)

        print(f"[{model_name}] Building Evaluator Graph...", flush=True)
        chunked_eval_fn = build_chunked_evaluator(env, env_params, network, config)

        for seed in TRAINING_SEEDS:
            print(f"[{model_name}] Clearing XLA Caches...", flush=True)
            jax.clear_caches()
            gc.collect()

            print(f"[{model_name}] Initializing W&B (Seed {seed})...", flush=True)
            base_tags = os.environ.get("WANDB_TAGS", "").split(",") if os.environ.get("WANDB_TAGS") else []
            wandb_tags = base_tags + [model_name]

            init_wandb(
                project=WANDB_PROJECT,
                group=model_name,
                name=f"{model_name}-Seed-{seed}",
                tags=wandb_tags,
                notes=os.environ.get("WANDB_NOTES", ""),
                config=config
            )

            rng = jax.random.PRNGKey(seed)
            print(f"[{model_name}] Building Custom PPO Update Graph (with Tikhonov injection)...", flush=True)
            raw_update_fn, runner_state = make_custom_update(rng, config, env, env_params, network, rnn_carry_init)

            @functools.partial(jax.jit, donate_argnums=(0,))
            def compiled_update_and_reduce(r_state):
                out = raw_update_fn(r_state)
                m = out["metric"]

                done_mask = m.get("returned_episode", jnp.ones_like(m.get("entropy", jnp.zeros(1))))

                def get_done_mean(x):
                    return jnp.sum(x * done_mask) / jnp.maximum(1.0, jnp.sum(done_mask))

                reduced = {
                    "train_ret": get_done_mean(m.get("returned_episode_returns", jnp.zeros_like(done_mask))),
                    "train_ent": jnp.mean(m.get("entropy", jnp.zeros(1))),
                    "train_vloss_var": jnp.var(m.get("value_loss", jnp.zeros(1))),
                    "train_tikhonov": jnp.mean(m.get("tikhonov_loss", jnp.zeros(1))),
                }

                native_achievements = out["runner_state"][1].env_state.achievements
                for i, name in enumerate(ACH_KEYS):
                    reduced[f"train_{name}"] = jnp.mean(native_achievements[:, i].astype(jnp.float32))

                return out["runner_state"], reduced

            NUM_LOG_LOOPS = TOTAL_UPDATES // LOG_INTERVAL_UPDATES

            start_wall_time = time.time()
            cumulative_eval_time = 0.0

            print(f"[{model_name}] Starting 1st Heavy XLA Compilation (Wait ~1-3 minutes)...", flush=True)
            for loop_idx in range(NUM_LOG_LOOPS):

                runner_state, reduced_metric = compiled_update_and_reduce(runner_state)
                runner_state = jax.block_until_ready(runner_state)
                reduced_metric = jax.device_get(reduced_metric)

                if loop_idx == 0:
                    print(f"[{model_name}] -> XLA Compilation SUCCESSFUL!", flush=True)

                steps_processed = (loop_idx + 1) * LOG_INTERVAL_UPDATES * STEPS_PER_UPDATE

                train_ret = float(reduced_metric["train_ret"])
                train_ent = float(reduced_metric["train_ent"])
                train_vloss_var = float(reduced_metric["train_vloss_var"])
                train_tik_loss = float(reduced_metric["train_tikhonov"])

                active_train_time = time.time() - start_wall_time - cumulative_eval_time
                sps = steps_processed / max(active_train_time, 1.0)

                wandb_log_dict = {
                    "System/SPS": sps,
                    "System/Wall_Clock_Time": active_train_time,
                    "Train/Mean_Return": train_ret,
                    "Train/Policy_Entropy": train_ent,
                    "Train/Value_Loss_Var": train_vloss_var,
                    "Train/Tikhonov_Penalty": train_tik_loss,
                }

                for k in ACH_KEYS:
                    wandb_log_dict[f"Train_Achv/{k}"] = float(reduced_metric[f"train_{k}"])

                # Evaluate dynamically based on steps
                is_eval_loop = (steps_processed % EVAL_INTERVAL_STEPS == 0) or (loop_idx == NUM_LOG_LOOPS - 1)

                if is_eval_loop:
                    print(f"  [{model_name}] Running Evaluation Chunk...", flush=True)
                    eval_start = time.time()

                    eval_returns = []
                    eval_react_chunks = []
                    eval_strat_chunks = []
                    eval_ach_chunks = {k: [] for k in ACH_KEYS}

                    for chunk in range(8):
                        start_test_seed = 10000 + (chunk * 128)
                        ret_c, max_ach_c = jax.device_get(
                            chunked_eval_fn(runner_state[0].params, start_test_seed, rng, 128)
                        )
                        eval_returns.append(ret_c)

                        react_c = sum(max_ach_c[k] for k in ACH_KEYS[:20])
                        strat_c = sum(max_ach_c[k] for k in ACH_KEYS[40:])
                        eval_react_chunks.append(react_c)
                        eval_strat_chunks.append(strat_c)

                        for k in ACH_KEYS:
                            eval_ach_chunks[k].append(max_ach_c[k])

                    mean_test_ret = float(np.mean(np.concatenate(eval_returns)))
                    mean_test_react = float(np.mean(np.concatenate(eval_react_chunks)))
                    mean_test_strat = float(np.mean(np.concatenate(eval_strat_chunks)))
                    mean_eval_ach = {k: float(np.mean(np.concatenate(eval_ach_chunks[k]))) for k in ACH_KEYS}

                    eval_end = time.time()
                    cumulative_eval_time += (eval_end - eval_start)

                    wandb_log_dict.update({
                        "Eval/Mean_Return": mean_test_ret,
                        "Eval/Reactive_Achievements": mean_test_react,
                        "Eval/Strategic_Achievements": mean_test_strat,
                    })
                    for k in ACH_KEYS:
                        wandb_log_dict[f"Eval_Achv/{k}"] = mean_eval_ach[k]

                    print(
                        f"  [EVAL] Step: {steps_processed / 1e6:>5.1f}M | Ret: {mean_test_ret:>5.1f} | Strat: {mean_test_strat:>4.1f} | SPS: {int(sps):>5}",
                        flush=True)

                    row = [
                        model_name, seed, steps_processed, active_train_time, sps,
                        train_ret, train_ent, train_vloss_var, train_tik_loss, mean_test_ret,
                        mean_test_react, mean_test_strat
                    ]
                    row.extend([float(reduced_metric[f"train_{k}"]) for k in ACH_KEYS])
                    row.extend([mean_eval_ach[k] for k in ACH_KEYS])

                    csv_writer.writerow(row)
                    csv_file.flush()

                    del eval_returns, eval_ach_chunks, eval_react_chunks, eval_strat_chunks, ret_c, max_ach_c

                else:
                    # Pad missing eval columns with 0.0 during standard train steps
                    row = [
                        model_name, seed, steps_processed, active_train_time, sps,
                        train_ret, train_ent, train_vloss_var, train_tik_loss, 0.0,
                        0.0, 0.0
                    ]
                    row.extend([float(reduced_metric[f"train_{k}"]) for k in ACH_KEYS])
                    row.extend([0.0 for k in ACH_KEYS])
                    csv_writer.writerow(row)
                    csv_file.flush()

                    print(
                        f"  [TRAIN] Step: {steps_processed / 1e6:>5.1f}M | Ret: {train_ret:>5.1f} | Tik_Loss: {train_tik_loss:>6.3f} | SPS: {int(sps):>5}",
                        flush=True)

                wandb.log(wandb_log_dict, step=steps_processed)

                del wandb_log_dict
                gc.collect()

            print(f"[{model_name}] Run Complete. Closing W&B.", flush=True)
            wandb.finish()
            jax.clear_caches()
            gc.collect()

    csv_file.close()


# =====================================================================
# 4. VISUALIZATION & WANDB UPLOAD
# =====================================================================
def plot_experiment12():
    print("\n[SYSTEM] Generating Scientific Plots...", flush=True)

    # Dynamic path logic: Force CSV and Plot to load from the directory of this file
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    csv_path = "scripts/diagnostics/architecture/representation_geometry/no_rope_ablation.csv"
    if not os.path.exists(csv_path):
        return

    plt.close('all')
    plt.clf()
    plt.cla()

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    train_ach_cols = [f"Train_{k}" for k in ACH_KEYS]  # NEW: Tracks ALL training achievements dynamically

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    colors = {"SdLGHybrid-NoRoPE": "#d7191c", "SdLGHybrid-Tikhonov": "#4daf4a", "SdLGHybrid-RoPE": "#2c7bb6"}
    styles = {"SdLGHybrid-NoRoPE": "--", "SdLGHybrid-Tikhonov": "-.", "SdLGHybrid-RoPE": "-"}

    for model_name in MODELS_TO_RUN:
        model_data = df[df["Architecture"] == model_name]
        if model_data.empty: continue

        grp = model_data.groupby("Total_Steps")
        mean_df = grp.mean(numeric_only=True)
        steps_train = mean_df.index.values / 1e6

        # Panel A: Train Return (Continuous learning curve)
        axes[0, 0].plot(steps_train, mean_df["Train_Return"].values, label=model_name, color=colors[model_name],
                        linestyle=styles[model_name], linewidth=2.5)

        # Panel B: Train Total Capabilities (Continuous learning curve showing wood-to-iron progress)
        mean_df["Train_Total_Sum"] = mean_df[train_ach_cols].sum(axis=1)  # NEW: Unlocking curves mapped dynamically
        axes[0, 1].plot(steps_train, mean_df["Train_Total_Sum"].values, label=model_name, color=colors[model_name],
                        linestyle=styles[model_name], linewidth=2.5)

        # Panel C: Value Loss Volatility
        vloss_var = mean_df["Train_VLoss_Var"].values
        axes[1, 0].plot(steps_train, vloss_var, label=model_name, color=colors[model_name],
                        linestyle=styles[model_name], linewidth=1.5, alpha=0.8)

        # Panel D: Tikhonov Penalty Tracking
        tik_loss = mean_df["Train_Tikhonov_Loss"].values
        axes[1, 1].plot(steps_train, tik_loss, label=model_name, color=colors[model_name], linestyle=styles[model_name],
                        linewidth=2.5)

    axes[0, 0].set_title("Training Progression: Mean Episode Return", fontweight='bold')
    axes[0, 0].set_ylabel("Mean Episode Return")

    axes[0, 1].set_title("Capabilities: Total Achievements Unlocked", fontweight='bold')
    axes[0, 1].set_ylabel("Sum Achievements Unlocked")

    axes[1, 0].set_title(r"Optimization: Value Loss Variance ($\sigma^2_{VF}$)", fontweight='bold')
    axes[1, 0].set_ylabel("Variance (Log Scale)")
    axes[1, 0].set_yscale('log')

    axes[1, 1].set_title(r"Orthogonality Penalty ($\mathcal{L}_{Tikhonov}$)", fontweight='bold')
    axes[1, 1].set_ylabel("Penalty Magnitude")

    for ax in axes.flatten():
        ax.set_xlabel("Environment Steps (Millions)")
        ax.legend(frameon=True)

    plt.tight_layout()
    plot_path = "scripts/diagnostics/architecture/representation_geometry/no_rope_ablation.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close('all')
    print(f"\n[SYSTEM] Rendered 4-Panel Scientific Matrix to {plot_path}", flush=True)

    # --- WANDB PLOT UPLOAD ---
    print("[SYSTEM] Uploading Summary Plot to W&B...", flush=True)
    wandb_project = os.environ.get("WANDB_PROJECT", "SdLGHybrid-Exp12-Representation")
    init_wandb(
        project=wandb_project,
        group="Summary",
        name="Exp12-Final-Plot",
        tags=["summary", "plot"],
        notes="Auto-generated multi-model comparison matrix."
    )
    wandb.log({"Final_Comparison_Matrix": wandb.Image(plot_path)})
    wandb.finish()
    print("[SYSTEM] Upload complete! Experiment fully finished.", flush=True)


if __name__ == "__main__":
    run_experiment12()
    plot_experiment12()