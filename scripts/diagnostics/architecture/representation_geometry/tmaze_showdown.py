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
from src.models.common.sdlg_hybrid import (
    BatchedSdLGHybrid,
    GatedAttentionGLALayer,
    rope_rotate,
    parallel_recurrent_scan
)
from src.trainers.purejax_trainer import Transition, flush_carry_state
import src.trainers.purejax_trainer as trainer_module

from craftax.craftax.constants import Achievement

ACH_KEYS = [ach.name for ach in Achievement]

# =====================================================================
# --- DYNAMIC FREQUENCY SCALING PATCH ---
# We intercept GatedAttentionGLALayer.__call__ globally in-memory
# to scale frequencies statically without mutating sdlg_hybrid.py
# =====================================================================
GLOBAL_FREQ_SCALE = 1.0  # Will be dynamically updated per model


def patched_gla_call(self, inputs, terminations, last_memory, carry_step_count):
    T, _ = inputs.shape
    tilde_k_prev, tilde_v_prev, theta_prev = last_memory

    # Dynamic frequency scaling applied cleanly at compile-time
    omegas = jnp.linspace(-jnp.pi, jnp.pi, self.r) * GLOBAL_FREQ_SCALE

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
    joint_gla = nn.Dense(total_gla_dims, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)),
                         bias_init=nn.initializers.constant(0.0),
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

    attn_out = nn.Dense(self.d_model, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)),
                        bias_init=nn.initializers.constant(0.0),
                        name="out_proj")(attn_out)
    g = jax.nn.sigmoid(
        nn.Dense(self.d_model, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)),
                 bias_init=nn.initializers.constant(0.0), name="g_proj")(inputs))
    out = g * attn_out

    new_tilde_k = final_tilde_k[-1]
    new_tilde_v = final_tilde_v[-1]
    new_theta = theta[-1]
    new_step_count = step_counts[-1]

    return out, intrinsic_bonus, (new_tilde_k, new_tilde_v, new_theta), new_step_count


# Inject the patch into memory
GatedAttentionGLALayer.__call__ = patched_gla_call
print("[SYSTEM] GLA dynamic patch applied safely.", flush=True)

# =====================================================================
# --- EXPERIMENT 12 T-MAZE CONFIGURATION ---
# =====================================================================
TOTAL_STEPS = 50_000_000
TRAIN_LOG_INTERVAL_STEPS = 500_000

MODELS_TO_RUN = ["SdLGHybrid-NoRoPE", "SdLGHybrid-RoPE-Standard", "SdLGHybrid-RoPE-Bounded"]
TRAINING_SEEDS = [42, 54, 99, 123, 456]

WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "SdLGHybrid-Exp12-TMaze")


# =====================================================================

# =====================================================================
# 2. CUSTOM PPO TRAINER
# =====================================================================
def make_custom_update(rng, config, env, env_params, network, rnn_carry_init):
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    config["MINIBATCH_SIZE"] = config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]

    env = BatchEnvWrapper(env, config["NUM_ENVS"])

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    rng, _rng = jax.random.split(rng)
    init_x = (jnp.zeros((1, config["NUM_ENVS"], *env.observation_space(env_params).shape)),
              jnp.zeros((1, config["NUM_ENVS"])))
    init_hstate = rnn_carry_init(config["NUM_ENVS"])
    network_params = network.init(_rng, init_hstate, init_x)

    tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                     optax.adam(learning_rate=linear_schedule, eps=1e-5))
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

                    tikhonov_loss = 0.0
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
# 3. BENCHMARK RUNNER
# =====================================================================
def run_experiment12():
    print("====================================================================", flush=True)
    print("=== EXPERIMENT 12: TMAZE REPRESENTATION GEOMETRY SHOWDOWN        ===", flush=True)
    print("====================================================================", flush=True)

    ENV_NAME = "tmaze"
    NUM_ENVS = 256
    NUM_STEPS = 64  # Enforce strict 64-step horizon (no horizon cheat)

    STEPS_PER_UPDATE = NUM_ENVS * NUM_STEPS
    TOTAL_UPDATES = TOTAL_STEPS // STEPS_PER_UPDATE
    LOG_INTERVAL_UPDATES = TRAIN_LOG_INTERVAL_STEPS // STEPS_PER_UPDATE

    # Dynamic path logic: Force CSV and Plot to save alongside the .py file
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    csv_path = "scripts/diagnostics/architecture/representation_geometry/tmaze_showdown.csv"

    print("[SYSTEM] Initializing CSV file...", flush=True)
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "Architecture", "Seed", "Total_Steps", "Wall_Clock_Time", "SPS",
        "Train_Return", "Train_Entropy", "Train_VLoss_Var"
    ])
    csv_file.flush()

    # Apply hybrid network configuration directly
    trainer_module.ActorCriticRNN = ActorCriticSdLGHybrid

    for model_name in MODELS_TO_RUN:
        use_rope = (model_name != "SdLGHybrid-NoRoPE")

        # --- THE FREQUENCY SCALING TOGGLE ---
        global GLOBAL_FREQ_SCALE
        if model_name == "SdLGHybrid-RoPE-Bounded":
            GLOBAL_FREQ_SCALE = 0.01  # Dynamically scale clock speed down [3]
            print(f"  [SYSTEM] Scaling Rotary clock to 1/MEMORY_LEN (0.01) to prevent Nyquist aliasing [3].",
                  flush=True)
        else:
            GLOBAL_FREQ_SCALE = 1.0  # Standard full rotation speed

        print(f"\n[SYSTEM] Configuring Model: {model_name} (RoPE={use_rope}, ClockScale={GLOBAL_FREQ_SCALE})",
              flush=True)

        # T-Maze uses a highly lightweight, fast-compiling sequence stack
        config = {
            "NUM_ENVS": NUM_ENVS, "NUM_STEPS": NUM_STEPS, "TOTAL_TIMESTEPS": TOTAL_STEPS,
            "UPDATE_EPOCHS": 4, "NUM_MINIBATCHES": 8, "GAMMA": 0.99, "GAE_LAMBDA": 0.95,
            "CLIP_EPS": 0.2, "ENT_COEF": 0.01, "VF_COEF": 0.5, "MAX_GRAD_NORM": 0.5,
            "LR": 0.0003, "OPTIMISTIC_RESETS": False, "ANNEAL_LR": True,
            "LOG_INTERVAL": LOG_INTERVAL_UPDATES,
            "USE_TIKHONOV": False,
            "RNN": {
                "RNN_TYPE": "sdlg_hybrid",
                "N_LAYERS": 2, "D_MODEL": 64, "D_HEAD": 16, "N_HEADS": 2,
                "MEMORY_LEN": 64, "D_K": 16, "D_V": 16, "D_FFC": 256,
                "R": 4, "WINDOW_SIZE": 8, "ALPHA": 0.1, "ETA": 0.0,
                "USE_ROPE_PHASE": use_rope,
                "USE_SOFTPLUS_DENOMINATOR": True,
                "USE_PER_RANK_NORM": True
            }
        }

        static_env = StaticEnvConfig(env_name=ENV_NAME, num_envs=NUM_ENVS, num_steps=50, max_episode_steps=100,
                                     optimistic_resets=False)
        dynamic_train = DynamicTrainingConfig(
            total_timesteps=TOTAL_STEPS, learning_rate=0.0003, gamma=0.99, gae_lambda=0.95,
            clip_eps=0.2, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, update_epochs=4,
            minibatch_size=1024, log_interval=LOG_INTERVAL_UPDATES
        )

        env, env_params = build_purejaxrl_env(static_env, dynamic_train)
        from src.envs.wrappers import LogWrapper
        if not isinstance(env, LogWrapper):
            env = LogWrapper(env)
        action_dim = env.action_space(env_params).n

        network = ActorCriticSdLGHybrid(action_dim=action_dim, config=config)

        def rnn_carry_init(batch_size):
            unbatched = BatchedSdLGHybrid.initialize_carry(
                batch_size=batch_size, n_layers=2, d_k=16, d_v=16, r=4, window_size=8)
            return jax.tree_util.tree_map(lambda x: jnp.repeat(jnp.expand_dims(x, 0), batch_size, axis=0), unbatched)

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
            print(f"[{model_name}] Building PPO Update Graph...", flush=True)
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
                    "train_tikhonov": jnp.mean(m.get("tikhonov_loss", jnp.zeros(1))),  # <--- ADD THIS LINE
                }
                return out["runner_state"], reduced

            # --- SAFE CRATER LOOP BOUNDING ---
            NUM_LOG_LOOPS = TOTAL_UPDATES // LOG_INTERVAL_UPDATES

            consecutive_solves = 0
            steps_processed = 0
            start_wall_time = time.time()
            cumulative_eval_time = 0.0

            print(f"[{model_name}] Starting 1st Heavy XLA Compilation (Wait ~15-30 seconds)...", flush=True)
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

                print(
                    f"  [TRAIN] Step: {steps_processed / 1e6:>5.1f}M | Ret: {train_ret:>5.2f} (Accuracy) | SPS: {int(sps):>5}",
                    flush=True)

                csv_writer.writerow([
                    model_name, seed, steps_processed, active_train_time, sps,
                    train_ret, train_ent, train_vloss_var, train_tik_loss
                ])
                csv_file.flush()

                wandb.log(wandb_log_dict, step=steps_processed)

                # --- EARLY STOPPING SOLVE GUARD ---
                # If training accuracy is >= 99% for two consecutive checkpoints, we consider the stage solved
                if train_ret >= 0.99:
                    consecutive_solves += 1
                else:
                    consecutive_solves = 0

                if consecutive_solves >= 2:
                    print(
                        f"  --> [SOLVED] {model_name} solved the 50-step T-Maze in {steps_processed:,} steps! Exiting seed early.",
                        flush=True)
                    break

                del wandb_log_dict
                gc.collect()

            print(f"[{model_name}] Run Complete. Closing W&B.", flush=True)
            wandb.finish()
            jax.clear_caches()
            gc.collect()

    csv_file.close()


# =====================================================================
# 4. VISUALIZATION
# =====================================================================
def plot_experiment12():
    print("\n[SYSTEM] Generating Scientific Plots...", flush=True)
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    csv_path = "scripts/diagnostics/architecture/representation_geometry/tmaze_showdown.csv"
    if not os.path.exists(csv_path):
        return

    plt.close('all')
    plt.clf()
    plt.cla()

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    colors = {"SdLGHybrid-NoRoPE": "#d7191c", "SdLGHybrid-RoPE-Standard": "#e6ab02",
              "SdLGHybrid-RoPE-Bounded": "#2c7bb6"}
    styles = {"SdLGHybrid-NoRoPE": "--", "SdLGHybrid-RoPE-Standard": "-.", "SdLGHybrid-RoPE-Bounded": "-"}

    for model_name in MODELS_TO_RUN:
        model_data = df[df["Architecture"] == model_name]
        if model_data.empty: continue

        grp = model_data.groupby("Total_Steps")
        mean_df = grp.mean(numeric_only=True)
        steps_train = mean_df.index.values / 1e6

        # Panel A: Train Return (Continuous learning curve showing accurate % retrieval)
        axes[0, 0].plot(steps_train, mean_df["Train_Return"].values, label=model_name, color=colors[model_name],
                        linestyle=styles[model_name], linewidth=2.5)

        # Panel B: Final Asymptotic Retrieval Bar Summary
        axes[0, 1].bar([model_name], [mean_df["Train_Return"].values[-1]], color=colors[model_name], alpha=0.8)

        # Panel C: Value Loss Volatility
        vloss_var = mean_df["Train_VLoss_Var"].values
        axes[1, 0].plot(steps_train, vloss_var, label=model_name, color=colors[model_name],
                        linestyle=styles[model_name], linewidth=1.5, alpha=0.8)

        # Panel D: Empty placeholder or SPS
        sps = mean_df["SPS"].values
        axes[1, 1].plot(steps_train, sps, label=model_name, color=colors[model_name], linestyle=styles[model_name],
                        linewidth=2.5)

    axes[0, 0].set_title("Training Progression: Cue Retrieval Accuracy", fontweight='bold')
    axes[0, 0].set_ylabel("Mean Accuracy (% Correct)")

    axes[0, 1].set_title("Final Asymptotic Retrieval Capability", fontweight='bold')
    axes[0, 1].set_ylabel("Final Accuracy")

    axes[1, 0].set_title(r"Optimization: Value Loss Variance ($\sigma^2_{VF}$)", fontweight='bold')
    axes[1, 0].set_ylabel("Variance (Log Scale)")
    axes[1, 0].set_yscale('log')

    axes[1, 1].set_title(r"Hardware Efficiency: Throughput ($\eta_{SPS}$)", fontweight='bold')
    axes[1, 1].set_ylabel("Steps Per Second (SPS)")

    for ax in axes.flatten():
        ax.set_xlabel("Environment Steps (Millions)")
        ax.legend(frameon=True)

    plt.tight_layout()
    plot_path = "scripts/diagnostics/architecture/representation_geometry/tmaze_showdown.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close('all')
    print(f"\n[SYSTEM] Rendered 4-Panel Scientific Matrix to {plot_path}", flush=True)

    # --- WANDB PLOT UPLOAD ---
    print("[SYSTEM] Uploading Summary Plot to W&B...", flush=True)
    wandb_project = os.environ.get("WANDB_PROJECT", "SdLGHybrid-Exp12-TMaze")
    init_wandb(
        project=wandb_project,
        group="Summary",
        name="Exp12-TMaze-Final-Plot",
        tags=["summary", "plot"],
        notes="Auto-generated T-Maze 3-model frequency ablation matrix."
    )
    wandb.log({"Final_TMaze_Matrix": wandb.Image(plot_path)})
    wandb.finish()
    print("[SYSTEM] Upload complete! TMaze Showdown fully finished.", flush=True)


if __name__ == "__main__":
    run_experiment12()
    plot_experiment12()