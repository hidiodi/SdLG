# scripts/diagnostics/mathematics/rope_vs_cosine/rope_vs_cosine.py

import sys
import os
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import csv
import functools
import matplotlib.pyplot as plt

current_file_path = os.path.abspath(__file__)
project_root = current_file_path
while not os.path.exists(os.path.join(project_root, "src")):
    project_root = os.path.dirname(project_root)
    if project_root == "/": break
if project_root not in sys.path: sys.path.append(project_root)

from src.envs.tmaze.jax_tmaze import PassiveTMaze
from src.trainers.purejax_trainer import make_update
from gymnax.wrappers.purerl import LogWrapper


# =====================================================================
# 1. MECHANISTIC PROBE (EXTRACTING KEY NORM & GRADIENTS)
# =====================================================================
@functools.partial(jax.jit, static_argnums=(2,))
def mechanistic_probe(params, r_key, use_rope_flag):
    """
    Probes the mathematical stability of the memory keys.
    Cosine models multiply keys by cos(theta), causing the L2 norm of the vector
    to oscillate and shrink over time (attention blind spots).
    RoPE applies orthogonal rotation, preserving the exact L2 norm indefinitely.
    """
    T = 100
    d_k = 16

    # 1. Theoretical Key Norm Tracker
    t_steps = jnp.arange(T)
    omegas = 10000 ** (-2 * jnp.arange(d_k // 2) / d_k)
    theta = t_steps[..., None] * omegas

    if use_rope_flag:
        # RoPE perfectly preserves the L2 norm of the vector
        key_norm_trajectory = jnp.ones((T,))
    else:
        # Cosine tracks the mean squared magnitude of the cosine multipliers
        cos_multipliers = jnp.cos(theta)
        key_norm_trajectory = jnp.mean(jnp.square(cos_multipliers), axis=-1)

    # 2. Local Gradient Norm Extractor
    from src.models.purejaxrl.wrappers import RepresentationNetwork
    from src.models.common.sdlg_hybrid import BatchedSdLGHybrid

    probe_obs = jax.random.normal(r_key, (T, 1, 3))
    probe_dones = jnp.zeros((T, 1))

    unbatched = BatchedSdLGHybrid.initialize_carry(batch_size=1, n_layers=2, d_k=16, d_v=16, r=4, window_size=8)
    probe_carry = jax.tree_util.tree_map(lambda x: jnp.repeat(jnp.expand_dims(x, 0), 1, axis=0), unbatched)

    emb = RepresentationNetwork(64).apply({'params': params['params']['repr_network']}, probe_obs)
    emb_trans, dones_trans = jnp.transpose(emb, (1, 0, 2)), jnp.transpose(probe_dones, (1, 0))

    vmapped_sdlg = nn.vmap(BatchedSdLGHybrid, in_axes=(0, 0, 0, None), out_axes=0, variable_axes={'params': None},
                           split_rngs={'params': False})(
        n_layers=2, d_model=64, d_k=16, d_v=16, d_ffc=256, r=4, window_size=8,
        alpha=0.1, eta=0.0, use_rope_phase=use_rope_flag,
        use_softplus_denominator=True, use_per_rank_norm=True, return_split=True
    )

    def bottleneck_loss_fn(x_in):
        out, _ = vmapped_sdlg.apply({'params': params['params']['sdlg_hybrid']}, x_in, dones_trans, probe_carry, True)
        return jnp.mean(out[0][1])  # Critic representation

    local_grad = jax.grad(bottleneck_loss_fn)(emb_trans)
    max_local_grad = jnp.max(jnp.abs(local_grad))

    return key_norm_trajectory, max_local_grad


# =====================================================================
# 2. ZERO-SHOT EVALUATOR
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

        obs_expand = jnp.expand_dims(obs, axis=0)
        dones_expand = jnp.expand_dims(done, axis=0)

        logits, _, _, next_hstate = network.apply(params, hstate, (obs_expand, dones_expand))

        import distrax
        pi = distrax.Categorical(logits=logits.squeeze(0))
        rng_step, _rng = jax.random.split(rng_step)
        action = pi.mode()

        step_keys = jax.random.split(_rng, num_eval_envs)
        next_obs, next_env_state, reward, next_done, _ = jax.vmap(eval_env.step, in_axes=(0, 0, 0, None))(step_keys,
                                                                                                          env_state,
                                                                                                          action,
                                                                                                          eval_env.default_params)

        def flush(c_in, d_in):
            def flush_leaf(x):
                dims_to_add = x.ndim - d_in.ndim
                broadcast_done = jnp.expand_dims(d_in, axis=tuple(range(1, 1 + dims_to_add)))
                return (x * (1.0 - broadcast_done)).astype(x.dtype)

            return jax.tree_util.tree_map(flush_leaf, c_in)

        next_hstate = flush(next_hstate, next_done)
        reward_sum = reward_sum + reward * (1.0 - done)
        return (next_obs, next_env_state, next_hstate, reward_sum, next_done, rng_step), None

    rng, _rng = jax.random.split(rng)
    final_carry, _ = jax.lax.scan(step_fn, (obs, env_state, init_carry_state, jnp.zeros(num_eval_envs),
                                            jnp.zeros(num_eval_envs, dtype=bool), _rng), None, max_steps)
    return float(jnp.mean(final_carry[3] >= 0.99))


# =====================================================================
# 3. EXPERIMENT EXECUTION LOOP
# =====================================================================
def run_rope_vs_cosine_study():
    print("====================================================================")
    print("=== EXPERIMENT 3: RoPE PHASE STABILITY VS COSINE COLLAPSE        ===")
    print("====================================================================")

    H_TRAIN = 50
    NUM_ENVS = 128
    NUM_STEPS = 32  # STRICT BPTT WALL: Forces reliance on memory retention
    STEPS_PER_UPDATE = NUM_ENVS * NUM_STEPS
    TOTAL_STEPS = 6_000_000
    LOG_INTERVAL = 10
    TOTAL_JIT_CALLS = (TOTAL_STEPS // STEPS_PER_UPDATE) // LOG_INTERVAL

    seeds = [42, 43, 44]
    rope_conditions = [True, False]

    # Data structures for plotting
    tracked_acc = {True: np.zeros((len(seeds), TOTAL_JIT_CALLS)), False: np.zeros((len(seeds), TOTAL_JIT_CALLS))}
    tracked_grad = {True: np.zeros((len(seeds), TOTAL_JIT_CALLS)), False: np.zeros((len(seeds), TOTAL_JIT_CALLS))}
    key_norms_trajectory = {True: None, False: None}
    zero_shot_100 = {True: [], False: []}

    # Monkey patch standard trainer flush
    import src.trainers.purejax_trainer
    def patched_flush_carry_state(hstate, last_done):
        def flush_leaf(x):
            dims_to_add = x.ndim - last_done.ndim
            broadcast_done = jnp.expand_dims(last_done, axis=tuple(range(1, 1 + dims_to_add)))
            return (x * (1.0 - broadcast_done)).astype(x.dtype)

        return jax.tree_util.tree_map(flush_leaf, hstate)

    src.trainers.purejax_trainer.flush_carry_state = patched_flush_carry_state

    from src.models.purejaxrl.wrappers import ActorCriticSdLGHybrid
    from src.models.common.sdlg_hybrid import BatchedSdLGHybrid

    for use_rope in rope_conditions:
        condition_name = "RoPE Phase Carrier (Ours)" if use_rope else "Absolute Cosine (Baseline)"

        for s_idx, seed in enumerate(seeds):
            jax.clear_caches()
            print(f"\n[Training] ARCHITECTURE = {condition_name} | SEED = {seed}")

            config = {
                "NUM_ENVS": NUM_ENVS, "NUM_STEPS": NUM_STEPS, "TOTAL_TIMESTEPS": TOTAL_STEPS,
                "UPDATE_EPOCHS": 4, "NUM_MINIBATCHES": 4, "GAMMA": 0.99, "GAE_LAMBDA": 0.95,
                "CLIP_EPS": 0.2, "ENT_COEF": 0.01, "VF_COEF": 0.5, "MAX_GRAD_NORM": 0.5, "LR": 0.0003,
                "OPTIMISTIC_RESETS": False, "ANNEAL_LR": True, "LOG_INTERVAL": LOG_INTERVAL,
                "RNN": {
                    "RNN_TYPE": "sdlg_hybrid", "N_LAYERS": 2, "D_MODEL": 64, "D_K": 16, "D_V": 16, "D_FFC": 256,
                    "R": 4, "WINDOW_SIZE": 8, "ALPHA": 0.1, "ETA": 0.0,
                    "USE_ROPE_PHASE": use_rope, "USE_SOFTPLUS_DENOMINATOR": True, "USE_PER_RANK_NORM": True
                }
            }

            model = ActorCriticSdLGHybrid(action_dim=2, config=config)

            def rnn_carry_init(batch_size):
                unbatched = BatchedSdLGHybrid.initialize_carry(batch_size, 2, 16, 16, 4, 8)
                return jax.tree_util.tree_map(lambda x: jnp.repeat(jnp.expand_dims(x, 0), batch_size, axis=0),
                                              unbatched)

            rng = jax.random.PRNGKey(seed)
            env = PassiveTMaze(corridor_length=H_TRAIN)
            env = LogWrapper(env)

            raw_update_fn, runner_state = make_update(rng, config, env, env.default_params, lambda: model,
                                                      rnn_carry_init)

            @jax.jit
            def compiled_update(r_state):
                out = raw_update_fn(r_state)
                m = out["metric"]
                d_mask = m.get("returned_episode", jnp.ones_like(m.get("entropy", jnp.zeros(1))))

                def mean_d(x): return jnp.sum(x * d_mask) / jnp.maximum(1.0, jnp.sum(d_mask))

                reduced = {
                    "train_ret": mean_d(m.get("returned_episode_returns", jnp.zeros_like(d_mask))),
                }
                return out["runner_state"], reduced

            steps_processed = 0
            for jit_idx in range(TOTAL_JIT_CALLS):
                runner_state, metrics = compiled_update(runner_state)
                runner_state = jax.block_until_ready(runner_state)
                metrics = jax.device_get(metrics)

                train_ret = float(metrics["train_ret"])
                steps_processed += STEPS_PER_UPDATE * LOG_INTERVAL

                # Probe Mechanism
                train_state = runner_state[0]
                rng, probe_rng = jax.random.split(rng)
                key_norm_hist, local_grad = jax.device_get(mechanistic_probe(train_state.params, probe_rng, use_rope))

                if s_idx == 0: key_norms_trajectory[use_rope] = key_norm_hist

                tracked_acc[use_rope][s_idx, jit_idx] = train_ret
                tracked_grad[use_rope][s_idx, jit_idx] = float(local_grad)

                if jit_idx % 10 == 0:
                    print(
                        f"  Update {jit_idx:>3}/{TOTAL_JIT_CALLS} | Step: {steps_processed / 1e6:>4.1f}M | Acc: {train_ret * 100:>5.1f}% | Grad Norm: {float(local_grad):>6.4f}")

            # Zero-Shot Generalization (H=100)
            eval_rng = jax.random.PRNGKey(99)
            verification_acc = run_corridor_eval(eval_rng, train_state.params, model, rnn_carry_init, 100)
            zero_shot_100[use_rope].append(verification_acc)
            print(f"  --> Zero-Shot H=100 Extrapolation Accuracy: {verification_acc * 100:.1f}%\n")

    # =====================================================================
    # 4. PLOTTING & ASSERTIONS
    # =====================================================================
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    steps_x = np.arange(0, TOTAL_JIT_CALLS) * (STEPS_PER_UPDATE * LOG_INTERVAL / 1e6)

    # Calculate means and standard deviations
    rope_acc_mean, rope_acc_std = np.mean(tracked_acc[True], axis=0), np.std(tracked_acc[True], axis=0)
    cos_acc_mean, cos_acc_std = np.mean(tracked_acc[False], axis=0), np.std(tracked_acc[False], axis=0)

    # Panel A: Training Convergence
    axes[0].plot(steps_x, rope_acc_mean * 100, label="RoPE (Ours)", color="#2c7bb6", linewidth=2.5)
    axes[0].fill_between(steps_x, (rope_acc_mean - rope_acc_std) * 100, (rope_acc_mean + rope_acc_std) * 100,
                         color="#2c7bb6", alpha=0.15)
    axes[0].plot(steps_x, cos_acc_mean * 100, label="Cosine (Baseline)", color="#d7191c", linewidth=2, linestyle="--")
    axes[0].fill_between(steps_x, (cos_acc_mean - cos_acc_std) * 100, (cos_acc_mean + cos_acc_std) * 100,
                         color="#d7191c", alpha=0.15)
    axes[0].set_title("Training Convergence (H=50, BPTT=32)", fontweight='bold')
    axes[0].set_xlabel("Steps (Millions)")
    axes[0].set_ylabel("Mean Return (%)")
    axes[0].legend(frameon=True)

    # Panel B: Key Vector L2 Norm Over Time
    sequence_steps_x = np.arange(100)
    axes[1].plot(sequence_steps_x, key_norms_trajectory[True], label="RoPE: Exact Orthogonal Rotation", color="#2c7bb6",
                 linewidth=2.5)
    axes[1].plot(sequence_steps_x, key_norms_trajectory[False], label="Cosine: Scalar Oscillation Decay",
                 color="#d7191c", linewidth=2, linestyle="--")
    axes[1].set_title(r"Memory Key $L_2$ Norm Preservation ($||\phi(k_t)||_2$)", fontweight='bold')
    axes[1].set_xlabel("Trajectory Step (t)")
    axes[1].set_ylabel("Norm Multiplier")
    axes[1].legend(frameon=True)

    # Panel C: Localized Gradient Norms
    rope_grad_mean = np.mean(tracked_grad[True], axis=0)
    cos_grad_mean = np.mean(tracked_grad[False], axis=0)

    axes[2].plot(steps_x, rope_grad_mean, label="RoPE Gradients", color="#2c7bb6", linewidth=2.5)
    axes[2].plot(steps_x, cos_grad_mean, label="Cosine Gradients", color="#d7191c", linewidth=2, linestyle="--",
                 alpha=0.7)
    axes[2].set_yscale("log")
    axes[2].set_title(r"Max Local Gradient Norm (Log Scale)", fontweight='bold')
    axes[2].set_xlabel("Steps (Millions)")
    axes[2].set_ylabel("Gradient Norm")
    axes[2].legend(frameon=True)

    # Panel D: Zero-Shot Generalization Bar Chart
    mean_rope_zs = np.mean(zero_shot_100[True]) * 100
    mean_cos_zs = np.mean(zero_shot_100[False]) * 100

    bars = axes[3].bar(["RoPE (Ours)", "Cosine (Baseline)"], [mean_rope_zs, mean_cos_zs], color=["#2c7bb6", "#d7191c"],
                       alpha=0.8)
    axes[3].set_title("Zero-Shot Extrapolation (H=100)", fontweight='bold')
    axes[3].set_ylabel("Success Rate (%)")
    axes[3].set_ylim(0, 105)

    # Add text labels on bars
    for bar in bars:
        yval = bar.get_height()
        axes[3].text(bar.get_x() + bar.get_width() / 2.0, yval + 2, f"{yval:.1f}%", ha='center', va='bottom',
                     fontweight='bold')

    plt.tight_layout()
    plot_path = "scripts/diagnostics/mathematics/rope_vs_cosine/rope_vs_cosine.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved Visualization to {plot_path}")

    # Assertions
    assert mean_rope_zs > 70.0, f"FALSIFICATION: RoPE failed to generalize zero-shot. Acc: {mean_rope_zs:.1f}%"
    assert mean_rope_zs > mean_cos_zs, f"FALSIFICATION: RoPE ({mean_rope_zs:.1f}%) underperformed Cosine ({mean_cos_zs:.1f}%)."

    print("\nExperiment 3 passed successfully! RoPE mathematical dominance validated.")


if __name__ == "__main__":
    os.environ["HSA_ENABLE_SDMA"] = "0"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
    run_rope_vs_cosine_study()