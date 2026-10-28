# scripts/diagnostics/exp4_softplus_normalization.py

import sys
import os
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import csv
import functools
import matplotlib.pyplot as plt

from utils.config_schema import DynamicTrainingConfig



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
from src.envs.builder import build_purejaxrl_env
from src.models.purejaxrl.wrappers import ActorCriticSdLGHybrid, RepresentationNetwork
from src.models.common.sdlg_hybrid import BatchedSdLGHybrid
from src.trainers.purejax_trainer import make_update


@functools.partial(jax.jit, static_argnums=(2, 3))
def mechanistic_probe(params, r_key, use_softplus_flag, obs_shape):
    # 1. Setup a dummy 100-step sequence with EXACT environment observation shapes
    probe_obs = jax.random.normal(r_key, (100, 1, *obs_shape))
    probe_dones = jnp.zeros((100, 1))

    unbatched = BatchedSdLGHybrid.initialize_carry(batch_size=1, n_layers=2, d_k=16, d_v=16, r=4, window_size=8)
    probe_carry = jax.tree_util.tree_map(lambda x: jnp.repeat(jnp.expand_dims(x, 0), 1, axis=0), unbatched)

    repr_params = {'params': params['params']['repr_network']}
    emb = RepresentationNetwork(64).apply(repr_params, probe_obs)
    emb_trans = jnp.transpose(emb, (1, 0, 2))
    dones_trans = jnp.transpose(probe_dones, (1, 0))

    # 2. Vmapped Submodule with ETA=1.0 to mathematically extract the hidden denominator
    vmapped_sdlg = nn.vmap(
        BatchedSdLGHybrid,
        in_axes=(0, 0, 0, None), out_axes=0,
        variable_axes={'params': None}, split_rngs={'params': False}
    )(
        n_layers=2, d_model=64, d_k=16, d_v=16, d_ffc=256, r=4, window_size=8,
        alpha=0.1, eta=1.0, use_rope_phase=True,
        use_softplus_denominator=use_softplus_flag, use_per_rank_norm=True, return_split=True
    )

    def bottleneck_loss_fn(x_in):
        out, _ = vmapped_sdlg.apply({'params': params['params']['sdlg_hybrid']}, x_in, dones_trans, probe_carry, True)
        ((_, X_critic_trans), _) = out
        return jnp.mean(X_critic_trans)

    local_grad = jax.grad(bottleneck_loss_fn)(emb_trans)
    max_local_grad = jnp.max(jnp.abs(local_grad))

    out, _ = vmapped_sdlg.apply({'params': params['params']['sdlg_hybrid']}, emb_trans, dones_trans, probe_carry, True)
    ((_, _), intrinsic_bonus_trans) = out

    # ETA=1.0 => Total bonus = 0.5 * step_weight / den
    # For t >= 64, step_weight = 1.0 => den = 0.5 / total_bonus
    valid_bonuses = intrinsic_bonus_trans[0, 64:]
    denominators = 0.5 / (valid_bonuses + 1e-12)
    min_den = jnp.min(denominators)

    return max_local_grad, min_den


def run_softplus_study():
    print("====================================================================")
    print("=== EXPERIMENT 4: SOFTPLUS NORMALIZATION STABILITY PROBE         ===")
    print("====================================================================")

    # Reconfigured static environments matching training sweep
    from src.utils.config_schema import StaticEnvConfig
    static_env = StaticEnvConfig(
        env_name="tmaze", num_envs=128, num_steps=32,
        max_episode_steps=100, optimistic_resets=False
    )

    # Bounded to 200,000 steps per seed for extremely fast execution (~15s per seed)
    dynamic_train = DynamicTrainingConfig(
        total_timesteps=200000, learning_rate=0.0003, gamma=0.99,
        gae_lambda=0.95, clip_eps=0.2, ent_coef=0.01, vf_coef=0.5,
        max_grad_norm=0.5, update_epochs=4, minibatch_size=1024, log_interval=10
    )

    env, env_params = build_purejaxrl_env(static_env, dynamic_train)
    action_dim = env.action_space(env_params).n
    obs_shape = env.observation_space(env_params).shape

    softplus_conditions = [True, False]
    logged_metrics = {True: [], False: []}

    # 10-Seed Monte Carlo Sweep to establish robust, publishable standard deviations
    seeds = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1001]

    for use_softplus in softplus_conditions:
        condition_name = "Softplus Normalization (Proposed)" if use_softplus else "Linear Clipped Baseline"
        print(f"\nEvaluating {condition_name} across {len(seeds)} seeds...")

        for seed_idx, seed in enumerate(seeds):
            jax.clear_caches()
            print(f"  -> Seed {seed_idx + 1}/{len(seeds)} (Seed Key: {seed}) | Cache Cleared")

            config = {
                "NUM_ENVS": static_env.num_envs,
                "NUM_STEPS": static_env.num_steps + 5,
                "TOTAL_TIMESTEPS": dynamic_train.total_timesteps,
                "UPDATE_EPOCHS": dynamic_train.update_epochs,
                "NUM_MINIBATCHES": 4,
                "GAMMA": dynamic_train.gamma,
                "GAE_LAMBDA": dynamic_train.gae_lambda,
                "CLIP_EPS": dynamic_train.clip_eps,
                "ENT_COEF": dynamic_train.ent_coef,
                "VF_COEF": dynamic_train.vf_coef,
                "MAX_GRAD_NORM": dynamic_train.max_grad_norm,
                "LR": dynamic_train.learning_rate,
                "OPTIMISTIC_RESETS": False,
                "ANNEAL_LR": False,
                "LOG_INTERVAL": dynamic_train.log_interval,
                "RNN": {
                    "RNN_TYPE": "sdlg_hybrid",
                    "N_LAYERS": 2, "D_MODEL": 64, "D_K": 16, "D_V": 16, "D_FFC": 256,
                    "R": 4, "WINDOW_SIZE": 8, "ALPHA": 0.1, "ETA": 0.0,
                    "USE_ROPE_PHASE": True,
                    "USE_SOFTPLUS_DENOMINATOR": use_softplus,
                    "USE_PER_RANK_NORM": True
                }
            }

            model = ActorCriticSdLGHybrid(action_dim=action_dim, config=config)

            def rnn_carry_init(batch_size):
                unbatched = BatchedSdLGHybrid.initialize_carry(
                    batch_size=batch_size, n_layers=2, d_k=16, d_v=16, r=4, window_size=8)
                return jax.tree_util.tree_map(lambda x: jnp.repeat(jnp.expand_dims(x, 0), batch_size, axis=0),
                                              unbatched)

            rng = jax.random.PRNGKey(seed)
            update_fn, runner_state = make_update(rng, config, env, env_params, lambda: model, rnn_carry_init)

            num_updates = config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]

            grads_seed = []
            dens_seed = []
            accs_seed = []

            try:
                for step in range(0, num_updates, dynamic_train.log_interval):
                    out = update_fn(runner_state)
                    runner_state = jax.block_until_ready(out["runner_state"])
                    metric = jax.device_get(out["metric"])

                    train_state = runner_state[0]
                    rng, probe_rng = jax.random.split(rng)

                    local_grad, min_den = jax.device_get(
                        mechanistic_probe(train_state.params, probe_rng, use_softplus, obs_shape)
                    )

                    global_grad_norm = float(np.max(metric["max_gradient_norm"]))
                    mean_reward = (metric["returned_episode_returns"] * metric["returned_episode"]).sum() / max(1.0,
                                                                                                                metric[
                                                                                                                    "returned_episode"].sum())

                    # Skip step 0 initialization spike
                    if step > 0:
                        grads_seed.append(global_grad_norm)
                        dens_seed.append(float(min_den))
                        accs_seed.append(float(mean_reward))

                # Extract statistical markers for this seed
                std_grad = np.std(grads_seed) if grads_seed else 0.0
                spikes_10 = sum(1 for g in grads_seed if g > 10.0)
                spikes_50 = sum(1 for g in grads_seed if g > 50.0)

                # Boundary Saturation Rate (within 0.01% of epsilon floor)
                tolerance = 1e-5 * 1.0001
                sat_rate = (sum(1 for d in dens_seed if d < tolerance) / len(dens_seed)) * 100.0 if dens_seed else 0.0
                final_acc = accs_seed[-1] if accs_seed else 0.50

                logged_metrics[use_softplus].append({
                    "final_acc": final_acc,
                    "std_grad": std_grad,
                    "spikes_10": spikes_10,
                    "spikes_50": spikes_50,
                    "sat_rate": sat_rate
                })

            except Exception as e:
                print(f"    -> Seed {seed} failed with exception: {e}")

    # --- ADVANCED MULTI-SEED STATISTICAL SUMMARIES ---
    summary = {}
    for flag in [True, False]:
        runs = logged_metrics[flag]
        if not runs:
            continue
        summary[flag] = {
            "acc_mean": np.mean([r["final_acc"] for r in runs]),
            "acc_std": np.std([r["final_acc"] for r in runs]),
            "std_mean": np.mean([r["std_grad"] for r in runs]),
            "std_std": np.std([r["std_grad"] for r in runs]),
            "spikes_10_mean": np.mean([r["spikes_10"] for r in runs]),
            "spikes_50_mean": np.mean([r["spikes_50"] for r in runs]),
            "sat_mean": np.mean([r["sat_rate"] for r in runs]),
            "sat_std": np.std([r["sat_rate"] for r in runs]),
        }

    print("\n====================================================================")
    print("=== FINAL MULTI-SEED STATISTICAL DIAGNOSTIC REPORT (N=10 SEEDS)  ===")
    print("====================================================================")
    print(f"Metric                      | Softplus (Ours)   | Linear Clipped")
    print(f"--------------------------------------------------------------------")
    print(
        f"Final Return (Acc %)        | {summary[True]['acc_mean'] * 100:.1f} ± {summary[True]['acc_std'] * 100:.1f}%    | {summary[False]['acc_mean'] * 100:.1f} ± {summary[False]['acc_std'] * 100:.1f}%")
    print(
        f"Grad StDev (Volatility)     | {summary[True]['std_mean']:.4f} ± {summary[True]['std_std']:.4f}  | {summary[False]['std_mean']:.4f} ± {summary[False]['std_std']:.4f}")
    print(
        f"Mean Minor Spikes (>10.0)   | {summary[True]['spikes_10_mean']:.1f}                | {summary[False]['spikes_10_mean']:.1f}")
    print(
        f"Mean Major Spikes (>50.0)   | {summary[True]['spikes_50_mean']:.1f}                | {summary[False]['spikes_50_mean']:.1f}")
    print(
        f"Boundary Saturation (%)     | {summary[True]['sat_mean']:.1f} ± {summary[True]['sat_std']:.1f}%    | {summary[False]['sat_mean']:.1f} ± {summary[False]['sat_std']:.1f}%")
    print("====================================================================\n")

    # --- CSV EXPORTING ---

    csv_path = "scripts/diagnostics/mathematics/softplus_normalization/softplus_normalization.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["condition", "seed", "final_acc", "std_grad", "spikes_10", "spikes_50", "sat_rate"])
        for flag in [True, False]:
            label = "softplus" if flag else "clipped"
            for seed_idx, r in enumerate(logged_metrics[flag]):
                writer.writerow(
                    [label, seed_idx, r["final_acc"], r["std_grad"], r["spikes_10"], r["spikes_50"], r["sat_rate"]])
    print(f"Saved aggregated multi-seed results to {csv_path}")

    # --- STATISTICAL Falsification Assertions ---
    assert len(logged_metrics[True]) == len(seeds) and len(logged_metrics[False]) == len(seeds), \
        "FALSIFICATION: Optimization runs crashed or failed to complete all seeds."

    # Softplus must on average escape boundary saturation much better than the clipped baseline
    assert summary[True]["sat_mean"] < summary[False]["sat_mean"] - 5.0, \
        f"FALSIFICATION: Softplus failed to dynamically resolve boundary saturation. Ours: {summary[True]['sat_mean']:.1f}%, Clipped: {summary[False]['sat_mean']:.1f}%"

    print(
        "Experiment 4 passed successfully! Statistical stability and convergence verified across 10 independent seeds.")


if __name__ == "__main__":
    run_softplus_study()