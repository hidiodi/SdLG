# scripts/diagnostics/exact_sdlg/exact_sdlg_craftax_ablations.py

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
import gc
import orbax.checkpoint as ocp
from pathlib import Path
from flax import struct
from typing import Any, Tuple, Callable
from flax.training.train_state import TrainState
from flax.linen.initializers import orthogonal, constant, zeros

current_file_path = os.path.abspath(__file__)
project_root = current_file_path
while not os.path.exists(os.path.join(project_root, "src")):
    project_root = os.path.dirname(project_root)
    if project_root == "/": break
if project_root not in sys.path: sys.path.append(project_root)

from src.trainers.purejax_trainer import make_update
from src.envs.builder import build_purejaxrl_env
from src.utils.config_schema import StaticEnvConfig, DynamicTrainingConfig
from craftax.craftax.constants import Achievement

# =====================================================================
# --- ABLATION CONFIGURATIONS ---
# =====================================================================
ABLATIONS = {
    "Baseline_ApproxDense": {"USE_EXACT": False, "USE_MOE": False, "GAMMA_BIAS": 0.0},
    "Ablation_A_ExactDense": {"USE_EXACT": True, "USE_MOE": False, "GAMMA_BIAS": "log"},
    "Ablation_B_ApproxMoE": {"USE_EXACT": False, "USE_MOE": True, "GAMMA_BIAS": 0.0},
    "Ablation_C_ExactMoE_LogBias": {"USE_EXACT": True, "USE_MOE": True, "GAMMA_BIAS": "log"},
    "Ablation_D_ExactMoE_ZeroBias": {"USE_EXACT": True, "USE_MOE": True, "GAMMA_BIAS": 0.0},
}

MODELS_TO_RUN = list(ABLATIONS.keys())
CHECKPOINT_BASE_DIR = os.path.abspath("checkpoints/exp18_craftax")
COMPLETED_RUNS_PATH = os.path.join(CHECKPOINT_BASE_DIR, "completed_craftax_ablations.json")


def load_completed_runs():
    if os.path.exists(COMPLETED_RUNS_PATH):
        try:
            with open(COMPLETED_RUNS_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_completed_run(model_name):
    completed = load_completed_runs()
    if model_name not in completed: completed.append(model_name)
    with open(COMPLETED_RUNS_PATH, "w") as f: json.dump(completed, f)


# =====================================================================
# --- INLINED FAST EXACT GLA & ABLATION WRAPPER ---
# =====================================================================
class FastExactMultiScaleGLA(nn.Module):
    d_model: int
    n_heads: int
    d_k: int
    d_v: int
    gamma_bias_type: str

    def setup(self):
        if self.gamma_bias_type == "log":
            base_timescales = jnp.array([16.0, 64.0, 256.0, 1024.0, 4096.0, 16384.0])
            if self.n_heads > len(base_timescales):
                pad_size = self.n_heads - len(base_timescales)
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

        phi_q = jax.nn.elu(q) + 1.0
        phi_k = jax.nn.elu(k) + 1.0

        gamma = jnp.exp(-jax.nn.softplus(self.delta_lambda_proj(inputs) + self.gamma_bias))
        beta = jax.nn.sigmoid(self.beta_proj(inputs))

        write_M = beta[..., None, None] * (phi_k[..., None] * v[..., None, :])
        write_z = beta[..., None] * phi_k

        discount_gamma = gamma * (1.0 - terminations[:, None])
        discount_M = discount_gamma[..., None, None]
        discount_z = discount_gamma[..., None]

        def combine(a, b):
            w_a, d_a = a
            w_b, d_b = b
            return w_b + d_b * w_a, d_a * d_b

        write_M = write_M.at[0].set(write_M[0] + discount_M[0] * M_prev)
        write_z = write_z.at[0].set(write_z[0] + discount_z[0] * z_prev)

        all_M, _ = jax.lax.associative_scan(combine, (write_M, discount_M), axis=0)
        all_z, _ = jax.lax.associative_scan(combine, (write_z, discount_z), axis=0)

        num = jnp.matmul(jnp.expand_dims(phi_q, -2), all_M).squeeze(-2)
        den = jnp.sum(phi_q * all_z, axis=-1, keepdims=True) + 1e-6
        out = self.out_proj((num / den).reshape((T, self.n_heads * self.d_v)))

        telemetry = {"beta_mean": jnp.mean(beta), "gamma_h0": jnp.mean(gamma[:, 0]), "gamma_hN": jnp.mean(gamma[:, -1])}
        return out, (all_M[-1], all_z[-1]), telemetry


class SparseMoESwiGLU(nn.Module):
    d_model: int
    d_ffc: int
    num_experts: int = 4
    top_k: int = 2

    @nn.compact
    def __call__(self, x):
        prefix_shape = x.shape[:-1]
        gate_logits = nn.Dense(self.num_experts, use_bias=False)(x)
        routing_probs = jax.nn.softmax(gate_logits, axis=-1)
        
        # Rule 2: Top-K Routing fix (use jax.lax.top_k instead of approx_max_k)
        topk_probs, topk_indices = jax.lax.top_k(routing_probs, k=self.top_k)
        expert_weights = topk_probs / (jnp.sum(topk_probs, axis=-1, keepdims=True) + 1e-9)

        # Rule 2: MoE Sequential Loop Fix (build list and stack)
        expert_outs = []
        for e in range(self.num_experts):
            gate = nn.Dense(self.d_ffc, use_bias=False)(x)
            up = nn.Dense(self.d_ffc, use_bias=False)(x)
            down = nn.Dense(self.d_model, use_bias=False)(jax.nn.silu(gate) * up)
            expert_outs.append(down)
        expert_outputs = jnp.stack(expert_outs, axis=-2)

        combined = jnp.zeros_like(x)
        for k in range(self.top_k):
            indices = jnp.broadcast_to(topk_indices[..., k, None, None], (*prefix_shape, 1, self.d_model))
            selected = jnp.take_along_axis(expert_outputs, indices, axis=-2).squeeze(-2)
            combined += expert_weights[..., k, None] * selected
        return combined


class ActorCriticAblation(nn.Module):
    action_dim: int
    config: dict

    @nn.compact
    def __call__(self, carry, x):
        obs, dones = x

        from src.models.purejaxrl.wrappers import RepresentationNetwork
        emb = RepresentationNetwork(d_model=self.config["RNN"]["D_MODEL"])(obs)

        emb_trans = jnp.transpose(emb, (1, 0, 2))
        dones_trans = jnp.transpose(dones, (1, 0))

        ablation_cfg = self.config["ABLATION"]

        if ablation_cfg["USE_EXACT"]:
            def scan_exact(carry_b, xs):
                emb_b, dones_b = xs
                out, new_c, tel = FastExactMultiScaleGLA(
                    d_model=self.config["RNN"]["D_MODEL"], n_heads=self.config["RNN"]["N_HEADS"],
                    d_k=32, d_v=32, gamma_bias_type=ablation_cfg["GAMMA_BIAS"]
                )(emb_b, dones_b, carry_b)
                return new_c, (out, tel)

            new_carry, (rnn_out_trans, telemetry) = jax.lax.scan(scan_exact, carry, (emb_trans, dones_trans))
        else:
            from src.models.common.sdlg_hybrid import BatchedSdLGHybrid
            VmappedSdLG = nn.vmap(
                BatchedSdLGHybrid,
                in_axes=(0, 0, 0, None),
                out_axes=0,
                variable_axes={'params': None},
                split_rngs={'params': False}
            )
            ((X_actor_trans, X_critic_trans), _), new_carry = VmappedSdLG(
                n_layers=1, d_model=self.config["RNN"]["D_MODEL"], d_k=32, d_v=32, d_ffc=1024, r=4, window_size=16,
                return_split=True
            )(emb_trans, dones_trans, carry, True)
            rnn_out_trans = X_actor_trans
            telemetry = {"beta_mean": 0.5, "gamma_h0": 0.95, "gamma_hN": 0.99}

        rnn_out = jnp.transpose(rnn_out_trans, (1, 0, 2))
        alpha = 0.1
        x_critic = rnn_out
        x_actor = jax.lax.stop_gradient(rnn_out) * (1.0 - alpha) + rnn_out * alpha

        if ablation_cfg["USE_MOE"]:
            actor_rep = SparseMoESwiGLU(self.config["RNN"]["D_MODEL"], 1024)(x_actor)
            critic_rep = SparseMoESwiGLU(self.config["RNN"]["D_MODEL"], 1024)(x_critic)
        else:
            actor_rep = x_actor
            critic_rep = x_critic

        action_logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(actor_rep)
        # Rule 2: NaN-Shielding for Actor logits
        action_logits = 5.0 * jnp.tanh(jnp.nan_to_num(action_logits, nan=0.0, posinf=5.0, neginf=-5.0))
        
        # Rule 2: NaN-Shielding for Critic values
        critic_raw = jnp.squeeze(nn.Dense(1, kernel_init=orthogonal(1.0))(critic_rep), axis=-1)
        critic_values = jnp.nan_to_num(critic_raw, nan=0.0, posinf=10.0, neginf=-10.0)

        return action_logits, critic_values, jnp.zeros_like(critic_values), new_carry


# =====================================================================
# --- BENCHMARK RUNNER ---
# =====================================================================
# Rule 1: Multi-Seed statistical rigor (N=5 seeds)
TRAINING_SEEDS = [42, 54, 99, 123, 456]

def run_craftax_ablations():
    print("====================================================================")
    print("=== EXP 18: CRAFTAX 300M ABLATION SUITE (WITH RAM FLUSHES) ===")
    print("====================================================================")

    os.makedirs(CHECKPOINT_BASE_DIR, exist_ok=True)
    completed = load_completed_runs()

    TOTAL_STEPS = 300_000_000
    NUM_ENVS = 256
    # Rule 3: strictly hardcode NUM_STEPS = 64
    NUM_STEPS = 64
    STEPS_PER_UPDATE = NUM_ENVS * NUM_STEPS
    FLUSH_INTERVAL = 50_000_000
    LOG_INTERVAL_UPDATES = 500_000 // STEPS_PER_UPDATE
    TOTAL_UPDATES = TOTAL_STEPS // STEPS_PER_UPDATE

    WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "SdLGHybrid-Exp8-Craftax")

    # Override trainers network routing natively
    import src.models.purejaxrl.wrappers
    src.models.purejaxrl.wrappers.ActorCriticSdLGHybrid = ActorCriticAblation

    for model_name in MODELS_TO_RUN:
        for seed in TRAINING_SEEDS:
            run_key = f"{model_name}_seed_{seed}"
            if run_key in completed:
                print(f"⏩ Skipping {run_key} (Already completed).")
                continue

            # Rule 1: Clean JAX traces and collect garbage between seeds
            jax.clear_caches()
            gc.collect()
            print(f"\n🚀 LAUNCHING CRAFTAX ABLATION: [{model_name}] | Seed: {seed}")

            ablation_cfg = ABLATIONS[model_name]
            config = {
                "NUM_ENVS": NUM_ENVS, "NUM_STEPS": NUM_STEPS, "TOTAL_TIMESTEPS": TOTAL_STEPS,
                "UPDATE_EPOCHS": 4, "NUM_MINIBATCHES": 8, "GAMMA": 0.99, "GAE_LAMBDA": 0.95,
                "CLIP_EPS": 0.2, "ENT_COEF": 0.01, "VF_COEF": 0.5, "MAX_GRAD_NORM": 0.5, "LR": 0.0002,
                "ANNEAL_LR": True, "OPTIMISTIC_RESETS": False,
                "LOG_INTERVAL": LOG_INTERVAL_UPDATES,
                "RNN": {"RNN_TYPE": "sdlg_hybrid", "D_MODEL": 256, "N_HEADS": 8},
                "ABLATION": ablation_cfg,
                "SEED": seed
            }

            static_env = StaticEnvConfig(env_name="Craftax-Symbolic-v1", num_envs=NUM_ENVS, num_steps=NUM_STEPS,
                                         max_episode_steps=10000)
            dynamic_train = DynamicTrainingConfig(
                total_timesteps=TOTAL_STEPS, learning_rate=0.0002, gamma=0.99, gae_lambda=0.95,
                clip_eps=0.2, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, update_epochs=4,
                minibatch_size=2048, log_interval=LOG_INTERVAL_UPDATES
            )

            # Rule 4: LogWrapper injection checking
            env, env_params = build_purejaxrl_env(static_env, dynamic_train)
            from src.envs.wrappers import LogWrapper
            if not isinstance(env, LogWrapper):
                env = LogWrapper(env)
                
            action_dim = env.action_space(env_params).n
            network = ActorCriticAblation(action_dim, config)

            def rnn_carry_init(batch_size):
                if ablation_cfg["USE_EXACT"]:
                    return (jnp.zeros((batch_size, 8, 32, 32)), jnp.zeros((batch_size, 8, 32)))
                else:
                    from src.models.common.sdlg_hybrid import BatchedSdLGHybrid
                    c = BatchedSdLGHybrid.initialize_carry(batch_size, 1, 32, 32, 4, 16)
                    return jax.tree_util.tree_map(lambda x: jnp.repeat(x[None, ...], batch_size, axis=0), c)

            ckpt_dir = Path(CHECKPOINT_BASE_DIR) / f"{model_name}_seed_{seed}_craftax"
            ckpt_manager = ocp.CheckpointManager(ckpt_dir.resolve(),
                                                 options=ocp.CheckpointManagerOptions(max_to_keep=1, create=True))

            rng = jax.random.PRNGKey(seed)
            raw_update_fn, runner_state = make_update(rng, config, env, env_params, lambda: network, rnn_carry_init)

            start_loop_idx = 0
            latest_step = ckpt_manager.latest_step()
            if latest_step is not None:
                print(f"  [RESTORE] Resuming from Step {latest_step * LOG_INTERVAL_UPDATES * STEPS_PER_UPDATE / 1e6:.1f}M")
                runner_state = \
                ckpt_manager.restore(latest_step, args=ocp.args.StandardRestore({"runner_state": runner_state}))[
                    "runner_state"]
                start_loop_idx = latest_step

            wandb.init(project=WANDB_PROJECT, group="exp18_craftax_ablations", name=f"{model_name}_seed_{seed}", config=config,
                       resume="allow", id=f"craftax_{model_name}_seed_{seed}")

            @jax.jit
            def compiled_update(r_state):
                out = raw_update_fn(r_state)
                m = out["metric"]
                d_mask = m.get("returned_episode", jnp.ones_like(m.get("entropy", jnp.zeros(1))))

                def mean_d(x): return jnp.sum(x * d_mask) / jnp.maximum(1.0, jnp.sum(d_mask))

                return out["runner_state"], {
                    "ret": mean_d(m.get("returned_episode_returns", jnp.zeros_like(d_mask))),
                    "vloss": jnp.mean(m.get("value_loss", jnp.zeros(1))),
                }

            NUM_LOG_LOOPS = TOTAL_UPDATES // LOG_INTERVAL_UPDATES

            for loop_idx in range(start_loop_idx, NUM_LOG_LOOPS):
                runner_state, metrics = compiled_update(runner_state)
                runner_state = jax.block_until_ready(runner_state)
                metrics = jax.device_get(metrics)

                steps = (loop_idx + 1) * LOG_INTERVAL_UPDATES * STEPS_PER_UPDATE
                wandb.log({"Train/Mean_Return": float(metrics["ret"]), "Train/Value_Loss": float(metrics["vloss"])},
                          step=steps)

                print(f"  [TRAIN] Step: {steps / 1e6:>5.1f}M | Ret: {float(metrics['ret']):>5.1f}")

                # INTENTIONAL RAM FLUSH BOUNDARY
                if steps % FLUSH_INTERVAL == 0 and loop_idx > start_loop_idx:
                    print(
                        f"\n🛑 [MEMORY FLUSH] Reached {steps / 1e6:.1f}M steps. Saving checkpoint and exiting for RAM clearance...")
                    ckpt_manager.save(loop_idx + 1, args=ocp.args.StandardSave({"runner_state": runner_state}))
                    ckpt_manager.wait_until_finished()
                    wandb.finish()
                    sys.exit(43)  # Triggers Immortal Watchdog to reboot this script!

            print(f"\n✅ [{model_name}] Seed {seed} 300M Steps Complete!")
            wandb.finish()
            save_completed_run(run_key)


if __name__ == "__main__":
    run_craftax_ablations()