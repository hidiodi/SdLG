# scripts/diagnostics/exact_sdlg/ablation_minesweeper_sdlg.py
import sys
import os
import time
import csv
import math
import jax
import jax.numpy as jnp
import numpy as np
import distrax
import gc
import wandb
from typing import Any, Tuple
from collections import namedtuple

current_file_path = os.path.abspath(__file__)
project_root = current_file_path
while not os.path.exists(os.path.join(project_root, "src")):
    project_root = os.path.dirname(project_root)
    if project_root == "/":
        break
if project_root not in sys.path:
    sys.path.append(project_root)

from src.envs.popgym_wrapper import make_popgym_env, POPGymEnv
from gymnax.wrappers.purerl import LogWrapper
import flax.linen as nn
from flax.linen.initializers import orthogonal, constant
from src.models.purejaxrl.wrappers import RepresentationNetwork
from src.models.common.sdlg_hybrid import (
    GatedAttentionGLALayer,
    GatedSlidingWindowAttention,
    GTrXLGRUGate
)
import src.trainers.purejax_trainer as trainer_module
from src.trainers.purejax_trainer import make_update

SdLGCarry = namedtuple("SdLGCarry", ["tilde_k", "tilde_v", "theta", "step_count", "swa_k", "swa_v"])

# =====================================================================
# --- ABLATION ARCHITECTURE DEFINITIONS ---
# =====================================================================

class SingleSeqSdLGHybrid(nn.Module):
    """
    Unbatched SdLG sequence model designed to be vmapped over the batch dimension (dim 0).
    inputs: (T, d_model)
    terminations: (T,)
    carry: tuple of SdLGCarry
    """
    layer_types: tuple
    d_model: int = 128
    d_k: int = 32
    d_v: int = 32
    d_ffc: int = 512
    r: int = 4
    window_size: int = 64
    alpha: float = 0.1
    eta: float = 0.05
    gru_bias: float = 2.0

    @nn.compact
    def __call__(self, inputs, terminations, carry, deterministic=True):
        if inputs.shape[0] == 1:
            reset_mask = 1.0 - terminations[0]
            carry = jax.tree_util.tree_map(lambda x: x * reset_mask, carry)

        if inputs.shape[-1] != self.d_model:
            x = nn.Dense(self.d_model, name='emb_layer', kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(inputs)
            x = nn.relu(x)
        else:
            x = inputs

        new_carry = []
        for layer_idx, ltype in enumerate(self.layer_types):
            layer_carry = carry[layer_idx]
            norm_x = nn.LayerNorm(name=f'ln1_layer_{layer_idx}')(x)

            if ltype == 'swa':
                attn_out, new_swa_k, new_swa_v = GatedSlidingWindowAttention(
                    d_model=self.d_model,
                    d_k=self.d_k,
                    d_v=self.d_v,
                    window_size=self.window_size,
                    name=f'swa_{layer_idx}'
                )(norm_x, terminations, layer_carry.swa_k, layer_carry.swa_v, layer_carry.step_count, deterministic=True)

                if inputs.shape[0] > 1:
                    def compute_sc(c_sc, term):
                        next_step = c_sc * (1.0 - term) + 1.0
                        return next_step, next_step
                    _, step_counts = jax.lax.scan(compute_sc, layer_carry.step_count, terminations)
                    new_sc = step_counts[-1]
                else:
                    new_sc = layer_carry.step_count * (1.0 - terminations[0]) + 1.0

                updated_carry = layer_carry._replace(swa_k=new_swa_k, swa_v=new_swa_v, step_count=new_sc)

            elif ltype == 'gla':
                attn_out, _, (new_k, new_v, new_th), new_sc = GatedAttentionGLALayer(
                    d_model=self.d_model,
                    d_k=self.d_k,
                    d_v=self.d_v,
                    r=self.r,
                    eta=self.eta,
                    name=f'gla_{layer_idx}'
                )(norm_x, terminations, (layer_carry.tilde_k, layer_carry.tilde_v, layer_carry.theta), layer_carry.step_count)

                updated_carry = layer_carry._replace(tilde_k=new_k, tilde_v=new_v, theta=new_th, step_count=new_sc)

            gating_out = GTrXLGRUGate(self.d_model, self.gru_bias, name=f'gru1_layer_{layer_idx}')(x, attn_out)
            norm_gated = nn.LayerNorm(name=f'ln2_layer_{layer_idx}')(gating_out)

            ffn_out = nn.Sequential([
                nn.Dense(self.d_ffc, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0), name=f'dense1_{layer_idx}'),
                nn.gelu,
                nn.Dense(self.d_model, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0), name=f'dense2_{layer_idx}')
            ], name=f'ffn_{layer_idx}')(norm_gated)

            x = GTrXLGRUGate(self.d_model, self.gru_bias, name=f'gru2_layer_{layer_idx}')(gating_out, ffn_out)
            new_carry.append(updated_carry)

        X_shared = x
        X_critic = X_shared
        X_actor = jax.lax.stop_gradient(X_shared) * (1.0 - self.alpha) + X_shared * self.alpha
        return ((X_actor, X_critic), jnp.zeros_like(x[..., 0])), tuple(new_carry)


class ActorCriticAblation(nn.Module):
    rnn_module: Any = None
    action_dim: int = 2
    config: dict = None

    @nn.compact
    def __call__(self, carry, x, deterministic=True):
        obs, dones = x
        d_model = self.config["RNN"]["D_MODEL"]
        layer_types = self.config["RNN"]["LAYER_TYPES"]
        r_dim = self.config["RNN"]["R"]

        emb = RepresentationNetwork(d_model=d_model, name="repr_network")(obs)
        emb_trans = jnp.transpose(emb, (1, 0, 2))
        dones_trans = jnp.transpose(dones, (1, 0))

        VmappedModel = nn.vmap(
            SingleSeqSdLGHybrid,
            in_axes=(0, 0, 0, None),
            out_axes=0,
            variable_axes={'params': None},
            split_rngs={'params': False}
        )

        ((X_actor_trans, X_critic_trans), _), new_carry = VmappedModel(
            layer_types=layer_types,
            d_model=d_model,
            d_k=self.config["RNN"]["D_K"],
            d_v=self.config["RNN"]["D_V"],
            r=r_dim,
            name="sdlg_ablation"
        )(emb_trans, dones_trans, carry, deterministic)

        X_actor = jnp.transpose(X_actor_trans, (1, 0, 2))
        X_critic = jnp.transpose(X_critic_trans, (1, 0, 2))

        raw_logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), name="actor_head")(X_actor)
        action_logits = 5.0 * jnp.tanh(raw_logits / 5.0)
        action_logits = jnp.nan_to_num(action_logits, nan=0.0, posinf=5.0, neginf=-5.0)

        critic_values = nn.Dense(1, kernel_init=orthogonal(1.0), name="critic_head")(X_critic)
        critic_values = jnp.squeeze(critic_values, axis=-1)
        critic_values = jnp.nan_to_num(critic_values, nan=0.0, posinf=10.0, neginf=-10.0)

        return action_logits, critic_values, jnp.zeros_like(critic_values), new_carry


# =====================================================================
# --- ABLATION CONFIGURATIONS ---
# =====================================================================

ABLATION_VARIANTS = {
    "SdLG-1GLA-r16 (Rank Scale)": {"layer_types": ("swa", "gla", "swa"), "r": 16},
    "SdLG-2GLA-r4 (Recurrent Depth)": {"layer_types": ("swa", "gla", "gla"), "r": 4},
    "SdLG-3GLA-r4 (Homogeneous GLA)": {"layer_types": ("gla", "gla", "gla"), "r": 4},
}

TRAINING_SEEDS = [42, 54, 99]
TOTAL_STEPS = 15_000_000
NUM_ENVS = 256
NUM_STEPS = 64


def initialize_ablation_carry(batch_size, layer_types, d_k=32, d_v=32, r=4, window_size=64):
    carry_list = []
    for _ in layer_types:
        tilde_k = jnp.zeros((r, d_k))
        tilde_v = jnp.zeros((r, d_v))
        theta = jnp.zeros((r,))
        step_count = jnp.zeros(())
        swa_k = jnp.zeros((window_size, d_k))
        swa_v = jnp.zeros((window_size, d_v))
        carry_list.append(SdLGCarry(tilde_k=tilde_k, tilde_v=tilde_v, theta=theta, step_count=step_count, swa_k=swa_k, swa_v=swa_v))
    unbatched = tuple(carry_list)
    return jax.tree_util.tree_map(lambda x: jnp.repeat(jnp.expand_dims(x, 0), batch_size, axis=0), unbatched)


def get_normalized_score(env_name: str, return_val: float, H: int) -> float:
    max_ret = 1.0
    min_ret = 0.0
    score = (return_val - min_ret) / max(1e-5, max_ret - min_ret)
    return float(np.clip(score, 0.0, 1.0))


def make_eval_env(env_name, H, num_eval_envs=32):
    from src.envs.popgym_wrapper import POPGymEnv
    from src.envs.wrappers import BatchEnvWrapper
    raw_env = POPGymEnv(env_name, max_envs=num_eval_envs * 4)
    eval_len = min(H, raw_env.max_episode_length)
    raw_env.max_episode_length = eval_len
    eval_env = BatchEnvWrapper(LogWrapper(raw_env), num_envs=num_eval_envs)
    return eval_env, eval_len


def run_popgym_eval(rng, params, network, rnn_carry_init_fn, eval_env, eval_len, num_eval_envs=32):
    rng, _rng = jax.random.split(rng)
    obs, env_state = eval_env.reset(_rng, None)
    carry = rnn_carry_init_fn(num_eval_envs)
    dones = jnp.zeros(num_eval_envs, dtype=jnp.bool_)

    def eval_step_fn(runner_state, _):
        c, o, d, s, r_key, episode_ret, active = runner_state
        ac_in = (jnp.expand_dims(o, axis=0), jnp.expand_dims(d, axis=0))
        out = network.apply(params, c, ac_in)
        action_logits, _, _, next_c = out
        pi = distrax.Categorical(logits=action_logits)
        act = pi.mode().squeeze(0)

        r_key, step_key = jax.random.split(r_key)
        next_o, next_s, reward, next_d, _ = eval_env.step(step_key, s, act, None)
        next_episode_ret = episode_ret + reward * active
        next_active = active * (1.0 - next_d.astype(jnp.float32))
        return (next_c, next_o, next_d, next_s, r_key, next_episode_ret, next_active), None

    init_runner_state = (carry, obs, dones, env_state, _rng, jnp.zeros(num_eval_envs), jnp.ones(num_eval_envs, dtype=jnp.float32))
    (final_c, final_o, final_d, final_s, final_key, final_ret, final_active), _ = jax.lax.scan(
        eval_step_fn, init_runner_state, None, length=eval_len
    )

    raw_mean_ret = float(jnp.mean(final_ret))
    norm_score = get_normalized_score("MineSweeperEasy", raw_mean_ret, eval_len)
    return raw_mean_ret, norm_score


def main():
    print("====================================================")
    print("   MINESWEEPER ARCHITECTURAL ABLATION STUDY")
    print("====================================================")

    results_path = os.path.join(project_root, "results", "minesweeper_ablation_results.csv")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    if not os.path.exists(results_path):
        with open(results_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Variant", "Seed", "Step", "Train_Return", "Eval_NormScore_H100", "Eval_NormScore_H200", "SPS"])

    env, env_params = make_popgym_env("MineSweeperEasy", num_envs=NUM_ENVS)

    for variant_name, variant_cfg in ABLATION_VARIANTS.items():
        for SEED in TRAINING_SEEDS:
            print(f"\n🚀 Running Variant: {variant_name} | Seed: {SEED}...")
            jax.clear_caches()
            gc.collect()
    
            config = {
                "NUM_ENVS": NUM_ENVS,
                "NUM_STEPS": NUM_STEPS,
                "TOTAL_TIMESTEPS": TOTAL_STEPS,
                "UPDATE_EPOCHS": 4,
                "NUM_MINIBATCHES": 8,
                "GAMMA": 0.99,
                "GAE_LAMBDA": 0.95,
                "CLIP_EPS": 0.2,
                "ENT_COEF": 0.0001,
                "VF_COEF": 0.5,
                "MAX_GRAD_NORM": 0.5,
                "LR": 0.0003,
                "ANNEAL_LR": False,
                "OPTIMISTIC_RESETS": False,
                "LOG_INTERVAL": 10,
                "RNN": {
                    "RNN_TYPE": "ablation",
                    "D_MODEL": 128,
                    "N_HEADS": 4,
                    "N_LAYERS": len(variant_cfg["layer_types"]),
                    "D_FFC": 512,
                    "D_K": 32,
                    "D_V": 32,
                    "R": variant_cfg["r"],
                    "WINDOW_SIZE": 64,
                    "LAYER_TYPES": variant_cfg["layer_types"],
                },
                "SEED": SEED,
            }
    
            wandb_project = os.environ.get("WANDB_PROJECT", "PopGym-Benchmark")
            wandb.init(
                project=wandb_project,
                group="minesweeper_ablation",
                name=f"{variant_name}-seed{SEED}",
                config=config,
                reinit=True,
            )
    
            trainer_module.ActorCriticRNN = ActorCriticAblation
            rnn_carry_init_fn = lambda b: initialize_ablation_carry(
                b, layer_types=variant_cfg["layer_types"], d_k=32, d_v=32, r=variant_cfg["r"], window_size=64
            )
    
            rng = jax.random.PRNGKey(SEED)
            raw_update_fn, runner_state = make_update(
                rng, config, env, env_params, rnn_module=None, rnn_carry_init=rnn_carry_init_fn
            )
    
            jit_update_fn = jax.jit(raw_update_fn)
    
            steps_per_update = NUM_ENVS * NUM_STEPS
            log_interval = 2
            total_jit_calls = max(1, (TOTAL_STEPS // steps_per_update) // log_interval)
    
            eval_env_100, eval_len_100 = make_eval_env("MineSweeperEasy", H=100)
            eval_env_200, eval_len_200 = make_eval_env("MineSweeperEasy", H=200)
            network = ActorCriticAblation(rnn_module=None, action_dim=env.action_space(env_params).n, config=config)
    
            t0 = time.time()
            for i in range(1, total_jit_calls + 1):
                update_res = jit_update_fn(runner_state)
                runner_state = update_res["runner_state"]
                metrics = update_res["metric"]
    
                current_step = i * steps_per_update * log_interval
                elapsed = time.time() - t0
                sps = current_step / max(1.0, elapsed)
                train_ret = float(np.mean(metrics["returned_episode_returns"]))
    
                rng, eval_rng = jax.random.split(rng)
                current_params = runner_state[0].params
    
                _, score100 = run_popgym_eval(eval_rng, current_params, network, rnn_carry_init_fn, eval_env_100, eval_len_100)
                _, score200 = run_popgym_eval(eval_rng, current_params, network, rnn_carry_init_fn, eval_env_200, eval_len_200)
    
                print(f"[{variant_name}-S{SEED}] Step {current_step}/{TOTAL_STEPS} | Train Ret: {train_ret:.3f} | Eval H100: {score100:.3f} | H200: {score200:.3f} | SPS: {sps:.0f}", flush=True)
    
                with open(results_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([variant_name, SEED, current_step, train_ret, score100, score200, round(sps, 1)])
    
                wandb.log({
                    "global_step": current_step,
                    "train_return": train_ret,
                    "eval_norm_score_h100": score100,
                    "eval_norm_score_h200": score200,
                    "sps": sps
                })
                sys.stdout.flush()
    
            wandb.finish()

    print("\n✅ Minesweeper Ablation Study Complete!")


if __name__ == "__main__":
    main()
