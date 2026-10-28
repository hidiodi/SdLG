# scripts/diagnostics/exp16/exp16_compile_safety.py

import os
import sys
import time
import jax
import jax.numpy as jnp

# Append project root


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
from src.utils.config_schema import StaticEnvConfig, DynamicTrainingConfig
from src.trainers.purejax_trainer import make_update
from src.models.common.gtrxl import BatchedGTrXL
from src.models.common.sdlg_hybrid import BatchedSdLGHybrid


class TortureRewardWrapper:
    """
    Simulates Step 65-Million by artificially exploding the rewards
    and feeding massive scale variance into the Value function.
    """

    def __init__(self, env):
        self._env = env

    @property
    def default_params(self):
        return getattr(self._env, "default_params", None)

    def action_space(self, params):
        return self._env.action_space(params)

    def observation_space(self, params):
        return self._env.observation_space(params)

    def reset(self, key, params):
        return self._env.reset(key, params)

    def step(self, key, state, action, params):
        obs, env_state, reward, done, info = self._env.step(key, state, action, params)
        # 💣 THE TORTURE: Multiply all rewards by 1,000,000
        return obs, env_state, reward * 1000000.0, done, info


def check_for_nans(train_state_params):
    """Recursively scans the PyTree to detect any NaN or Inf weights."""
    all_finite = jax.tree_util.tree_reduce(
        lambda acc, x: acc & jnp.all(jnp.isfinite(x)),
        train_state_params,
        initializer=jnp.array(True)
    )
    return not jax.device_get(all_finite)


def run_torture_test():
    print("====================================================================")
    print("=== EXP 16: NUMERICAL TORTURE TEST (LR=0.5, REWARD=1,000,000x)   ===")
    print("====================================================================")

    # Base configuration tailored for pure, unadulterated mathematical violence
    TORTURE_CONFIG = {
        "NUM_ENVS": 64,
        "NUM_STEPS": 64,
        "TOTAL_TIMESTEPS": 64 * 64 * 50,  # 💣 Increased to 50 Updates
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 4,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,

        # 💣 TORTURE PARAMETERS
        "ENT_COEF": -5.0,  # 💣 NEGATIVE ENTROPY: Forces instant probability collapse
        "LR": 0.5,  # Massive Learning Rate
        "MAX_GRAD_NORM": 1e9,  # No Gradient Shielding
        "VF_COEF": 0.5,
        "HIDDEN": 512,
        "ANNEAL_LR": False,  # Keep it brutal throughout
        "OPTIMISTIC_RESETS": False,
        "LOG_INTERVAL": 10  # Perform 10 steps in one compiled scan
    }

    static_env = StaticEnvConfig(env_name="Craftax-Symbolic-v1", num_envs=64, num_steps=64, max_episode_steps=1000,
                                 optimistic_resets=False)

    dynamic_train = DynamicTrainingConfig(
        total_timesteps=TORTURE_CONFIG["TOTAL_TIMESTEPS"],
        learning_rate=TORTURE_CONFIG["LR"],
        gamma=TORTURE_CONFIG["GAMMA"],
        gae_lambda=TORTURE_CONFIG["GAE_LAMBDA"],
        clip_eps=TORTURE_CONFIG["CLIP_EPS"],
        ent_coef=TORTURE_CONFIG["ENT_COEF"],
        vf_coef=TORTURE_CONFIG["VF_COEF"],
        max_grad_norm=TORTURE_CONFIG["MAX_GRAD_NORM"],
        update_epochs=TORTURE_CONFIG["UPDATE_EPOCHS"],
        minibatch_size=(TORTURE_CONFIG["NUM_ENVS"] * TORTURE_CONFIG["NUM_STEPS"]) // TORTURE_CONFIG["NUM_MINIBATCHES"],
        log_interval=TORTURE_CONFIG["LOG_INTERVAL"]
    )

    base_env, env_params = build_purejaxrl_env(static_env, dynamic_train)

    # Wrap the environment in the Torture Chamber
    env = TortureRewardWrapper(base_env)

    models_to_test = ["GTrXL", "SdLGHybrid"]

    for model_name in models_to_test:
        print(f"\n🧪 Injecting {model_name} into the Torture Chamber...")
        config = TORTURE_CONFIG.copy()

        # 1. Architecture Setup
        if model_name == "GTrXL":
            config["RNN"] = {"RNN_TYPE": "gtrxl"}

            def make_gtrxl():
                return BatchedGTrXL(
                    head_dim=32, embedding_dim=64, head_num=2, mlp_num=2, layer_num=2, memory_len=128
                )

            def init_gtrxl_carry(batch_size):
                return BatchedGTrXL.initialize_carry(batch_size, 128, 64, 2)

            update_fn, runner_state = make_update(
                jax.random.PRNGKey(42), config, env, env_params, make_gtrxl, init_gtrxl_carry
            )
            compiled_update = jax.jit(update_fn)

        elif model_name == "SdLGHybrid":
            config["RNN"] = {
                "RNN_TYPE": "sdlg_hybrid",
                "N_LAYERS": 2, "D_MODEL": 64, "D_HEAD": 16, "N_HEADS": 4,
                "MEMORY_LEN": 128, "D_K": 16, "D_V": 16, "D_FFC": 256,
                "R": 4, "WINDOW_SIZE": 8, "ALPHA": 0.1, "ETA": 0.0,
                "USE_ROPE_PHASE": True, "USE_SOFTPLUS_DENOMINATOR": True, "USE_PER_RANK_NORM": True
            }

            def init_sdlg_carry(batch_size):
                unbatched = BatchedSdLGHybrid.initialize_carry(batch_size, 2, 16, 16, 4, 8)
                return jax.tree_util.tree_map(lambda x: jnp.repeat(jnp.expand_dims(x, 0), batch_size, axis=0),
                                              unbatched)

            update_fn, runner_state = make_update(
                jax.random.PRNGKey(42), config, env, env_params, None, init_sdlg_carry
            )
            compiled_update = jax.jit(update_fn)

        # 2. Execute the Torture Test
        print(f"   > Compiling & Executing 10 Violent Updates...")
        t0 = time.time()

        try:
            # Run the compiled function
            out = compiled_update(runner_state)
            runner_state = out["runner_state"]

            # Block until execution is physically completed on the GPU
            runner_state = jax.block_until_ready(runner_state)
            t1 = time.time()

            # 3. Assess the Damage
            has_nans = check_for_nans(runner_state[0].params)

            if has_nans:
                print(f"   ❌ [FAILED] {model_name} generated NaNs and mathematically collapsed in {t1 - t0:.2f}s!")
                sys.exit(42)  # Return code 42 signals unrecoverable NaN to run_all.py
            else:
                print(
                    f"   ✅ [PASSED] {model_name} successfully absorbed the extreme gradients in {t1 - t0:.2f}s! Parameters remain strictly finite.")

        except Exception as e:
            print(f"   ❌ [CRASH] {model_name} threw an exception during execution!")
            raise e

    print("\n====================================================================")
    print("🏆 ALL MODELS SURVIVED THE TORTURE TEST. NUMERICS ARE BULLETPROOF.")
    print("====================================================================")


if __name__ == "__main__":
    run_torture_test()