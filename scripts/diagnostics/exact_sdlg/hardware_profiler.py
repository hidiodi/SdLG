# hardware_profiler.py
import jax
import jax.numpy as jnp
import flax.linen as nn
import time
import os
import matplotlib.pyplot as plt
import numpy as np
import gc

# MUST be disabled to get accurate memory profiling in JAX
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["HSA_ENABLE_SDMA"] = "0"

from src.models.common.gtrxl import BatchedGTrXL
from src.models.common.exact_sdlg import UnbatchedExactSdLG
from src.models.common.sdlg_hybrid import BatchedSdLGHybrid


def run_profiler():
    print("=====================================================")
    print("=== SYNTHETIC HARDWARE PROFILER (OOM STRESS TEST) ===")
    print("=====================================================")

    BATCH_SIZE = 16
    D_MODEL = 256
    N_HEADS = 8
    SEQ_LENGTHS = [128, 256, 512, 1024, 2048, 4096]

    results = {
        "Exact-SdLG": {"vram": [], "time": []},
        "Approx-SdLG": {"vram": [], "time": []},
        "GTrXL": {"vram": [], "time": []}
    }

    for T in SEQ_LENGTHS:
        print(f"\n📏 Testing Sequence Length T={T}...")

        # SdLG variants expect (Batch, Time, Dim)
        dummy_x_sdlg = jnp.ones((BATCH_SIZE, T, D_MODEL))
        dummy_dones_sdlg = jnp.zeros((BATCH_SIZE, T))

        # GTrXL expects (Time, Batch, Dim)
        dummy_x_gtrxl = jnp.ones((T, BATCH_SIZE, D_MODEL))
        dummy_dones_gtrxl = jnp.zeros((T, BATCH_SIZE))

        # --- 1. Test Exact-SdLG ---
        print("  ▶ Profiling Exact-SdLG...")
        jax.clear_caches()
        gc.collect()
        try:
            sdlg_carry = jax.tree_util.tree_map(
                lambda x: jnp.repeat(jnp.expand_dims(x, 0), BATCH_SIZE, axis=0),
                UnbatchedExactSdLG.initialize_carry(n_layers=2, n_heads=N_HEADS, d_k=32, d_v=32, window_size=0)
            )
            Model = nn.vmap(UnbatchedExactSdLG, in_axes=(0, 0, 0, None), out_axes=(0, 0, 0),
                            variable_axes={'params': None}, split_rngs={'params': False})
            sdlg = Model(n_layers=2, d_model=D_MODEL, n_heads=N_HEADS, d_k=32, d_v=32, d_ffc=512, window_size=0)
            params_sdlg = sdlg.init(jax.random.PRNGKey(0), dummy_x_sdlg, dummy_dones_sdlg, sdlg_carry, True)

            @jax.jit
            def sdlg_step(p, x, d, c):
                out, _, _ = sdlg.apply(p, x, d, c, True)
                return jnp.mean(out)

            _ = sdlg_step(params_sdlg, dummy_x_sdlg, dummy_dones_sdlg, sdlg_carry).block_until_ready()

            start_mem = jax.devices()[0].memory_stats().get('bytes_in_use', 0)
            start_time = time.time()
            for _ in range(5): _ = sdlg_step(params_sdlg, dummy_x_sdlg, dummy_dones_sdlg,
                                             sdlg_carry).block_until_ready()
            end_time = time.time()
            peak_mem = jax.devices()[0].memory_stats().get('peak_bytes_in_use', 0)

            mem_gb = max((peak_mem - start_mem) / (1024 ** 3), 0.01)
            step_ms = ((end_time - start_time) / 5) * 1000
            results["Exact-SdLG"]["vram"].append(mem_gb)
            results["Exact-SdLG"]["time"].append(step_ms)
            print(f"    ✅ Success | VRAM: {mem_gb:.3f} GB | Latency: {step_ms:.1f} ms")
        except Exception as e:
            print(f"    ❌ FAILED: {str(e).splitlines()[0]}")
            results["Exact-SdLG"]["vram"].append(np.nan)
            results["Exact-SdLG"]["time"].append(np.nan)

        # --- 2. Test Approx-SdLG ---
        print("  ▶ Profiling Approx-SdLG...")
        jax.clear_caches()
        gc.collect()
        try:
            ModelApprox = nn.vmap(BatchedSdLGHybrid, in_axes=(0, 0, 0, None), out_axes=0,
                                  variable_axes={'params': None}, split_rngs={'params': False})
            approx = ModelApprox(n_layers=2, d_model=D_MODEL, d_k=32, d_v=32, d_ffc=512, r=4, window_size=0, alpha=0.1,
                                 return_split=False)

            c = BatchedSdLGHybrid.initialize_carry(BATCH_SIZE, 2, 32, 32, 4, 0)
            approx_carry = jax.tree_util.tree_map(lambda x: jnp.repeat(jnp.expand_dims(x, 0), BATCH_SIZE, axis=0), c)

            params_approx = approx.init(jax.random.PRNGKey(0), dummy_x_sdlg, dummy_dones_sdlg, approx_carry, True)

            @jax.jit
            def approx_step(p, x, d, c):
                (out, _), _ = approx.apply(p, x, d, c, True)
                return jnp.mean(out)

            _ = approx_step(params_approx, dummy_x_sdlg, dummy_dones_sdlg, approx_carry).block_until_ready()

            start_mem = jax.devices()[0].memory_stats().get('bytes_in_use', 0)
            start_time = time.time()
            for _ in range(5): _ = approx_step(params_approx, dummy_x_sdlg, dummy_dones_sdlg,
                                               approx_carry).block_until_ready()
            end_time = time.time()
            peak_mem = jax.devices()[0].memory_stats().get('peak_bytes_in_use', 0)

            mem_gb = max((peak_mem - start_mem) / (1024 ** 3), 0.01)
            step_ms = ((end_time - start_time) / 5) * 1000
            results["Approx-SdLG"]["vram"].append(mem_gb)
            results["Approx-SdLG"]["time"].append(step_ms)
            print(f"    ✅ Success | VRAM: {mem_gb:.3f} GB | Latency: {step_ms:.1f} ms")
        except Exception as e:
            print(f"    ❌ FAILED: {str(e).splitlines()[0]}")
            results["Approx-SdLG"]["vram"].append(np.nan)
            results["Approx-SdLG"]["time"].append(np.nan)

        # --- 3. Test GTrXL ---
        print("  ▶ Profiling GTrXL...")
        jax.clear_caches()
        gc.collect()
        try:
            gtrxl_carry = BatchedGTrXL.initialize_carry(BATCH_SIZE, memory_len=128, embedding_dim=D_MODEL, layer_num=2)
            gtrxl = BatchedGTrXL(head_dim=32, embedding_dim=D_MODEL, head_num=N_HEADS, layer_num=2, memory_len=128,
                                 use_embedding_layer=False)
            params_gtrxl = gtrxl.init(jax.random.PRNGKey(0), gtrxl_carry, (dummy_x_gtrxl, dummy_dones_gtrxl))

            @jax.jit
            def gtrxl_step(p, x, d, c):
                _, out = gtrxl.apply(p, c, (x, d))
                return jnp.mean(out)

            _ = gtrxl_step(params_gtrxl, dummy_x_gtrxl, dummy_dones_gtrxl, gtrxl_carry).block_until_ready()

            start_mem = jax.devices()[0].memory_stats().get('bytes_in_use', 0)
            start_time = time.time()
            for _ in range(5): _ = gtrxl_step(params_gtrxl, dummy_x_gtrxl, dummy_dones_gtrxl,
                                              gtrxl_carry).block_until_ready()
            end_time = time.time()
            peak_mem = jax.devices()[0].memory_stats().get('peak_bytes_in_use', 0)

            mem_gb = max((peak_mem - start_mem) / (1024 ** 3), 0.01)
            step_ms = ((end_time - start_time) / 5) * 1000
            results["GTrXL"]["vram"].append(mem_gb)
            results["GTrXL"]["time"].append(step_ms)
            print(f"    ✅ Success | VRAM: {mem_gb:.3f} GB | Latency: {step_ms:.1f} ms")
        except Exception as e:
            print(f"    ❌ FAILED: {str(e).splitlines()[0]}")
            results["GTrXL"]["vram"].append(np.nan)
            results["GTrXL"]["time"].append(np.nan)

    # --- Plotting ---
    print("\nGenerating Figure 5.1...")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    t_vals = np.array(SEQ_LENGTHS)

    # Plot VRAM
    axes[0].plot(t_vals, results["Exact-SdLG"]["vram"], label="Exact-SdLG", color="#2c7bb6", linewidth=3,
                 marker="o")
    axes[0].plot(t_vals, results["Approx-SdLG"]["vram"], label="Approx-SdLG", color="#d7191c", linewidth=3,
                 marker="o", linestyle="-")
    axes[0].plot(t_vals, results["GTrXL"]["vram"], label="GTrXL", color="#fdae61", linewidth=3, marker="o",
                 linestyle="--")
    axes[0].set_title("VRAM Allocation vs Sequence Horizon (B=16)", fontweight="bold")
    axes[0].set_xlabel("Sequence Length (T)")
    axes[0].set_ylabel("Peak VRAM Usage (GB)")
    axes[0].set_xticks(t_vals)
    axes[0].legend(frameon=True)

    # Plot Latency
    axes[1].plot(t_vals, results["Exact-SdLG"]["time"], label="Exact-SdLG", color="#2c7bb6", linewidth=3,
                 marker="o")
    axes[1].plot(t_vals, results["Approx-SdLG"]["time"], label="Approx-SdLG", color="#d7191c", linewidth=3,
                 marker="o", linestyle="-")
    axes[1].plot(t_vals, results["GTrXL"]["time"], label="GTrXL", color="#fdae61", linewidth=3, marker="o",
                 linestyle="--")
    axes[1].set_title("Forward+Backward Step Latency", fontweight="bold")
    axes[1].set_xlabel("Sequence Length (T)")
    axes[1].set_ylabel("Execution Time (ms)")
    axes[1].set_xticks(t_vals)
    axes[1].legend(frameon=True)

    os.makedirs("plots", exist_ok=True)
    plt.tight_layout()
    plt.savefig("plots/profile_hardware_scaling.png", dpi=300, bbox_inches='tight')
    print("✅ Saved to plots/profile_hardware_scaling.png")


if __name__ == "__main__":
    run_profiler()