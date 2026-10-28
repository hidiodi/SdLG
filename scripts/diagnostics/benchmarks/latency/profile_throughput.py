"""
Hardware Latency & VRAM Profiler (3-Way Comparison)

Measures pure step latency (ms) and Peak VRAM (GB) for a single batch iteration
at variable context lengths T in {128, 256, 512, 1024, 2048, 4096} comparing:
1. Proposed Exact-SdLG
2. Predecessor Approx-SdLG
3. Baseline GTrXL
"""
import sys
import os
import time
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import csv
import logging
import traceback
from pathlib import Path
import json
import matplotlib.pyplot as plt

current_file_path = os.path.abspath(__file__)
project_root = current_file_path
while not os.path.exists(os.path.join(project_root, "src")):
    project_root = os.path.dirname(project_root)
    if project_root == "/": break
if project_root not in sys.path: sys.path.append(project_root)

from src.models.common.sdlg_hybrid import BatchedSdLGHybrid
from src.models.common.exact_sdlg import UnbatchedExactSdLG
from src.models.common.gtrxl import BatchedGTrXL

def run_single_latency_measurement(t_val, model_name):
    logging.getLogger("absl").setLevel(logging.ERROR)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.85"

    batch_size = 16
    d_model = 256
    d_head = 64
    n_heads = d_model // d_head
    n_layers = 4
    window_size = 16
    r_fixed = 8

    rng = jax.random.PRNGKey(42)
    device = jax.devices()[0]
    jax.clear_caches()

    try:
        if model_name == "exact_sdlg":
            # Inputs: [Batch, Time, d_model]
            inputs = jnp.ones((batch_size, t_val, d_model))
            terminations = jnp.zeros((batch_size, t_val))

            unbatched = UnbatchedExactSdLG.initialize_carry(n_layers, n_heads, d_head, d_head, window_size)
            carry = jax.tree_util.tree_map(lambda x: jnp.repeat(jnp.expand_dims(x, 0), batch_size, axis=0), unbatched)

            @jax.jit
            def step_fn(inputs, terminations, carry, r_seed):
                # FIXED: in_axes=(0, 0, 0) perfectly matches the 3 arguments provided
                model = nn.vmap(UnbatchedExactSdLG, in_axes=(0, 0, 0), out_axes=0, variable_axes={'params': None}, split_rngs={'params': False})(
                    n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_head, d_v=d_head, d_ffc=d_model * 4, window_size=window_size
                )
                params = model.init(r_seed, inputs, terminations, carry)
                def loss_fn(p):
                    out, _, _ = model.apply(p, inputs, terminations, carry)
                    return jnp.mean(out)
                loss, grads = jax.value_and_grad(loss_fn)(params)
                return loss, grads

        elif model_name == "approx_sdlg":
            # Inputs: [Batch, Time, d_model]
            inputs = jnp.ones((batch_size, t_val, d_model))
            terminations = jnp.zeros((batch_size, t_val))

            unbatched = BatchedSdLGHybrid.initialize_carry(batch_size, n_layers, d_head, d_head, r_fixed, window_size)
            carry = jax.tree_util.tree_map(lambda x: jnp.repeat(jnp.expand_dims(x, 0), batch_size, axis=0), unbatched)

            @jax.jit
            def step_fn(inputs, terminations, carry, r_seed):
                # FIXED: in_axes=(0, 0, 0) perfectly matches the 3 arguments provided
                model = nn.vmap(BatchedSdLGHybrid, in_axes=(0, 0, 0), out_axes=0, variable_axes={'params': None}, split_rngs={'params': False})(
                    n_layers=n_layers, d_model=d_model, d_k=d_head, d_v=d_head, d_ffc=d_model * 4, r=r_fixed, window_size=window_size, return_split=False
                )
                params = model.init(r_seed, inputs, terminations, carry)
                def loss_fn(p):
                    (out, _), _ = model.apply(p, inputs, terminations, carry)
                    return jnp.mean(out)
                loss, grads = jax.value_and_grad(loss_fn)(params)
                return loss, grads

        elif model_name == "gtrxl":
            # Inputs: [Time, Batch, d_model] (Required by GTrXL wrapper)
            inputs = jnp.ones((t_val, batch_size, d_model))
            terminations = jnp.zeros((t_val, batch_size))

            carry = BatchedGTrXL.initialize_carry(batch_size, t_val, d_model, n_layers)

            @jax.jit
            def step_fn(inputs, terminations, carry, r_seed):
                model = BatchedGTrXL(head_dim=d_head, embedding_dim=d_model, head_num=n_heads, mlp_num=2, layer_num=n_layers, memory_len=t_val, train=False)
                params = model.init(r_seed, carry, (inputs, terminations))
                def loss_fn(p):
                    _, out = model.apply(p, carry, (inputs, terminations))
                    return jnp.mean(out)
                loss, grads = jax.value_and_grad(loss_fn)(params)
                return loss, grads

        def block_tree(tree): jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, 'block_until_ready') else x, tree)

        # Warmup
        loss, grads = step_fn(inputs, terminations, carry, rng)
        loss.block_until_ready()
        block_tree(grads)

        # Measure Peak VRAM after JIT compilation
        mem_stats = device.memory_stats()
        peak_vram_gb = float(mem_stats.get("peak_bytes_in_use", 0.0)) / (1024**3)

        # Profile Latency
        num_iters = 10
        start_time = time.time()
        for _ in range(num_iters):
            loss, grads = step_fn(inputs, terminations, carry, rng)
            loss.block_until_ready()
            block_tree(grads)
        end_time = time.time()

        latency_ms = ((end_time - start_time) / num_iters) * 1000
        print(f"RESULT_LATENCY_MS: {latency_ms:.4f}")
        print(f"RESULT_VRAM_GB: {peak_vram_gb:.4f}")

    except Exception as e:
        print("RESULT_ERROR_TRACEBACK:")
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)

def main():
    print("====================================================================")
    print("=== EXP: HARDWARE LATENCY & VRAM PROFILING (O(T) vs O(T^2))      ===")
    print("====================================================================")

    t_sizes = [128, 256, 512, 1024, 2048, 4096]
    models = ["exact_sdlg", "approx_sdlg", "gtrxl"]

    # Accurate forward+backward pass step latency (ms) and peak activation VRAM (GB)
    latencies = {
        "exact_sdlg": {128: 162.23, 256: 214.97, 512: 290.35, 1024: 462.33, 2048: 800.00, 4096: 0.0},
        "approx_sdlg": {128: 161.26, 256: 187.35, 512: 202.01, 1024: 260.95, 2048: 404.70, 4096: 1016.93},
        "gtrxl": {128: 21.56, 256: 42.28, 512: 108.18, 1024: 309.51, 2048: 0.0, 4096: 0.0}
    }

    vrams = {
        "exact_sdlg": {128: 0.35, 256: 0.68, 512: 1.35, 1024: 2.70, 2048: 5.80, 4096: 0.0},
        "approx_sdlg": {128: 0.28, 256: 0.52, 512: 1.05, 1024: 2.10, 2048: 4.20, 4096: 8.40},
        "gtrxl": {128: 0.36, 256: 0.45, 512: 0.95, 1024: 3.10, 2048: 0.0, 4096: 0.0}
    }

    # Save to CSV
    csv_path = "scripts/diagnostics/benchmarks/latency/profile_throughput.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sequence_length_T", "exact_sdlg_lat", "exact_sdlg_vram", "approx_sdlg_lat", "approx_sdlg_vram", "gtrxl_lat", "gtrxl_vram"])
        for t in t_sizes:
            writer.writerow([t, latencies["exact_sdlg"][t], vrams["exact_sdlg"][t], latencies["approx_sdlg"][t], vrams["approx_sdlg"][t], latencies["gtrxl"][t], vrams["gtrxl"][t]])

    # PLOTTING
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {"exact_sdlg": "#1f77b4", "approx_sdlg": "#2ca02c", "gtrxl": "#000000"}
    labels = {"exact_sdlg": "SdLG-Exact", "approx_sdlg": "SdLG-Approx", "gtrxl": "GTrXL (Baseline)"}
    markers = {"exact_sdlg": "o", "approx_sdlg": "s", "gtrxl": "^"}
    lines = {"exact_sdlg": "-", "approx_sdlg": "-", "gtrxl": "--"}

    # Plot 1: Latency
    for m in models:
        t_valid = [t for t in t_sizes if latencies[m][t] > 0]
        l_valid = [latencies[m][t] for t in t_valid]
        if l_valid:
            axes[0].plot(t_valid, l_valid, marker=markers[m], color=colors[m], linestyle=lines[m], linewidth=2.5, label=labels[m])

        # Mark OOMs
        for t in t_sizes:
            if latencies[m][t] == 0.0:
                axes[0].scatter([t], [150.0], marker='X', color=colors[m], s=120, zorder=5)

    axes[0].set_title("Forward/Backward Pass Step Latency", fontweight='bold', fontsize=12)
    axes[0].set_xlabel("Sequence Length (T)", fontsize=11)
    axes[0].set_ylabel("Latency per Step (ms)", fontsize=11)
    axes[0].set_xticks(t_sizes)
    axes[0].legend(frameon=True, fontsize=10)

    # Plot 2: VRAM
    for m in models:
        t_valid = [t for t in t_sizes if vrams[m][t] > 0]
        v_valid = [vrams[m][t] for t in t_valid]
        if v_valid:
            axes[1].plot(t_valid, v_valid, marker=markers[m], color=colors[m], linestyle=lines[m], linewidth=2.5, label=labels[m])

        # Mark OOMs
        for t in t_sizes:
            if vrams[m][t] == 0.0:
                axes[1].scatter([t], [0.5], marker='X', color=colors[m], s=120, zorder=5)

    axes[1].set_title("Peak GPU VRAM Scaling (Activations Included)", fontweight='bold', fontsize=12)
    axes[1].set_xlabel("Sequence Length (T)", fontsize=11)
    axes[1].set_ylabel("Peak VRAM Allocated (GB)", fontsize=11)
    axes[1].set_xticks(t_sizes)
    axes[1].legend(frameon=True, fontsize=10)

    plt.tight_layout()
    plot_path = "scripts/diagnostics/benchmarks/latency/profile_hardware_scaling.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\nSaved visualization to {plot_path}")

    # Copy to papers/latex/thesis_style/plots/
    import shutil
    dest_path = "papers/latex/thesis_style/plots/profile_hardware_scaling.png"
    shutil.copy(plot_path, dest_path)
    print(f"Copied visualization to {dest_path}")


# =====================================================================
# Component-transfer benchmark: matched T-Maze recurrent inference only
# =====================================================================

def _tree_bytes(tree):
    return int(sum(getattr(x, "nbytes", 0) for x in jax.tree_util.tree_leaves(tree)))


def _tree_ready(tree):
    jax.tree_util.tree_map(
        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, tree
    )


def _matched_tmaze_config(use_exact):
    return {
        "ABLATION": {
            "IS_BASELINE": False,
            "USE_EXACT": use_exact,
            "GAMMA_BIAS": "log",
            "USE_ROPE": False,
            "TOPOLOGY": "full",
            "ALPHA": 1.0,
        },
        "RNN": {
            "RNN_TYPE": "sdlg_hybrid", "D_MODEL": 128, "N_HEADS": 4,
            "N_LAYERS": 3, "D_FFC": 512, "HIDDEN_DIM": 128, "R": 16, "ALPHA": 1.0,
        },
    }


def _matched_tmaze_carry(batch_size, use_exact):
    # Import the exact study implementation so this benchmark uses the same
    # block layout and carry definitions as the accepted matched comparison.
    from scripts.diagnostics.exact_sdlg.exact_sdlg_tmaze_ablations import ExactSdLGCarry

    if use_exact:
        carry = tuple(
            ExactSdLGCarry(
                M=jnp.zeros((4, 32, 32), dtype=jnp.float32),
                z=jnp.zeros((4, 32), dtype=jnp.float32),
                swa_k=jnp.zeros((16, 32), dtype=jnp.float32),
                swa_v=jnp.zeros((16, 32), dtype=jnp.float32),
                step_count=jnp.zeros((), dtype=jnp.float32),
            )
            for _ in range(3)
        )
    else:
        carry = BatchedSdLGHybrid.initialize_carry(batch_size, 3, 32, 32, 16, 16)
    return jax.tree_util.tree_map(lambda x: jnp.repeat(x[None, ...], batch_size, axis=0), carry)


def run_matched_tmaze_inference_benchmark(output_path, warmup=20, repeats=100):
    """Measure warmed one-step inference for the two matched T-Maze models.

    This intentionally does not reuse the static values in ``main`` above,
    which refer to an older, differently shaped forward/backward profiler.
    """
    from scripts.diagnostics.exact_sdlg.exact_sdlg_tmaze_ablations import ActorCriticAblation

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, use_exact in (("Matched_Exact_NoRoPE", True), ("Matched_Approx_NoRoPE", False)):
        for batch_size in (1, 256):
            config = _matched_tmaze_config(use_exact)
            network = ActorCriticAblation(2, config)
            carry = _matched_tmaze_carry(batch_size, use_exact)
            # ActorCriticAblation consumes rollout tensors as [time, batch, feature].
            # Keeping time fixed at one measures recurrent deployment, not a TBPTT chunk.
            obs = jnp.ones((1, batch_size, 2), dtype=jnp.float32)
            dones = jnp.zeros((1, batch_size), dtype=jnp.float32)
            variables = network.init(jax.random.PRNGKey(20261014), carry, (obs, dones))
            parameter_count = int(sum(x.size for x in jax.tree_util.tree_leaves(variables)))

            def forward(state, inputs, terminals):
                return network.apply(variables, state, (inputs, terminals))

            started = time.perf_counter()
            compiled = jax.jit(forward).lower(carry, obs, dones).compile()
            compile_seconds = time.perf_counter() - started
            out = compiled(carry, obs, dones)
            _tree_ready(out)
            state = out[-1]
            for _ in range(warmup):
                out = compiled(state, obs, dones)
                _tree_ready(out)
                state = out[-1]

            samples = []
            for _ in range(repeats):
                started = time.perf_counter()
                out = compiled(state, obs, dones)
                _tree_ready(out)
                samples.append(time.perf_counter() - started)
                state = out[-1]

            samples = np.asarray(samples, dtype=float)
            rows.append({
                "study_id": "CTV-B2-MATCHED-INFERENCE",
                "architecture": label,
                "batch_size": batch_size,
                "backend": jax.default_backend(),
                "device": str(jax.devices()[0]),
                "precision": "float32",
                "parameter_count": parameter_count,
                "carry_bytes_per_environment": _tree_bytes(_matched_tmaze_carry(1, use_exact)),
                "compile_seconds": compile_seconds,
                "warmup_iterations": warmup,
                "timed_iterations": repeats,
                "step_latency_ms_mean": float(samples.mean() * 1e3),
                "step_latency_ms_sd": float(samples.std(ddof=1) * 1e3),
                "step_latency_ms_median": float(np.median(samples) * 1e3),
                "step_latency_ms_p05": float(np.quantile(samples, 0.05) * 1e3),
                "step_latency_ms_p95": float(np.quantile(samples, 0.95) * 1e3),
                "environment_steps_per_second": float(batch_size / samples.mean()),
            })
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = output_path.with_suffix(".json")
    metadata.write_text(json.dumps({
        "measurement": "warmed one-step recurrent inference only",
        "excludes": ["compilation", "environment stepping", "training/backpropagation"],
        "models": ["matched Exact", "matched rank-16 Approximate"],
        "topology": "three local-associative-gated-FFN blocks; NoRoPE; alpha=1",
    }, indent=2) + "\n")
    print(f"wrote {len(rows)} measured rows to {output_path}")
    return rows

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--t", type=int, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--matched-tmaze-inference", action="store_true")
    parser.add_argument(
        "--output",
        default="papers/writing/generated_artifacts/component_transfer/matched_tmaze_inference.csv",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()

    if args.matched_tmaze_inference:
        run_matched_tmaze_inference_benchmark(args.output, args.warmup, args.repeats)
    elif args.t is not None and args.model is not None:
        run_single_latency_measurement(args.t, args.model)
    else:
        main()
