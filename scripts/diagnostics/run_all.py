import subprocess
import os
import sys
import time
import json

# Hardware tuning specifically for RDNA2 (RX 6800 XT) under ROCm
ENV_VARS = {
    "HSA_OVERRIDE_GFX_VERSION": "10.3.0",
    "MIOPEN_FIND_MODE": "2",
    "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
    "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.60",
    "PYTHONUNBUFFERED": "1",
}

DIAGNOSTICS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(DIAGNOSTICS_DIR))

ACTIVE_EXPERIMENTS = {
    # --- Fast Profiling & Diagnostic Scripts ---
    #"benchmarks/latency/profile_throughput.py": {"tags": ["latency", "profiling"], "notes": "Measures hardware O(1) vs O(T^2) execution limits."}, # done but it crashed for the exact version even though it shouldnt
    #"safety_and_compilation/numerical_safety/numerical_safety.py": {"tags": ["compilation", "safety", "numerical"], "notes": "JAX compilation and numerical stability analysis."}, worked is in chapter 4
    #"architecture/memory_scaling/memory_scaling.py": {"tags": ["memory", "scaling"], "notes": "Mathematical evaluation of memory scaling O(rd) vs O(T^2)."},

    # --- Short Ablation & Classic Sweeps (15M - 25M Steps) ---
    #"architecture/swa_ablation/swa_ablation.py": {"tags": ["ablation", "swa"], "notes": "Ablation study removing local Stochastic Weight Averaging."},
    #"architecture/gradient_shielding/gradient_shielding.py": {"tags": ["ablation", "ggs", "gradient_shielding"], "notes": "Ablation study on Gradient Shielding (GGS) scaling."},

    # --- Medium Horizon & Key Benchmark sweeps (50M Steps) ---
    #"exact_sdlg/exact_sdlg_xminigrid_keycorridor.py": {"tags": ["exact_sdlg", "ablation"], "notes": "Xminigrid sweep"}, we have the results
    #"benchmarks/rliable_aggregation.py": {"tags": ["statistics", "iqm"],"notes": "Fetches cross-suite runs from W&B and computes IQM bounds."},

    #"benchmarks/xminigrid/xminigrid_eval.py": {"tags": ["exact_sdlg", "xminigrid", "ppo"], "notes": "Ultra-Long Horizon Exact GLA XMiniGrid Benchmark"},
    #"benchmarks/mystery_path/mystery_path.py": {"tags": ["mystery_path", "benchmark"], "notes": "Mystery Path evaluation"},
    #"benchmarks/craftax_eval_ablation/eval_ablation.py": {"tags": ["ablation", "eval", "craftax"], "notes": "Evaluation Ablation on Craftax."},
    #"exact_sdlg/exact_sdlg_craftax_classic.py": {"tags": ["exact_sdlg", "ablation"], "notes": "CraftaxClassic"},

    # --- Phase 1: New Empirical Ablations ---
    #"exact_sdlg/exact_sdlg_tmaze_ablations.py": {"tags": ["tmaze", "ablation", "rank_sweep"], "notes": "Rank sweep for SdLG-Approx on T-Maze"},
    #"exact_sdlg/ablation_minesweeper_sdlg.py": {"tags": ["minesweeper", "ablation", "sdlg_architecture"], "notes": "Architectural ablation suite for SdLG on Minesweeper (Rank & Recurrent Depth)."},
    "exact_sdlg/popgym_benchmark.py": {"tags": ["popgym", "benchmark"], "notes": "POPGym POMDP benchmark (Online PPO, T=16)."},

    #"benchmarks/craftax_full/craftax_classic_benchmark.py": {"tags": ["benchmark"], "notes": "50M steps benchmark on Craftax-Classic."},

    # --- Heavy Long-Horizon Benchmark & Ablation Sweeps (300M Steps) ---
    #"architecture/moe/moe_ablation.py": {"tags": ["ablation", "moe"], "notes": "MoE Sparse Scaling Benchmark"}, # is the same as the craftax_full script
    #"benchmarks/craftax_full/craftax_full.py": {"tags": ["benchmark", "300M"], "oh i notes": "300M steps benchmark on Craftax."},
    #"exact_sdlg/exact_sdlg_craftax_ablations.py": {"tags": ["exact_sdlg", "ablation"], "notes": "Craftax sweep"},

    # --- Excluded / Diagnostic-only / Undocumented Experiments ---
    # "mathematics/softplus_normalization/softplus_normalization.py": {"tags": ["softplus", "normalization"], "notes": "Bound characteristics of Softplus normalization."},
    # "mathematics/rank_normalization/rank_normalization.py": {"tags": ["rank", "normalization", "equivalence"], "notes": "Parallel vs Sequential normalization equivalence."},
    # "mathematics/boundary_reset/boundary_reset.py": {"tags": ["boundary", "reset", "dynamics"], "notes": "Validation of sequence boundary resets."},
    # "mathematics/gradient_scaling/gradient_scaling.py": {"tags": ["gradient", "scaling", "bptt"], "notes": "Analysis of BPTT gradient retention and stability."},
    # "architecture/gating_ablation/gating_ablation.py": {"tags": ["ablation", "decay"],"notes": "Decay Mechanism Showdown (Sigmoid GLA vs Mamba SSM vs Static LTI)."},
    # "architecture/gating_ablation/decay_ablation_tmaze.py": {"tags": ["tmaze","ablation", "decay"],"notes": "Decay Mechanism Showdown (Sigmoid GLA vs Mamba SSM vs Static LTI)."},
    # "architecture/representation_geometry/no_rope_ablation.py": {"tags": ["ablation", "representation"],"notes": "Representation Geometry Showdown. Compares RoPE (Geometric) vs Tikhonov (Algebraic) vs Scalar Decay for preventing rank collapse."},
    # "architecture/representation_geometry/tmaze_showdown.py": {"tags": ["tmaze", "ablation", "representation"],"notes": "Representation Geometry Showdown. Compares RoPE (Geometric) vs Tikhonov (Algebraic) vs Scalar Decay for preventing rank collapse."},
    #"architecture/linear_probing/probe.py": {"tags": ["interpretability", "linear_probe"], "notes": "Extracts memory matrix and fits L2 logistic regression."},
    #"architecture/representation_geometry/extract_gate_telemetry.py": {"tags": ["interpretability", "telemetry"], "notes": "Extracts gate telemetry."},
}

def run_experiment(script_name, wandb_config, log_file_path, attempt, timeout_inactivity=600):
    script_path = os.path.join(DIAGNOSTICS_DIR, script_name)
    python_bin = os.path.join(PROJECT_ROOT, ".venv", "bin", "python3")
    cmd = f"{python_bin} {script_path}"

    env = os.environ.copy()
    env.update(ENV_VARS)
    env["WANDB_PROJECT"] = "PopGym-Benchmark"
    if "tags" in wandb_config:
        env["WANDB_TAGS"] = ",".join(wandb_config["tags"])
    if "notes" in wandb_config:
        env["WANDB_NOTES"] = wandb_config["notes"]

    t0 = time.time()
    with open(log_file_path, "a") as log_file:
        log_file.write(f"\n=== Starting {script_name} (Attempt {attempt}) ===\n")
        log_file.write(f"Command: {cmd}\n")
        log_file.write("===================================================\n\n")
        log_file.flush()

        proc = subprocess.Popen(
            cmd,
            shell=True,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT
        )

        last_size = 0
        last_change_time = time.time()

        while proc.poll() is None:
            time.sleep(5)
            try:
                current_size = os.path.getsize(log_file_path)
            except OSError:
                current_size = last_size

            if current_size > last_size:
                last_size = current_size
                last_change_time = time.time()
            else:
                inactive_duration = time.time() - last_change_time
                if inactive_duration > timeout_inactivity:
                    log_file.write(f"\n⚠️ [WATCHDOG TIMEOUT] No log activity detected for {inactive_duration:.1f}s. Terminating hung process (PID {proc.pid})...\n")
                    log_file.flush()
                    try:
                        proc.kill()
                        proc.wait(timeout=5)
                    except Exception:
                        pass
                    duration = time.time() - t0
                    return False, -9, duration

        retcode = proc.returncode

    duration = time.time() - t0
    return retcode == 0, retcode, duration


def main():
    print("====================================================")
    print("      DLRhybrid IMMORTAL VALIDATION RUNNER          ")
    print("====================================================")

    results = {}
    summary_path = os.path.join(PROJECT_ROOT, "run_summary.json")

    for exp, wandb_config in ACTIVE_EXPERIMENTS.items():
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        safe_name = exp.replace("/", "_").replace("\\", "_").replace(".py", "")
        script_dir = os.path.join(DIAGNOSTICS_DIR, os.path.dirname(exp))
        log_file_path = os.path.join(script_dir, f"{safe_name}_{timestamp}.log")

        max_retries = 100
        retries = 0

        while retries < max_retries:
            print(f"\n🚀 Launching run: {exp} (Attempt {retries + 1}/{max_retries})...")
            passed, retcode, duration = run_experiment(exp, wandb_config, log_file_path, retries + 1)

            if passed:
                print(f"✅ {exp} passed successfully in {duration:.2f} seconds.")
                results[exp] = {"status": "PASSED", "duration": duration, "log_file": log_file_path}
                break
            elif retcode == 43:
                print(f"♻️  [PLANNED FLUSH] {exp} requested RAM flush (Exit Code 43).")
                time.sleep(5)
            elif retcode == 42:
                print(f"❌ [FATAL] {exp} halted due to unrecoverable NaN! Not restarting.")
                results[exp] = {"status": "FAILED (NaN)", "duration": duration, "log_file": log_file_path}
                break
            else:
                print(f"⚠️ {exp} crashed unexpectedly (Exit Code {retcode}). Logs: {log_file_path}")
                retries += 1
                time.sleep(5)

        with open(summary_path, "w") as f:
            json.dump(results, f, indent=4)

    print("\n====================================================")
    for exp, data in results.items():
        print(f" - {exp:<30} : {data.get('status', 'FAILED'):<15}")
    print("====================================================")

if __name__ == "__main__":
    main()
