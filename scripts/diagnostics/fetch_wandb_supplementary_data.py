import wandb
import pandas as pd
import os
import argparse
from tqdm import tqdm

# Mapping of experiments to their identifying WandB tags, projects, and metric patterns
EXPERIMENTS = {
    "Craftax_1B_300M": {
        "project": "ecraftax_classic_ablation",
        "tags": ["300M"],
        "metrics_patterns": [
            "Train/Mean_Return",
            "Train/Normalized_Return_Pct",
            "Train/Policy_Entropy",
            "Train/Value_Loss",
            "Train/Mean_Episode_Length",
            "Train/Max_Grad_Norm",
            "Train/Total_Achievements",
            "Train_Achv/",
            "System/SPS",
            "Eval/Mean_Return",
            "Train_Env/returned_episode_returns",
            "Train_Env/returned_episode_lengths",
            "Update/entropy",
            "Train_Achv/achievements_unlocked"
        ],
        "name_filter": lambda name: True
    },
    "Craftax_Classic_Exact_vs_Approx": {
        "project": "ecraftax_classic_ablation",
        "tags": ["exact_sdlg", "ablation"],
        "metrics_patterns": [
            "Train/Mean_Return",
            "Train/Mean_Length",
            "Train/Policy_Entropy",
            "Train/Value_Loss",
            "System/SPS",
            "Train_Env/returned_episode_returns",
            "Train_Env/returned_episode_lengths",
            "Update/entropy"
        ],
        "name_filter": lambda name: any(name.startswith(p) for p in [
            "Control_Exact_Dense", 
            "Ablation_Approx_Dense", 
            "Ablation_GammaZero", 
            "Ablation_GRUBias_0"
        ])
    },
    "Craftax_Classic_Capacity_Scaling": {
        "project": "ecraftax_classic_ablation",
        "tags": ["exact_sdlg", "ablation"],
        "metrics_patterns": [
            "Train/Mean_Return",
            "Train/Mean_Length",
            "Train/Policy_Entropy",
            "Train/Value_Loss",
            "System/SPS",
            "Train_Env/returned_episode_returns",
            "Train_Env/returned_episode_lengths",
            "Update/entropy"
        ],
        "name_filter": lambda name: name.startswith("Capacity_") or name.startswith("Control_Exact_Dense")
    },
    "Craftax_Classic_BPTT": {
        "project": "ecraftax_classic_ablation",
        "tags": ["ggs", "gradient_shielding", "bptt"],
        "metrics_patterns": [
            "Train/Mean_Return",
            "Train/Policy_Entropy",
            "Train/Value_Loss_Var",
            "System/SPS",
            "Diagnostics/Gradient_Cosine_Similarity",
            "Eval/Mean_Return",
            "Train_Env/returned_episode_returns",
            "Train_Env/returned_episode_lengths",
            "Update/entropy"
        ],
        "name_filter": lambda name: True
    },
    "MiniGrid_SWA_Ablation": {
        "project": "KeyCorridorXminigrid_Ablations",
        "tags": ["swa", "swa_ablation"],
        "metrics_patterns": [
            "Train/Mean_Return",
            "Train/Policy_Entropy",
            "Train/Value_Loss_Var",
            "System/SPS",
            "Eval/Mean_Return",
            "Train_Env/returned_episode_returns",
            "Train_Env/returned_episode_lengths",
            "Update/entropy"
        ],
        "name_filter": lambda name: True
    },
    "TMaze_Extrapolation": {
        "project": "Tmaze_Ablations",
        "tags": ["EXP18", "tmaze"],
        "metrics_patterns": [
            "Train/Mean_Return",
            "Train/Mean_Length",
            "Train/Policy_Entropy",
            "Train/Value_Loss",
            "System/SPS",
            "Eval/Success_H",
            "Train_Env/returned_episode_returns",
            "Train_Env/returned_episode_lengths",
            "Update/entropy"
        ],
        "name_filter": lambda name: True
    }
}


def fetch_data(entity, default_project, output_dir):
    api = wandb.Api()
    os.makedirs(output_dir, exist_ok=True)

    print(f"Fetching data using WandB entity: {entity}...")

    for exp_name, config in EXPERIMENTS.items():
        project = config.get("project", default_project)
        tags = config["tags"]
        patterns = config["metrics_patterns"]
        name_filter = config["name_filter"]

        print(f"\nProcessing {exp_name} (Project: {project}, Tags: {tags})...")

        # Fetch runs containing any of the tags in that project
        try:
            runs = api.runs(f"{entity}/{project}", {"tags": {"$in": tags}})
        except Exception as e:
            print(f"  Error fetching runs for project {project}: {e}")
            continue

        # Apply name filtering
        filtered_runs = [run for run in runs if name_filter(run.name)]

        if not filtered_runs:
            print(f"  No runs found for {exp_name} after filtering.")
            continue

        print(f"  Found {len(filtered_runs)} runs to process.")
        exp_dir = os.path.join(output_dir, exp_name)
        os.makedirs(exp_dir, exist_ok=True)

        for run in tqdm(filtered_runs, desc=f"Downloading {exp_name} runs"):
            # Discover all summary keys matching target patterns
            discovered_metrics = set()
            summary_keys = list(run.summary.keys())
            
            for pat in patterns:
                for k in summary_keys:
                    if k.startswith(pat) or pat in k:
                        discovered_metrics.add(k)

            # Auto-discover any evaluation or system performance metrics
            for k in summary_keys:
                k_lower = k.lower()
                if k.startswith("Eval") or "eval" in k_lower:
                    discovered_metrics.add(k)
                if "sps" in k_lower or "value_loss" in k_lower or "entropy" in k_lower:
                    discovered_metrics.add(k)
                if k.startswith("System/"):
                    discovered_metrics.add(k)

            metrics_list = sorted(list(discovered_metrics))
            
            try:
                # Query full history first (without keys) to prevent WandB inner-join/filtering gotchas
                history = run.history(pandas=True, samples=2000)
                if not history.empty:
                    # Filter to keep only step and discovered metrics
                    cols_to_keep = [c for c in ["_step"] + metrics_list if c in history.columns]
                    history = history[cols_to_keep]

                    # Align step name columns to ensure Total_Steps, env_steps, and Environment_Steps exist
                    if "_step" in history.columns:
                        history["Total_Steps"] = history["_step"]
                        history["env_steps"] = history["_step"]
                        history["Environment_Steps"] = history["_step"]
                    
                    safe_run_name = run.name.replace("/", "_").replace("\\", "_")
                    csv_path = os.path.join(exp_dir, f"{safe_run_name}.csv")
                    history.to_csv(csv_path, index=False)
            except Exception as e:
                print(f"  Error fetching history for run {run.name}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch W&B telemetry for LaTeX appendix plots.")
    parser.add_argument("--entity", type=str, required=True, help="Your WandB entity (username or team).")
    parser.add_argument("--project", type=str, default="ecraftax_classic_ablation", help="Default WandB project name.")
    parser.add_argument("--out", type=str, default="wandb_exports", help="Output directory for CSV files.")

    args = parser.parse_args()
    fetch_data(args.entity, args.project, args.out)
    print("\nData extraction complete! You can now use these CSVs to generate your LaTeX plots.")
