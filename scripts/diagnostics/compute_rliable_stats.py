import sys
import pandas as pd
import numpy as np
import scipy.stats
import wandb

def bootstrap_iqm(data, n_resamples=2000):
    """Calculates the Interquartile Mean (IQM) and 95% Confidence Interval using standard bootstrapping."""
    if len(data) == 0:
        return np.nan, np.nan, np.nan

    def iqm_stat(x):
        return scipy.stats.trim_mean(x, proportiontocut=0.25)

    # Actual IQM point estimate
    iqm = iqm_stat(data)

    # Bootstrap CI
    res = scipy.stats.bootstrap((data,), statistic=iqm_stat, n_resamples=n_resamples, confidence_level=0.95, method='percentile')
    return iqm, res.confidence_interval.low, res.confidence_interval.high

def compute_ablation_stats():
    print("Connecting to Weights & Biases API...")
    try:
        api = wandb.Api()
    except wandb.errors.UsageError:
        print("\n[ERROR] WandB is not authenticated! Please run 'wandb login' or set WANDB_API_KEY.")
        sys.exit(1)

    entity = "schweitzer-tim-rwu-hochschule-ravensburg-weingarten-univ"
    project = "ecraftax_classic_ablation"

    try:
        runs = api.runs(f"{entity}/{project}")
    except Exception as e:
        print(f"\n[ERROR] Could not fetch W&B data: {e}")
        print("Please ensure you have internet access and the project exists on WandB.")
        sys.exit(1)

    final_returns = {}

    print("Fetching final returns for Craftax-Classic ablations...")
    for run in runs:
        run_name = run.name
        if "-seed-" not in run_name:
            continue

        condition = run_name.rsplit("-seed-", 1)[0]

        # Fast extraction: get the summary value directly instead of downloading full history
        final_return = run.summary.get("Train/Mean_Return", None)
        if final_return is None:
            # Fallback if summary is missing
            history = run.history(keys=["Train/Mean_Return"], samples=10)
            if not history.empty:
                final_return = history["Train/Mean_Return"].dropna().iloc[-1]

        if final_return is not None:
            if condition not in final_returns:
                final_returns[condition] = []
            final_returns[condition].append(final_return)

    if not final_returns:
        print("\n[ERROR] No data found in the W&B project!")
        print(f"Make sure to run the training script (exact_sdlg_craftax_ablations.py) and sync to {entity}/{project}.")
        sys.exit(1)

    print("\n" + "="*50)
    print("  IQM & 95% CI CALCULATIONS (CRAFTAX-CLASSIC)")
    print("="*50)

    # Process and print stats
    for condition, values in final_returns.items():
        data = np.array(values)
        if len(data) < 3:
            print(f"[{condition}] Need at least 3 seeds for CI. Found {len(data)}. Data: {data}")
            continue

        iqm, ci_low, ci_high = bootstrap_iqm(data)
        print(f"[{condition}]")
        print(f"   Seeds: {len(data)}")
        print(f"   Raw Values: {np.round(data, 2)}")
        print(f"   LaTeX String: an IQM of {iqm:.2f} (95% CI: [{ci_low:.2f}, {ci_high:.2f}])\n")

if __name__ == "__main__":
    compute_ablation_stats()
