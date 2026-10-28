import glob, os
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

project_root = '.'
files = glob.glob(os.path.join(project_root, 'checkpoints/craftax_50m_watchdog_T*/*_log.csv'))
name_mapping = {
    'SdLGHybrid_T16': 'SdLG-Approx (Ours)',
    'PPO-GRU_T64': 'PPO-GRU',
    'PPO-GTrXL_T64': 'PPO-GTrXL',
    'AGaLiTe_T64': 'AGaLiTe',
    'PPO-Dense_T16': 'PPO-Dense',
    'PPO-Mamba_T64': 'PPO-Mamba',
}
final_returns = {}

for f in sorted(files):
    dir_name = os.path.basename(os.path.dirname(f))
    bname = os.path.basename(f).replace('_log.csv', '')
    if '_seed_' in bname:
        bname = bname.split('_seed_')[0]
    
    model_name = name_mapping.get(bname, bname)
    df = pd.read_csv(f)
    if 'Train_Return' in df.columns and len(df) > 0:
        final_returns.setdefault(model_name, []).append(df['Train_Return'].iloc[-1])

print("Craftax-Classic 50M Returns:")
for model, rets in final_returns.items():
    print(f"{model}: {rets} (Mean: {np.mean(rets):.2f})")

def prob_of_improvement(x, y):
    """Computes Probability of Improvement P(X > Y) using Mann-Whitney U test."""
    stat, p = mannwhitneyu(x, y, alternative='greater')
    u = stat
    n_x = len(x)
    n_y = len(y)
    poi = u / (n_x * n_y)
    return poi

sdlg_rets = final_returns['SdLG-Approx (Ours)']
for baseline in ['PPO-GTrXL', 'PPO-Mamba', 'PPO-GRU', 'AGaLiTe']:
    baseline_rets = final_returns[baseline]
    poi = prob_of_improvement(sdlg_rets, baseline_rets)
    print(f"P(SdLG-Approx > {baseline}) = {poi:.3f}")
