import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set publication aesthetics suitable for ICML / Master's thesis
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Liberation Serif'],
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 15,
    'figure.autolayout': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})

# Color palette definition for models
COLOR_MAP = {
    'SdLG-Exact': '#1f77b4',   # Deep Blue
    'SdLG-Approx': '#d62728',  # Crimson Red
    'PPO-Mamba': '#9467bd',   # Purple
    'PPO-GRU': '#ff7f0e',     # Orange
    'PPO-GLA': '#2ca02c',     # Green
}

MODEL_ORDER = ['SdLG-Exact', 'SdLG-Approx', 'PPO-Mamba', 'PPO-GRU', 'PPO-GLA']

def get_normalized_score(env_name, return_val, H):
    """Normalize raw episodic return to [0.0, 1.0] standard score."""
    if 'CartPole' in env_name or 'Pendulum' in env_name:
        max_ret = 0.005 * H
        min_ret = 0.0
    elif 'Concentration' in env_name or 'Minesweeper' in env_name or 'MineSweeper' in env_name:
        max_ret = 1.0
        min_ret = 0.0
    else:
        max_ret = 1.0
        min_ret = -1.0
    score = (return_val - min_ret) / max(1e-5, max_ret - min_ret)
    return float(np.clip(score, 0.0, 1.0))

def load_data():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    csv_candidates = [
        os.path.join(project_root, "results", "popgym_results.csv"),
        os.path.join(project_root, "popgym_results.csv"),
        "results/popgym_results.csv",
        "popgym_results.csv"
    ]
    
    csv_path = None
    for path in csv_candidates:
        if os.path.exists(path):
            csv_path = path
            break
            
    if not csv_path:
        raise FileNotFoundError("Could not locate popgym_results.csv in results/ or root.")
        
    print(f"Loading data from: {csv_path}")
    
    lines = open(csv_path, "r").readlines()
    rows = []
    for line in lines[1:]:
        parts = line.strip().split(',')
        if len(parts) in (9, 11):
            env, model, seed, step = parts[0].strip(), parts[1].strip(), int(parts[2]), int(parts[3])
            tr, er100, er200 = float(parts[4]), float(parts[5]), float(parts[6])
            
            if len(parts) == 11:
                en100 = float(parts[7]) if parts[7] != '' else get_normalized_score(env, er100, 100)
                en200 = float(parts[8]) if parts[8] != '' else get_normalized_score(env, er200, 200)
                pe = float(parts[9]) if parts[9] != '' else 0.0
                sps = float(parts[10]) if parts[10] != '' else 0.0
            else:
                en100 = get_normalized_score(env, er100, 100)
                en200 = get_normalized_score(env, er200, 200)
                pe = float(parts[7]) if parts[7] != '' else 0.0
                sps = float(parts[8]) if parts[8] != '' else 0.0
                
            rows.append({
                'Environment': env,
                'Model': model,
                'Seed': seed,
                'Step': step,
                'Train_Return': tr,
                'Eval_Return_H100': er100,
                'Eval_Return_H200': er200,
                'Eval_NormScore_H100': en100,
                'Eval_NormScore_H200': en200,
                'Policy_Entropy': pe,
                'SPS': sps
            })
            
    df = pd.DataFrame(rows)
    return df

def generate_interim_figure():
    df = load_data()
    
    # Identify completed environments (having full model sweeps across seeds)
    completed_envs = ['PositionOnlyCartPoleEasy', 'RepeatPreviousEasy']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("POPGym Benchmark Interim Analysis: Convergence & Zero-Shot Length Extrapolation", fontsize=16, fontweight='bold', y=0.98)
    
    for row_idx, env_name in enumerate(completed_envs):
        sub_df = df[df['Environment'] == env_name].copy()
        
        # Subplot 1: Learning Curve (Step vs Eval NormScore H100)
        ax_lc = axes[row_idx, 0]
        
        models_in_env = [m for m in MODEL_ORDER if m in sub_df['Model'].unique()]
        
        for model in models_in_env:
            m_df = sub_df[sub_df['Model'] == model]
            
            # Group by Step across seeds
            grp = m_df.groupby('Step')['Eval_NormScore_H100']
            steps = grp.mean().index / 1e6  # Step in Millions
            mean_score = grp.mean().values
            std_score = grp.std().fillna(0).values
            
            color = COLOR_MAP.get(model, '#333333')
            ax_lc.plot(steps, mean_score, label=model, color=color, linewidth=2.2)
            ax_lc.fill_between(steps, np.maximum(0, mean_score - std_score), np.minimum(1.0, mean_score + std_score),
                            color=color, alpha=0.15)
            
        env_title_map = {
            'PositionOnlyCartPoleEasy': 'PositionOnlyCartPole-Easy',
            'RepeatPreviousEasy': 'RepeatPrevious-Easy'
        }
        display_title = env_title_map.get(env_name, env_name)
        
        ax_lc.set_title(f"({chr(97 + row_idx*2)}) {display_title} — Training Convergence", fontweight='bold')
        ax_lc.set_xlabel("Environment Steps (Millions)")
        ax_lc.set_ylabel("Eval Normalized Score (H=100)")
        ax_lc.set_ylim(-0.02, 1.05)
        ax_lc.legend(frameon=True, loc='lower right', facecolor='white', framealpha=0.9)
        ax_lc.grid(True, linestyle='--', alpha=0.5)
        
        # Subplot 2: Zero-Shot Length Extrapolation (Final Step Grouped Bar Chart)
        ax_bar = axes[row_idx, 1]
        
        # Filter for final evaluation step per seed
        idx_final = sub_df.groupby(['Model', 'Seed'])['Step'].idxmax()
        final_df = sub_df.loc[idx_final]
        
        # Compute mean and std for H100 and H200 per model
        bar_data = []
        for model in models_in_env:
            mod_final = final_df[final_df['Model'] == model]
            h100_mean = mod_final['Eval_NormScore_H100'].mean()
            h100_std = mod_final['Eval_NormScore_H100'].std() if len(mod_final) > 1 else 0.0
            h200_mean = mod_final['Eval_NormScore_H200'].mean()
            h200_std = mod_final['Eval_NormScore_H200'].std() if len(mod_final) > 1 else 0.0
            
            bar_data.append({
                'Model': model,
                'H100_mean': h100_mean,
                'H100_std': h100_std if not np.isnan(h100_std) else 0.0,
                'H200_mean': h200_mean,
                'H200_std': h200_std if not np.isnan(h200_std) else 0.0
            })
            
        bar_df = pd.DataFrame(bar_data)
        
        x = np.arange(len(models_in_env))
        bar_width = 0.35
        
        # Plot H100 (In-domain horizon) and H200 (Extrapolated horizon)
        rects1 = ax_bar.bar(x - bar_width/2, bar_df['H100_mean'], bar_width,
                           yerr=bar_df['H100_std'], capsize=4,
                           label='Eval H=100 (In-Domain)',
                           color=[COLOR_MAP.get(m, '#333333') for m in models_in_env],
                           edgecolor='black', linewidth=0.8, alpha=0.95)
                           
        rects2 = ax_bar.bar(x + bar_width/2, bar_df['H200_mean'], bar_width,
                           yerr=bar_df['H200_std'], capsize=4,
                           label='Eval H=200 (Zero-Shot 2x)',
                           color=[COLOR_MAP.get(m, '#333333') for m in models_in_env],
                           edgecolor='black', linewidth=0.8, alpha=0.45, hatch='//')
                           
        ax_bar.set_title(f"({chr(98 + row_idx*2)}) {display_title} — Zero-Shot Horizon Extrapolation (H=100 vs H=200)", fontweight='bold')
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(models_in_env, rotation=15, ha='right')
        ax_bar.set_ylabel("Normalized Score [0.0, 1.0]")
        ax_bar.set_ylim(0.0, 1.15)
        ax_bar.legend(frameon=True, loc='upper right', facecolor='white', framealpha=0.9)
        ax_bar.grid(True, linestyle='--', alpha=0.5, axis='y')
        
        # Annotate bars with mean values for precision
        for i, row in bar_df.iterrows():
            # H100 label
            ax_bar.text(i - bar_width/2, row['H100_mean'] + row['H100_std'] + 0.02,
                        f"{row['H100_mean']:.2f}", ha='center', va='bottom', fontsize=8, fontweight='bold')
            # H200 label
            ax_bar.text(i + bar_width/2, row['H200_mean'] + row['H200_std'] + 0.02,
                        f"{row['H200_mean']:.2f}", ha='center', va='bottom', fontsize=8, color='#333333')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure to requested destination paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    out_paths = [
        os.path.join(project_root, "popgym_interim_analysis.pdf"),
        os.path.join(project_root, "results", "popgym_interim_analysis.pdf"),
        os.path.join(project_root, "scripts", "diagnostics", "popgym_interim_analysis.pdf"),
    ]
    
    # Also attempt to save to papers directory if it exists
    thesis_plot_dir = os.path.join(project_root, "papers", "latex", "thesis_style", "plots")
    if os.path.exists(thesis_plot_dir):
        out_paths.append(os.path.join(thesis_plot_dir, "popgym_interim_analysis.pdf"))
        
    for path in out_paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to: {path}")
        
    plt.close()
    print("Plot generation complete!")

if __name__ == "__main__":
    generate_interim_figure()
