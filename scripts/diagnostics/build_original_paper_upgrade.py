#!/usr/bin/env python3
"""Build the original-paper evidence ledger and publication figures.

The original ICML paper is authoritative for reported experiment existence and
multi-seed summaries. Surviving CSVs are used to regenerate values where they
are complete enough, but are not used to invalidate reported experiments.
"""

from __future__ import annotations

from pathlib import Path
import csv
import glob
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


ROOT = Path(__file__).resolve().parents[2]
WRITING = ROOT / "papers" / "writing"
GEN = WRITING / "generated_artifacts" / "original_paper_upgrade"
PLOTS = ROOT / "papers" / "latex" / "tmlr_review" / "plots" / "rebuilt"
GEN.mkdir(parents=True, exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)

MODELS = ["SdLG-Exact", "SdLG-Approx", "PPO-GRU", "PPO-Mamba", "PPO-GLA"]


def experiment_rows():
    cols = [
        "source_table_or_figure", "environment", "task", "models_or_variants",
        "intervention_axis", "seed_count", "reported_seeds_if_known",
        "training_budget", "T", "evaluation_horizon", "metric",
        "reported_mean", "reported_sd", "throughput_metric",
        "parameter_count_if_reported", "main_scientific_question",
        "current_manuscript_overlap", "recommended_use", "raw_artifact_candidates",
        "notes",
    ]
    rows = []
    def add(*values): rows.append(dict(zip(cols, values)))

    add("ICML Fig. 2 / Table 2", "Passive T-Maze", "H=100 train; H=100/150/200 eval",
        "Exact/Approximate x RoPE/NoRoPE", "state execution; positional phase", "10/cell",
        "42,54,99,111,123,222,333,444,456,789", "max 50M; early stop 0.98 x3", "16",
        "100,150,200", "evaluation return; terminal step", "see canonical factorial", "see canonical factorial", "n/a",
        "matched within factorial", "How do execution and phase affect training and extrapolation?", "canonical main result",
        "main text; superseded by validated ten-seed reconstruction", "scripts/diagnostics/architecture/sdlg_v2_*csv", "Tier A; not a rank-only intervention")
    add("ICML Fig. 2 / Appendix D.2", "Passive T-Maze", "component and horizon sweeps",
        "Exact, Approximate, GLA, RoPE variants", "component/state/horizon", "reported multi-seed where shown", "see original manifests",
        "15M--50M", "16", "50,100,150,200", "return/success/entropy/length", "see original figure", "see original figure", "n/a", "reported matched configurations",
        "Which components and horizons alter T-Maze behavior?", "partly superseded", "appendix/supplement", "wandb_exports/TMaze_*; scripts/diagnostics/exact_sdlg/*tmaze*.csv", "retain as supporting; reject causal bootstrap language")
    add("Current Fig. 2 recovered from original runs", "Passive T-Maze", "external H=100 landscape",
        "Mamba; LTI-Linear; GRU; LSTM; Markovian", "architecture family", "6/model", "101,102,103,104,123,456",
        "49,971,200", "16", "100", "fixed-endpoint evaluation return", "see canonical CSV", "see canonical CSV", "n/a",
        "50,435--148,995", "How do external mechanisms behave under severe truncation?", "canonical main result", "main text", "papers/writing/generated_artifacts/tmaze_*", "Tier A descriptive; capacity unmatched")

    add("ICML Table 1", "Craftax Classic Symbolic", "50M macro benchmark T=64",
        "GLA; GRU; GTrXL; AGaLiTe; Mamba; SdLG-Approx; SdLG-Exact", "architecture family/state execution", "5/model", "reported N=5",
        "50M", "64", "native", "evaluation return/IQM", "14.79 Approx; 13.70 Exact; see table for others", "0.36; 0.52", "cumulative training SPS",
        "SdLG Approx 6.11M; Exact 8.78M; baseline counts in current table", "Where do external models and SdLG lie under dense control?", "five baseline rows only",
        "main text: add two SdLG rows", "original Table 1; canonical Craftax CSVs", "protocol matched, capacity unmatched")
    add("ICML Table 1", "Craftax Classic Symbolic", "50M short-unroll benchmark",
        "GRU; SdLG-Approx; Mamba; AGaLiTe; GLA", "rollout length/model", "5/model", "reported N=5", "50M", "16", "native",
        "evaluation return/IQM", "16.44;16.33;15.86;14.44;14.11", "0.11;0.37;0.07;0.61;0.59", "cumulative training SPS", "capacity unmatched",
        "How does the dense-control landscape change at short T?", "absent", "appendix table; discussion", "original Table 1", "do not pool with T=64 ranking")
    add("ICML Table 1", "Craftax Classic Symbolic", "extended/intermediate T probes",
        "GRU T128; AGaLiTe T32; GTrXL T128", "rollout length/model", "5/model", "reported N=5", "50M", "32/128", "native",
        "evaluation return/IQM", "15.62;14.99;13.83", "0.11;0.26;0.68", "cumulative training SPS", "capacity unmatched",
        "How do selected baselines respond to T?", "absent", "appendix table", "original Table 1", "sparse probe grid; no universal T law")
    add("Thesis Fig. 6.1 / Appendix Craftax exact-v-approx", "Craftax Classic", "historical 25M controlled protocol",
        "SdLG-Exact; SdLG-Approx", "state execution", "3/condition", "42,54,99", "24,576,000", "64", "native", "Train/Mean_Return at common endpoint",
        "13.664;13.678", "0.155;0.804", "System/SPS", "matched family", "Does state execution matter under dense-control learning dynamics?", "absent",
        "main-text design-ablation panel", "wandb_exports/Craftax_Classic_Exact_vs_Approx", "separate protocol from canonical evaluation landscape")
    add("Thesis Fig. 6.4", "Craftax Classic", "temporal prior ablation", "control; GammaZero", "decay/temporal prior", "3/condition", "42,54,99", "24,576,000", "64", "native",
        "Train/Mean_Return at common endpoint", "13.664;10.877", "0.155;0.837", "System/SPS", "matched family", "Does the temporal-prior configuration alter learning?", "absent",
        "main-text design-ablation panel", "wandb_exports/Craftax_Classic_Exact_vs_Approx", "association only; no mechanistic decay claim")
    add("Thesis Fig. 6.5", "Craftax Classic", "residual-gate bias ablation", "control; GRU bias 0", "residual-gate initialization", "3/condition", "42,54,99", "24,576,000", "64", "native",
        "Train/Mean_Return at common endpoint", "13.664;12.557", "0.155;0.264", "System/SPS", "matched family", "Does residual-gate initialization alter learning?", "absent",
        "main-text design-ablation panel", "wandb_exports/Craftax_Classic_Exact_vs_Approx", "avoid gradient-confusion causality")
    add("Thesis Fig. 6.6", "Craftax Classic", "capacity scaling", "control; 2H;16H;FFN512;FFN2048", "head and FFN allocation", "3/condition", "42,54,99", "24,576,000", "64", "native",
        "Train/Mean_Return at common endpoint", "13.664;13.543;14.031;13.893;13.125", "see generated table", "System/SPS", "varied",
        "Does larger backbone allocation uniformly improve return?", "absent", "main-text design-ablation panel", "wandb_exports/Craftax_Classic_Capacity_Scaling", "within-protocol; no perception-bottleneck claim")
    add("Thesis Sec. 6.6 / Appendix Craftax BPTT", "Craftax Classic", "BPTT horizon scaling",
        "SdLG-Approx T32/T64/T128", "rollout/TBPTT length", "reported multi-seed", "original report", "50M", "32,64,128", "native",
        "Train/Mean_Return trajectory", "reported ranges: 16--17;14.2--15.1;~12.6", "see original figure", "training SPS", "same model family",
        "How does rollout length alter dense-control learning?", "absent", "appendix figure and main-text summary", "plots/appendix/craftax_classic_bptt_return.png", "fixed total batch 32,768; do not infer mediation")
    add("ICML Fig. 1 / Appendix GGS", "Craftax Classic", "GGS ablation", "alpha=.1;alpha=1", "actor-gradient scaling", "5/condition", "42,54,99,123,456", "20,152,320", "64", "native",
        "Eval_Return and actor/critic probe cosine", "11.210;9.757", "0.557;0.299", "n/a", "matched", "Does GGS alter training and probe telemetry?", "appendix Tier B", "appendix/supporting", "wandb_exports/Craftax_Classic_BPTT/SdLGHybrid-GGS*.csv", "probe gradients are not PPO loss gradients")
    add("Original high-budget Craftax appendix", "Craftax", "300M/1B exploratory studies", "SdLG and selected baselines", "training budget", "reported runs", "see original", "300M/1B", "varied", "native",
        "return/achievements/length", "see original plots", "see original plots", "SPS", "varied", "Do rankings persist at high budget?", "absent", "supplement only", "wandb_exports/Craftax_1B_300M", "valid exploratory; redundant to design-map core")

    add("ICML MiniGrid figure / current Fig. 6", "MiniGrid", "MemoryS16", "W=16;W=0", "local window", "5/condition", "42,54,99,123,456", "10M", "64", "native", "endpoint return",
        "0.953;0.674", "0.086;0.391", "SPS reported historically", "1,118,565 matched", "Does local processing change transient-cue performance?", "canonical Tier B", "main text", "wandb_exports/MiniGrid_SWA_Ablation/*MemoryS16*", "unpaired difference CI includes zero")
    add("Original MiniGrid family", "MiniGrid", "MemoryS8", "W=16;W=0", "local window", "3/condition", "42,54,99", "9,830,400", "64", "native", "endpoint return",
        "0.294;0.987", "0.254;0.000", "n/a", "matched within task", "Does local-window sensitivity change with task horizon?", "absent", "main-text/appendix task-specific contrast", "wandb_exports/MiniGrid_SWA_Ablation/*MemoryS8*", "observed direction differs from MemoryS16")
    add("Original MiniGrid family", "MiniGrid", "DoorKey 8x8/16x16", "SWA; Conv1D", "local operator", "reported experiment", "see original", "reported", "reported", "native", "return/length/entropy",
        "see original", "see original", "n/a", "matched where reported", "Does the useful local operator depend on navigation structure?", "absent", "supplement; explicit placement note", "wandb_exports/MiniGrid_SWA_Ablation/*DoorKey*", "no aggregate reconstructed; do not invent values")

    add("ICML Table E.1", "Minesweeper", "structural ablation", "1GLA-r16;2GLA-r4;3GLA-r4", "recurrent depth/rank allocation", "3/variant", "reported N=3", "15M", "16", "100 and 200",
        "normalized evaluation score", "0.445;0.549;0.527", "0.031;0.024;0.028", "SPS 1698.8;1617.9;1488.7", "varied allocation",
        "How does recurrent depth/rank allocation trade off score and throughput?", "absent", "main-text design-ablation panel", "results/minesweeper_ablation_results.csv; original Table E.1", "same H100/H200 values; not length extrapolation evidence")

    add("ICML Tables E.2/E.3", "POPGym", "16 Easy + 7 Medium tasks", "SdLG Exact/Approx; GRU; Mamba; GLA", "task-operation heterogeneity", "3/model/task Easy; mixed Medium", "reported in original tables", "~15M", "16", "native H=100",
        "MMER evaluation return", "see complete matrix", "see complete matrix", "SPS available", "capacity unmatched",
        "Do model preferences vary across standardized memory operations?", "absent", "main-text heatmap + appendix numeric table", "results/popgym_results.csv", "no pooled score or win count; published baselines separate")

    add("Current Fig. 8", "Synthetic retrieval", "canonical factorial", "N x dk x dv", "addressing dimension/load", "10/cell", "0--9", "fixed synthetic", "n/a", "N=32,64,128",
        "mean retrieval cosine", "see 480-row artifact", "ten-seed SD", "n/a", "n/a", "How do addressing dimension and load affect retrieval?", "canonical Tier A", "main text", "papers/writing/generated_artifacts/synthetic_capacity_480_cells.csv", "source of record; supersedes historical grid")
    add("ICML Appendix F.7", "Synthetic retrieval", "overwrite/collateral interference", "Delta; Additive; eta variants", "update primitive/overwrite", "reported multi-seed", "see original", "fixed synthetic", "n/a", "controlled",
        "target retrieval and collateral cosine", "Delta .878 vs Additive .366; collateral .210 vs .331", ".032/.095/.023/.017", "n/a", "n/a", "How do update rules trade overwrite fidelity and collateral interference?", "absent", "appendix supporting diagnostic", "results/synthetic_overwrite_collateral.csv; results/overwrite_under_load.csv", "do not displace canonical factorial")
    add("Historical synthetic appendix", "Synthetic retrieval", "forgetting/head/update sweeps", "additive/delta/gated; head allocations", "forgetting/update/head allocation", "reported experiments", "see artifacts", "fixed synthetic", "n/a", "controlled",
        "cosine/MSE/forgetting metrics", "see artifacts", "see artifacts", "n/a", "n/a", "Which update and head controls affect retrieval/forgetting?", "absent", "supplement with index", "results/selective_forgetting_study.csv; results/multihead_subspaces_study.csv; results/fixed_head_capacity_study.csv", "valid distinct operations; not headline")

    add("ICML Fig. 5 / Thesis Fig. 5.1", "Systems microbenchmark", "sequence scaling", "SdLG Exact; SdLG Approx; GTrXL", "sequence length", "reported measurements", "n/a", "microbenchmark", "128--4096", "n/a",
        "latency and peak VRAM", "see systems CSV", "not reported", "ms / GB", "architecture dependent", "How do latency and VRAM scale with sequence length?", "absent", "appendix systems figure", "scripts/diagnostics/benchmarks/latency/profile_throughput.csv", "separate from Craftax training-loop SPS")
    add("Memory-scaling protocols A/B", "Systems microbenchmark", "carry footprint", "SdLG rank; SdLG vs GTrXL", "rank and horizon", "deterministic size accounting", "n/a", "microbenchmark", "64--1024", "n/a",
        "carry/peak VRAM MB", "see systems CSV", "n/a", "memory footprint", "architecture dependent", "How do recurrent-state and context footprints scale?", "absent", "appendix systems table/figure", "scripts/diagnostics/architecture/memory_scaling/*.csv", "keep accounting and measured peak distinct")
    add("Resource accounting study", "Systems", "resource accounting", "SdLG variants", "state/update choice", "reported configurations", "n/a", "microbenchmark", "varied", "n/a",
        "state bytes and SPS", "see CSV", "n/a", "SPS/state size", "varied", "How do state representations trade memory and speed?", "absent", "supplement", "results/resource_accounting_summary.csv", "supporting resource inventory")
    add("Thesis XLA micro-validation", "Systems", "compilation stability", "dynamic vs static masking", "shape/static execution", "10 seeds", "reported", "100 forward passes", "n/a", "n/a",
        "recompilations/dispatch latency", "14 vs 0; 1338.19 vs 98.04 ms", "see thesis", "dispatch latency", "matched", "Does static masking avoid recompilation overhead?", "absent", "appendix/supplement", "scripts/diagnostics/safety_and_compilation/xla_torture_test", "implementation evidence, not architecture quality")
    add("Original numerical-safety appendix", "Numerical diagnostics", "FP32/BF16 recurrence", "precision variants", "numerical precision", "reported trials", "see original", "diagnostic", "n/a", "long carry",
        "drift/error", "see numerical-safety artifact", "see artifact", "n/a", "n/a", "How does carry precision affect numerical drift?", "appendix note only", "supplement/Tier C", "scripts/diagnostics/safety_and_compilation/numerical_safety", "do not promote as main mechanism")
    return cols, rows


def write_ledger():
    cols, rows = experiment_rows()
    out = WRITING / "original_paper_experiment_ledger.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); w.writerows(rows)
    return rows


def craftax_endpoint_table():
    label_map = {
        "Control_Exact_Dense": "Exact control",
        "Ablation_Approx_Dense": "Approximate",
        "Ablation_GammaZero": r"$\Gamma=0$ ablation",
        "Ablation_GRUBias_0": r"residual bias $b=0$",
        "Capacity_2Heads": "2 heads",
        "Capacity_16Heads": "16 heads",
        "Capacity_NarrowFFN_512": "FFN 512",
        "Capacity_WideFFN_2048": "FFN 2048",
    }
    records = []
    dirs = [ROOT / "wandb_exports/Craftax_Classic_Exact_vs_Approx", ROOT / "wandb_exports/Craftax_Classic_Capacity_Scaling"]
    seen = set()
    for directory in dirs:
        for path in sorted(directory.glob("*.csv")):
            stem = re.sub(r"-seed-\d+$", "", path.stem)
            if stem not in label_map: continue
            seed = int(re.search(r"seed-(\d+)", path.stem).group(1))
            key = (stem, seed)
            if key in seen: continue
            seen.add(key)
            d = pd.read_csv(path).sort_values("Environment_Steps")
            row = d.iloc[-1]
            records.append({"condition": label_map[stem], "source_label": stem, "seed": seed,
                            "environment_steps": int(row["Environment_Steps"]),
                            "train_mean_return": float(row["Train/Mean_Return"]),
                            "system_sps": float(row["System/SPS"])})
    raw = pd.DataFrame(records)
    raw.to_csv(GEN / "craftax_design_ablation_endpoints.csv", index=False)
    summary = raw.groupby(["source_label","condition"], as_index=False).agg(
        n=("seed","count"), mean=("train_mean_return","mean"), sd=("train_mean_return","std"), mean_sps=("system_sps","mean"))
    summary.to_csv(GEN / "craftax_design_ablation_summary.csv", index=False)
    return raw, summary


def minigrid_family_table():
    records=[]
    for path in glob.glob(str(ROOT / "wandb_exports/MiniGrid_SWA_Ablation/*MemoryS*.csv")):
        name=Path(path).stem
        m=re.search(r"MemoryS(8|16)-Seed-(\d+)", name)
        if not m: continue
        horizon=int(m.group(1)); seed=int(m.group(2))
        window="W=16" if "SdLG-SWA" in name else "W=0"
        d=pd.read_csv(path)
        eval_cols=[c for c in d.columns if "Eval" in c and "Return" in c]
        if not eval_cols: continue
        step_col="Environment_Steps" if "Environment_Steps" in d.columns else ("_step" if "_step" in d.columns else d.columns[0])
        d=d.sort_values(step_col)
        val=float(d.iloc[-1][eval_cols[0]])
        records.append({"task":f"MemoryS{horizon}","window":window,"seed":seed,"endpoint_return":val,"source":str(Path(path).relative_to(ROOT))})
    raw=pd.DataFrame(records)
    raw.to_csv(GEN/"minigrid_local_processing_family.csv",index=False)
    raw.groupby(["task","window"],as_index=False).agg(n=("seed","count"),mean=("endpoint_return","mean"),sd=("endpoint_return","std")).to_csv(GEN/"minigrid_local_processing_summary.csv",index=False)
    return raw


TASK_FAMILY = {
    "Autoencode":"reconstruction", "Battleship":"associative/search", "Concentration":"associative/search",
    "MineSweeper":"associative/search", "Minesweeper":"associative/search", "CountRecall":"accumulation",
    "HigherLower":"comparison", "LabyrinthEscape":"navigation", "LabyrinthExplore":"navigation",
    "MultiarmedBandit":"bandit", "NoisyPositionOnlyCartPole":"partial-state control",
    "NoisyPositionOnlyPendulum":"partial-state control", "PositionOnlyCartPole":"partial-state control",
    "PositionOnlyPendulum":"partial-state control", "VelocityOnlyCartpole":"partial-state control",
    "RepeatFirst":"sequence recall", "RepeatPrevious":"sequence recall",
}


def task_base(name):
    return re.sub(r"(Easy|Medium)$", "", name)


def popgym_table():
    d=pd.read_csv(ROOT/"results/popgym_results.csv")
    d=d[d.Model.isin(MODELS)].copy()
    d["Environment"]=d["Environment"].str.replace("MineSweeper","Minesweeper",regex=False)
    # Original MMER convention: maximum recorded H100 evaluation per seed.
    per_seed=d.groupby(["Environment","Model","Seed"],as_index=False)["Eval_Return_H100"].max()
    agg=per_seed.groupby(["Environment","Model"],as_index=False).agg(mean=("Eval_Return_H100","mean"),sd=("Eval_Return_H100","std"),n=("Seed","nunique"))
    agg["difficulty"]=agg.Environment.str.extract(r"(Easy|Medium)$",expand=False)
    agg["task_family"]=agg.Environment.map(lambda x:TASK_FAMILY.get(task_base(x),"other"))
    # The original paper is authoritative for the rows it tabulates.  Override
    # rediscovered aggregates with the reported means/SDs rather than allowing
    # later or expanded filesystem exports to change the historical result.
    tex=(ROOT/"papers/latex/icmlV2/sections/appendix.tex").read_text(encoding="utf-8")
    reported=[]
    for line in tex.splitlines():
        if not line.lstrip().startswith(r"\textbf{"): continue
        if "Easy}" not in line and "Medium}" not in line: continue
        parts=[p.strip() for p in line.split("&")]
        if len(parts)<6: continue
        task=re.sub(r"\\textbf\{|\}|\$|\\dagger|\^", "", parts[0]).strip()
        task=task.replace("{","").replace("}","")
        if not (task.endswith("Easy") or task.endswith("Medium")): continue
        for model,cell in zip(MODELS,parts[1:6]):
            nums=re.findall(r"-?\d+\.\d+",cell)
            if not nums: continue
            reported.append((task,model,float(nums[0]),float(nums[1]) if len(nums)>1 else np.nan,1 if "dagger" in cell else 3))
    for task,model,mean,sd,n in reported:
        mask=(agg.Environment==task)&(agg.Model==model)
        if mask.any():
            agg.loc[mask,["mean","sd","n"]]=[mean,sd,n]
        else:
            agg=pd.concat([agg,pd.DataFrame([{"Environment":task,"Model":model,"mean":mean,"sd":sd,"n":n,
                "difficulty":"Medium" if task.endswith("Medium") else "Easy",
                "task_family":TASK_FAMILY.get(task_base(task),"other")}])],ignore_index=True)
    agg.to_csv(GEN/"popgym_complete_mmer_summary.csv",index=False)
    per_seed.to_csv(GEN/"popgym_complete_mmer_per_seed.csv",index=False)
    return agg


def plot_craftax_and_minesweeper(summary):
    plt.rcParams.update({"font.size":8,"axes.titlesize":9,"axes.labelsize":8})
    fig, axes=plt.subplots(1,3,figsize=(10.6,3.35),constrained_layout=True)
    order1=["Exact control","Approximate",r"$\Gamma=0$ ablation",r"residual bias $b=0$"]
    order2=["Exact control","2 heads","16 heads","FFN 512","FFN 2048"]
    colors=["#315a7d","#56a0a9","#d99058","#bd5b5b","#8b75b4"]
    for ax,order,title in [(axes[0],order1,"(a) State execution and priors"),(axes[1],order2,"(b) Capacity allocation")]:
        s=summary.set_index("condition").loc[order].reset_index()
        x=np.arange(len(s))
        ax.bar(x,s["mean"],yerr=s["sd"],capsize=3,color=colors[:len(s)],edgecolor="white")
        ax.set_xticks(x,order,rotation=28,ha="right")
        ax.set_ylabel("Endpoint training return")
        ax.set_title(title,loc="left",fontweight="bold")
        ax.grid(axis="y",alpha=.2)
        ax.set_ylim(9.5,15.2)
    names=["1 GLA\nr=16","2 GLA\nr=4","3 GLA\nr=4"]
    means=np.array([.445,.549,.527]); sds=np.array([.031,.024,.028]); sps=np.array([1698.8,1617.9,1488.7])
    ax=axes[2]; x=np.arange(3)
    ax.bar(x,means,yerr=sds,capsize=3,color=colors[:3],edgecolor="white")
    ax.set_xticks(x,names); ax.set_ylabel("Normalized score"); ax.set_ylim(.35,.60); ax.grid(axis="y",alpha=.2)
    ax.set_title("(c) Minesweeper depth/rank",loc="left",fontweight="bold")
    ax2=ax.twinx(); ax2.plot(x,sps,"o--",color="#9b4c32",lw=1.5); ax2.set_ylabel("SPS",color="#9b4c32"); ax2.tick_params(axis='y',colors="#9b4c32"); ax2.set_ylim(1400,1760)
    fig.suptitle("Controlled SdLG design axes beyond T-Maze",fontweight="bold",fontsize=11)
    for ext in ["pdf","png"]: fig.savefig(PLOTS/f"fig7_sdlg_design_ablations.{ext}",dpi=240,bbox_inches="tight")
    plt.close(fig)


def plot_popgym(agg):
    families=["reconstruction","associative/search","accumulation","comparison","navigation","bandit","partial-state control","sequence recall"]
    tasks=[]
    for fam in families:
        vals=sorted(agg.loc[agg.task_family.eq(fam),"Environment"].unique(),key=lambda x:("Medium" in x,x))
        tasks.extend(vals)
    matrix=np.full((len(tasks),len(MODELS)),np.nan); ns=np.full_like(matrix,np.nan); sds=np.full_like(matrix,np.nan)
    for i,t in enumerate(tasks):
        for j,m in enumerate(MODELS):
            s=agg[(agg.Environment==t)&(agg.Model==m)]
            if len(s): matrix[i,j]=s.iloc[0]["mean"]; sds[i,j]=s.iloc[0]["sd"]; ns[i,j]=s.iloc[0]["n"]
    row_min=np.nanmin(matrix,axis=1,keepdims=True); row_max=np.nanmax(matrix,axis=1,keepdims=True)
    scaled=(matrix-row_min)/np.where(row_max-row_min<1e-12,1,row_max-row_min)
    fig,ax=plt.subplots(figsize=(10.8,8.7))
    im=ax.imshow(scaled,aspect="auto",cmap="viridis",vmin=0,vmax=1)
    short=["Exact","Approx","GRU","Mamba","GLA"]
    ax.set_xticks(range(len(short)),short,fontsize=9)
    labels=[]; previous=None
    for t in tasks:
        fam=TASK_FAMILY.get(task_base(t),"other")
        task_label=t.replace("Easy"," (E)").replace("Medium"," (M)")
        labels.append((fam+"  ·  " if fam!=previous else "")+task_label)
        previous=fam
    ax.set_yticks(range(len(tasks)),labels,fontsize=7.2)
    for i in range(len(tasks)):
        for j in range(len(MODELS)):
            if np.isfinite(matrix[i,j]):
                txt=f"{matrix[i,j]:.2f}"
                if ns[i,j]==1: txt+="†"
                color="white" if scaled[i,j]<.35 or scaled[i,j]>.78 else "black"
                ax.text(j,i,txt,ha="center",va="center",fontsize=6.6,color=color)
    boundaries=[]; start=0
    for fam in families:
        count=sum(TASK_FAMILY.get(task_base(t),"other")==fam for t in tasks)
        if count:
            start+=count; boundaries.append(start-.5)
    for y in boundaries[:-1]: ax.axhline(y,color="white",lw=2)
    ax.set_title("POPGym: task-conditioned model preferences across memory operations",fontweight="bold",pad=10)
    ax.set_xlabel("Evaluated model (raw task-native MMER shown; color is row-normalized)")
    cb=fig.colorbar(im,ax=ax,fraction=.025,pad=.02); cb.set_label("Within-task visualization scale")
    fig.subplots_adjust(left=.27,right=.91,top=.94,bottom=.08)
    for ext in ["pdf","png"]: fig.savefig(PLOTS/f"fig8_popgym_task_matrix.{ext}",dpi=260,bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(matrix,index=tasks,columns=MODELS).to_csv(GEN/"popgym_complete_mmer_matrix.csv")


def plot_systems():
    d=pd.read_csv(ROOT/"scripts/diagnostics/benchmarks/latency/profile_throughput.csv")
    for c in ["exact_sdlg_lat","exact_sdlg_vram","approx_sdlg_lat","approx_sdlg_vram","gtrxl_lat","gtrxl_vram"]:
        d.loc[d[c]<=0,c]=np.nan
    fig,axes=plt.subplots(1,2,figsize=(8.6,3.5),constrained_layout=True)
    specs=[("exact_sdlg","SdLG-Exact","#315a7d"),("approx_sdlg","SdLG-Approx","#56a0a9"),("gtrxl","GTrXL","#d99058")]
    for prefix,label,color in specs:
        axes[0].plot(d.sequence_length_T,d[f"{prefix}_lat"],"o-",label=label,color=color)
        axes[1].plot(d.sequence_length_T,d[f"{prefix}_vram"],"o-",label=label,color=color)
    for ax,title,ylabel in [(axes[0],"(a) Step latency","Latency (ms)"),(axes[1],"(b) Peak VRAM","VRAM (GB)")]:
        ax.set_xscale("log",base=2); ax.set_xlabel("Sequence length T"); ax.set_ylabel(ylabel); ax.set_title(title,loc="left",fontweight="bold"); ax.grid(alpha=.25); ax.legend(frameon=False,fontsize=7)
    fig.suptitle("Sequence-length resource scaling (separate microbenchmark)",fontweight="bold")
    for ext in ["pdf","png"]: fig.savefig(PLOTS/f"fig9_systems_scaling.{ext}",dpi=240,bbox_inches="tight")
    plt.close(fig)


def plot_tmaze_context_and_craftax():
    # Figure 2 deliberately separates the external fixed-endpoint landscape
    # from the two base SdLG references drawn from the ten-seed factorial.
    ext=pd.read_csv(WRITING/"generated_artifacts/tmaze_external_landscape.csv")
    fac=pd.read_csv(WRITING/"generated_artifacts/tmaze_rank_rope_zero_shot.csv")
    fig,axes=plt.subplots(1,2,figsize=(9.4,3.5),gridspec_kw={"width_ratios":[2.2,1]},constrained_layout=True)
    order=["Mamba","LTI-Linear","GRU","LSTM","Markovian"]
    colors=["#56a0a9","#8b75b4","#315a7d","#d99058","#888888"]
    rng=np.random.default_rng(12)
    for i,(name,color) in enumerate(zip(order,colors)):
        vals=ext.loc[ext.family.eq(name),"H100"].to_numpy()
        axes[0].scatter(i+rng.uniform(-.10,.10,len(vals)),vals,s=24,color=color,alpha=.8,edgecolor="white",lw=.35)
        axes[0].errorbar(i,np.mean(vals),yerr=np.std(vals,ddof=1),fmt="_",ms=16,mew=2,color="black",capsize=3)
    axes[0].set_xticks(range(len(order)),order,rotation=20,ha="right")
    axes[0].set_title("(a) External fixed-endpoint landscape",loc="left",fontweight="bold")
    axes[0].set_ylabel("Evaluation return at H=100"); axes[0].set_ylim(.35,1.07); axes[0].grid(axis="y",alpha=.2)
    refs=[("Exact_NoRoPE","SdLG-Exact\n(NoRoPE)","#315a7d"),("Approx_RoPE","SdLG-Approx\n(RoPE)","#56a0a9")]
    for i,(key,label,color) in enumerate(refs):
        vals=fac.loc[fac.model.eq(key),"val_H100"].to_numpy()
        axes[1].scatter(i+rng.uniform(-.10,.10,len(vals)),vals,s=24,color=color,alpha=.8,edgecolor="white",lw=.35)
        axes[1].errorbar(i,np.mean(vals),yerr=np.std(vals,ddof=1),fmt="_",ms=16,mew=2,color="black",capsize=3)
    axes[1].set_xticks(range(2),[x[1] for x in refs]); axes[1].set_ylim(.35,1.07); axes[1].grid(axis="y",alpha=.2)
    axes[1].set_title("(b) Base SdLG references",loc="left",fontweight="bold")
    fig.suptitle("Passive T-Maze at the trained horizon (protocols shown separately)",fontweight="bold")
    for extn in ["pdf","png"]: fig.savefig(PLOTS/f"fig2_tmaze_external_landscape.{extn}",dpi=240,bbox_inches="tight")
    plt.close(fig)

    perf=pd.read_csv(WRITING/"generated_artifacts/table1_craftax_performance.csv")
    perf=pd.concat([perf,pd.DataFrame([
        {"family":"SdLG-Approx","mean":14.79,"sd":.36,"ci_low":np.nan,"ci_high":np.nan,"parameters":6110000},
        {"family":"SdLG-Exact","mean":13.70,"sd":.52,"ci_low":np.nan,"ci_high":np.nan,"parameters":8780000},
    ])],ignore_index=True)
    perf.to_csv(GEN/"craftax_t64_seven_family_performance.csv",index=False)
    sps=pd.read_csv(WRITING/"generated_artifacts/table2_craftax_throughput.csv")
    sps=pd.concat([sps,pd.DataFrame([
        {"family":"SdLG-Approx","mean_sps":13020,"sd_sps":np.nan,"ci_low":np.nan,"ci_high":np.nan},
        {"family":"SdLG-Exact","mean_sps":9400,"sd_sps":np.nan,"ci_low":np.nan,"ci_high":np.nan},
    ])],ignore_index=True)
    sps.to_csv(GEN/"craftax_t64_seven_family_throughput.csv",index=False)
    order=["PPO-GRU","PPO-Mamba","SdLG-Approx","AGaLiTe","PPO-GLA","PPO-GTrXL","SdLG-Exact"]
    s=sps.set_index("family").loc[order].reset_index(); x=np.arange(len(s))
    fig,ax=plt.subplots(figsize=(9.2,4.0),constrained_layout=True)
    barcolors=["#315a7d","#56a0a9","#4b8b6f","#d99058","#8b75b4","#bd5b5b","#2b6454"]
    ax.bar(x,s.mean_sps,color=barcolors,edgecolor="white")
    known=s.sd_sps.notna().to_numpy(); ax.errorbar(x[known],s.loc[known,"mean_sps"],yerr=s.loc[known,"sd_sps"],fmt="none",ecolor="black",capsize=3,lw=1)
    ax.set_xticks(x,order,rotation=24,ha="right"); ax.set_ylabel("Environment steps/s"); ax.grid(axis="y",alpha=.2)
    ax.set_title("Craftax T=64 cumulative end-to-end training-loop throughput",fontweight="bold")
    ax.text(.99,.97,"Error bars: seed SD where available;\nSdLG bars are original-paper reported aggregates.",transform=ax.transAxes,ha="right",va="top",fontsize=7)
    for extn in ["pdf","png"]: fig.savefig(PLOTS/f"fig4_craftax_throughput.{extn}",dpi=240,bbox_inches="tight")
    plt.close(fig)


def write_text_artifacts(rows):
    (WRITING/"original_to_current_evidence_map.md").write_text("""# Original-to-current evidence map

The original paper is the authoritative experiment inventory. The current canonical T-Maze and 480-row synthetic artifacts supersede older versions of the same studies; richer-task studies retain their original internally matched protocols.

| Original evidence | Scientific role | Current disposition |
|---|---|---|
| T-Maze external and state-execution × RoPE studies | Controlled core / external landscape | Main text, Figures 2–3 |
| Craftax 50M T=64 benchmark, including SdLG | External architecture context | Main text, Table 1 and Figure 4 |
| Craftax Exact/Approx, temporal prior, residual bias, capacity | Controlled dense-control design axes | Main text, Figure 5; numeric appendix table |
| Craftax BPTT T=32/64/128 | Rollout/truncation sensitivity | Main-text summary; original trajectory in appendix |
| Minesweeper 1/2/3-GLA variants | Structural depth/rank allocation | Main text, Figure 5; appendix table |
| MiniGrid MemoryS16 and MemoryS8 | Task-conditioned local processing | Main text (MemoryS16) and appendix family table |
| DoorKey and Conv1D local-operator studies | Local-operator robustness | Supplement, with no reconstructed aggregate invented |
| POPGym 16 Easy + 7 Medium tasks | Cross-task external validity | Main text, Figure 7; full numeric appendix/supplement |
| Canonical N×dk×dv factorial | Controlled associative retrieval | Main text, Figure 8; supersedes historical fixed-grid study |
| Overwrite, forgetting, head-allocation synthetic studies | Supporting diagnostics | Appendix/supplement |
| GGS and boundary probes | Supporting diagnostics | Appendix / short supporting result |
| Latency, VRAM, carry size, resource accounting | Systems evidence | Appendix Figure 10 and supplement |
| High-budget Craftax and XLA/numerical diagnostics | Exploratory/implementation evidence | Supplement; explicit placement in ledger |
""",encoding="utf-8")
    (WRITING/"proposed_main_text_promotions.md").write_text("""# Main-text promotions

1. **Implemented:** SdLG-Approx and SdLG-Exact appear in the same-protocol Craftax $T=64$ table and SPS figure.
2. **Implemented:** a compact controlled-design subsection reports Craftax state/prior/capacity interventions and the Minesweeper structural trade-off.
3. **Implemented:** the complete 23-task POPGym task-by-model matrix is grouped by operation without a pooled ranking.
4. **Implemented:** Discussion contains a central operating-regime/design-axis table.
5. **Implemented:** MemoryS16 remains in the main text and MemoryS8 supplies the task-specific counterexample.

The BPTT study is summarized in text and shown in the appendix because its metric is a historical training-return trajectory rather than the canonical Craftax fixed-endpoint evaluation return.
""",encoding="utf-8")
    (WRITING/"proposed_appendix_promotions.md").write_text("""# Appendix and supplement promotions

- Move the experiment-matrix table from Protocol to the appendix and extend it with original-paper studies.
- Add complete numeric tables for Craftax design ablations, Minesweeper, MiniGrid MemoryS8/MemoryS16, and POPGym.
- Add the original Craftax BPTT trajectory with protocol qualification.
- Add sequence-length latency/VRAM scaling as a distinct systems microbenchmark.
- Index overwrite/collateral-interference, forgetting, head-allocation, high-budget Craftax, XLA, numerical-safety, DoorKey, and Conv1D artifacts in the supplement.
- Retain the historical fixed-grid synthetic study only for auditability because the 480-row canonical factorial supersedes it.
""",encoding="utf-8")
    axes=pd.DataFrame([
        ["Long-horizon behavior","State execution","Passive T-Maze","Exact conditions stronger at trained and unseen horizons","Tier A; execution changes recurrence/read as well as factorization"],
        ["Positional extrapolation","RoPE / NoRoPE","Passive T-Maze","Phase changes training/extrapolation relationship","Tier A within factorial; condition-dependent"],
        ["Associative addressing","N, dk, dv","Synthetic retrieval","Strong monotonic dk effect; weak dv; degradation with N/dk","Tier A controlled; no sharp-threshold claim"],
        ["Local/transient processing","Local window / local operator","MiniGrid family","Observed local-window effect changes across MemoryS8 and S16","Tier B/task-specific; small n"],
        ["Depth/rank allocation","1/2/3 GLA variants","Minesweeper","Two lower-rank recurrent layers outperform one high-rank layer; third layer slower without added score","Valid 3-seed structural ablation; factors vary jointly"],
        ["Dense-control state execution","Exact / Approximate","Craftax historical protocol","Similar endpoint training return; Approximate higher reported SPS","Separate 25M protocol"],
        ["Rollout sensitivity","T=32/64/128","Craftax","Reported training return decreases as T increases","Historical matched study; mechanism not identified"],
        ["Capacity allocation","heads / FFN width","Craftax","Larger allocation does not uniformly improve endpoint return","Three-seed internal protocol"],
        ["Optimization coupling","GGS alpha","Craftax","Different returns and probe trajectories","Tier B; no PPO-gradient conflict claim"],
        ["Task-operation heterogeneity","model family","POPGym 23 tasks","Relative model behavior varies substantially by task","Task-native scales; capacity unmatched; no pooled leaderboard"],
        ["Systems scaling","latency / VRAM / SPS / carry","Craftax and microbenchmarks","Behavioral and resource orderings differ","Timing definitions remain separate"],
    ],columns=["operational_demand","controlled_design_axis","environment_evidence","observed_effect","confidence_limitation"])
    axes.to_csv(WRITING/"sdlg_design_axis_summary.csv",index=False)
    placements=["# Original-paper experiment placement report","",f"Ledger rows: {len(rows)}.",""]
    for r in rows:
        placements.append(f"- **{r['source_table_or_figure']} — {r['task']}**: {r['recommended_use']}. {r['notes']}")
    (WRITING/"original_paper_experiment_placement_report.md").write_text("\n".join(placements)+"\n",encoding="utf-8")


def write_appendix_tables(pop):
    pivot_mean=pop.pivot(index="Environment",columns="Model",values="mean")
    pivot_sd=pop.pivot(index="Environment",columns="Model",values="sd")
    pivot_n=pop.pivot(index="Environment",columns="Model",values="n")
    families=["reconstruction","associative/search","accumulation","comparison","navigation","bandit","partial-state control","sequence recall"]
    tasks=[]
    for fam in families:
        tasks.extend(sorted(pop.loc[pop.task_family.eq(fam),"Environment"].unique(),key=lambda x:("Medium" in x,x)))
    lines=[r"\begin{table}[h]",r"\centering",r"\scriptsize",r"\setlength{\tabcolsep}{3pt}",
           r"\begin{tabular}{lrrrrr}",r"\toprule",r"Task & Exact & Approx. & GRU & Mamba & GLA \\",r"\midrule"]
    previous=None
    for task in tasks:
        fam=TASK_FAMILY.get(task_base(task),"other")
        if previous is not None and fam!=previous: lines.append(r"\addlinespace")
        vals=[]
        for model in MODELS:
            try: mean=pivot_mean.loc[task,model]
            except KeyError: mean=np.nan
            if not np.isfinite(mean): vals.append("--"); continue
            sd=pivot_sd.loc[task,model]; n=pivot_n.loc[task,model]
            vals.append(f"{mean:.2f}"+(f"$\\pm${sd:.2f}" if np.isfinite(sd) and n>1 else r"$^\dagger$"))
        safe=task.replace("_",r"\_")
        lines.append(safe+" & "+" & ".join(vals)+r" \\")
        previous=fam
    lines += [r"\bottomrule",r"\end{tabular}",
              r"\caption{Complete task-native POPGym MMER summary for our evaluated models. Easy cells are three-seed means where reported; $\dagger$ marks single-seed Medium pilots. Missing cells were not reported for that model/task. Reward scales are task-specific, so no across-row average is computed.}",
              r"\label{tab:popgym_full}",r"\end{table}"]
    (ROOT/"papers/latex/tmlr_review/popgym_complete_table.tex").write_text("\n".join(lines)+"\n",encoding="utf-8")


def main():
    rows=write_ledger()
    raw,summary=craftax_endpoint_table()
    minigrid_family_table()
    pop=popgym_table()
    plot_craftax_and_minesweeper(summary)
    plot_popgym(pop)
    plot_systems()
    plot_tmaze_context_and_craftax()
    write_text_artifacts(rows)
    write_appendix_tables(pop)
    print(f"wrote ledger with {len(rows)} experiment families")
    print(summary[["condition","n","mean","sd"]].to_string(index=False))
    print(f"POPGym tasks: {pop.Environment.nunique()}, cells: {len(pop)}")


if __name__ == "__main__":
    main()
