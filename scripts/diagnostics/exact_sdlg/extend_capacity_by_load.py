"""Canonical ten-seed extension of the synthetic capacity factorial to N=32,64."""

from pathlib import Path
import importlib.util
import csv
import jax
import jax.numpy as jnp
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path(__file__).with_name("factorial_addressing_forgetting_suite.py")
OUT = ROOT / "papers" / "writing" / "generated_artifacts"
EXTENDED = OUT / "synthetic_capacity_load_extension_n32_n64_ten_seed.csv"
COMBINED = OUT / "synthetic_capacity_480_cells.csv"

spec = importlib.util.spec_from_file_location("capacity_suite", SOURCE)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

rows = []
for seed in range(10):
    rng = jax.random.PRNGKey(seed)
    for N in (32, 64):
        for d_k in (16, 32, 64, 128):
            for d_v in (8, 16, 32, 64):
                rng, k1, k2 = jax.random.split(rng, 3)
                keys = jax.random.normal(k1, (N, d_k))
                vals = jax.random.normal(k2, (N, d_v))
                keys = keys / (jnp.linalg.norm(keys, axis=-1, keepdims=True) + 1e-5)
                vals = vals / (jnp.linalg.norm(vals, axis=-1, keepdims=True) + 1e-5)
                cosine = float(mod.run_factorial_step(keys, vals, "delta", 0.25))
                rows.append({
                    "seed": seed,
                    "N": N,
                    "d_k": d_k,
                    "d_v": d_v,
                    "load_ratio": N / d_k,
                    "cosine": cosine,
                })

extension = pd.DataFrame(rows)
extension.to_csv(EXTENDED, index=False)

canonical_path = OUT / "synthetic_capacity_ten_seed_cells.csv"
canonical = pd.read_csv(canonical_path)
canonical["load_ratio"] = canonical["N"] / canonical["d_k"]
combined = pd.concat([extension, canonical], ignore_index=True)
combined = combined.sort_values(["N", "seed", "d_k", "d_v"]).reset_index(drop=True)
combined.to_csv(COMBINED, index=False)

print(f"extension_rows={len(extension)}")
print(f"combined_rows={len(combined)}")
print("d_k_marginals")
print(extension.groupby(["N", "d_k"]).cosine.mean().round(4).to_string())
print("d_v_marginals")
print(extension.groupby(["N", "d_v"]).cosine.mean().round(4).to_string())
print("load_ratio_marginals")
print(combined.groupby("load_ratio").cosine.agg(["mean", "std", "count"]).round(4).to_string())
