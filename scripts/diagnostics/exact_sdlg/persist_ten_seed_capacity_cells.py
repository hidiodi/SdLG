"""Recreate the frozen 480-cell associative-capacity diagnostic.

The frozen paper combines the original N=128 factorial with its predeclared
N=32/64 extension.  The extension and original factorial used different PRNG
streams, so their historical draw order is intentionally preserved here.
"""
from pathlib import Path
import importlib.util
import argparse
import pandas as pd
import jax
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path(__file__).with_name("factorial_addressing_forgetting_suite.py")

spec = importlib.util.spec_from_file_location("capacity_suite", SOURCE)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "associative-capacity" / "capacity_cells.csv")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in range(10):
        # The N=32/64 extension was one contiguous stream per seed.
        extension_rng = jax.random.PRNGKey(seed)
        for N in (32, 64):
            for d_k in (16, 32, 64, 128):
                for d_v in (8, 16, 32, 64):
                    extension_rng, k1, k2 = jax.random.split(extension_rng, 3)
                    keys = jax.random.normal(k1, (N, d_k))
                    vals = jax.random.normal(k2, (N, d_v))
                    keys = keys / (jnp.linalg.norm(keys, axis=-1, keepdims=True) + 1e-5)
                    vals = vals / (jnp.linalg.norm(vals, axis=-1, keepdims=True) + 1e-5)
                    cosine = float(mod.run_factorial_step(keys, vals, "delta", 0.25))
                    rows.append({"seed": seed, "N": N, "d_k": d_k, "d_v": d_v,
                                 "load_ratio": N / d_k, "cosine": cosine})

        # N=128 is the earlier standalone factorial, with its own stream.
        canonical_rng = jax.random.PRNGKey(seed)
        for d_k in (16, 32, 64, 128):
            for d_v in (8, 16, 32, 64):
                canonical_rng, k1, k2 = jax.random.split(canonical_rng, 3)
                keys = jax.random.normal(k1, (128, d_k))
                vals = jax.random.normal(k2, (128, d_v))
                keys = keys / (jnp.linalg.norm(keys, axis=-1, keepdims=True) + 1e-5)
                vals = vals / (jnp.linalg.norm(vals, axis=-1, keepdims=True) + 1e-5)
                cosine = float(mod.run_factorial_step(keys, vals, "delta", 0.25))
                rows.append({"seed": seed, "N": 128, "d_k": d_k, "d_v": d_v,
                             "load_ratio": 128 / d_k, "cosine": cosine})
    df = pd.DataFrame(rows).sort_values(["N", "seed", "d_k", "d_v"]).reset_index(drop=True)
    df.to_csv(args.output, index=False)
    print(f"wrote {len(df)} rows to {args.output}")
    print(df.groupby("d_k").cosine.mean().round(4).to_dict())
    print(df.groupby("d_v").cosine.mean().round(4).to_dict())


if __name__ == "__main__":
    main()
