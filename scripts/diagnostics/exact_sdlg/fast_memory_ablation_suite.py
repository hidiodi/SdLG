#!/usr/bin/env python3
"""
⚡ FAST UNIFIED MEMORY ABLATION DIAGNOSTIC SUITE (Families A - G)
========================================================================
Executes high-speed supervised gradient training on JAX across 3 canonical
associative memory tasks using 1 single seed (seed 42) in < 2 minutes:
  1. Multi-Query Associative Recall (MQAR, N=16 pairs)
  2. Selective Overwrite & Collateral Damage (N=16 pairs)
  3. Capacity Saturation Sweep (N ∈ {8, 16, 32, 64})

Evaluates 10 Core Hypotheses:
  - Family A: SdLG-Exact (Additive), SdLG-Delta, SdLG-ADAM
  - Family B: SdLG-ChannelDecay (Retention Isolation)
  - Family C: SdLG-Window32 (Window size 32)
  - Family D: SdLG-Approx-ADAM-r4, SdLG-Approx-ADAM-r16 (Low rank)
  - Family E: SdLG-ADAM-Matched (Parity matched d_ffc=448)
  - Family F: SdLG-Depth4-r4 (Depth scaling)
  - Family G: SdLG-LinearDenom (Denominator stability)
"""

import os
import sys
import time
import functools
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import pandas as pd
import optax
from typing import Dict, List, Tuple, Any

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.85"

current_file_path = os.path.abspath(__file__)
project_root = current_file_path
while not os.path.exists(os.path.join(project_root, "src")):
    project_root = os.path.dirname(project_root)
    if project_root == "/":
        break
if project_root not in sys.path:
    sys.path.append(project_root)

from src.models.common.sdlg_adam import UnbatchedSdLGADAM
from src.models.common.exact_sdlg import UnbatchedExactSdLG as UnbatchedSdLG
from src.models.common.sdlg_approx_adam import UnbatchedSdLGApproxADAM

OUTPUT_DIR = os.path.join(project_root, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =====================================================================
# Vectorized Pure JAX Data Generators
# =====================================================================

@functools.partial(jax.jit, static_argnums=(1, 2, 3, 4))
def generate_mqar_batch(rng, batch_size=32, seq_len=128, num_kv_pairs=16, vocab_size=64):
    key1, key2, key3 = jax.random.split(rng, 3)
    keys = jax.random.randint(key1, (batch_size, num_kv_pairs), minval=1, maxval=vocab_size // 2)
    vals = jax.random.randint(key2, (batch_size, num_kv_pairs), minval=vocab_size // 2, maxval=vocab_size - 1)
    
    kv_interleaved = jnp.stack([keys, vals], axis=-1).reshape(batch_size, 2 * num_kv_pairs)
    pad_len = seq_len - 2 * num_kv_pairs - num_kv_pairs
    padding = jnp.zeros((batch_size, pad_len), dtype=jnp.int32)
    
    perm_keys = jax.random.split(key3, batch_size)
    query_indices = jax.vmap(lambda k: jax.random.permutation(k, num_kv_pairs))(perm_keys)
    
    query_keys = jnp.take_along_axis(keys, query_indices, axis=-1)
    target_vals = jnp.take_along_axis(vals, query_indices, axis=-1)
    
    full_seq = jnp.concatenate([kv_interleaved, padding, query_keys], axis=1)
    return full_seq, target_vals


@functools.partial(jax.jit, static_argnums=(1, 2, 3, 4))
def generate_overwrite_batch(rng, batch_size=32, seq_len=128, num_kv_pairs=16, vocab_size=64):
    key1, key2 = jax.random.split(rng, 2)
    keys = jax.random.randint(key1, (batch_size, num_kv_pairs), minval=1, maxval=vocab_size // 2)
    vals_old = jax.random.randint(key2, (batch_size, num_kv_pairs), minval=vocab_size // 2, maxval=vocab_size - 1)
    vals_new = (vals_old + 5) % (vocab_size - vocab_size // 2) + vocab_size // 2
    
    kv_initial = jnp.stack([keys, vals_old], axis=-1).reshape(batch_size, 2 * num_kv_pairs)
    kv_overwrite = jnp.stack([keys[:, :1], vals_new[:, :1]], axis=-1).reshape(batch_size, 2)
    
    target_vals = jnp.concatenate([vals_new[:, :1], vals_old[:, 1:]], axis=-1)
    query_keys = keys
    
    pad_len = seq_len - (2 * num_kv_pairs + 2 + num_kv_pairs)
    padding = jnp.zeros((batch_size, pad_len), dtype=jnp.int32)
    
    full_seq = jnp.concatenate([kv_initial, kv_overwrite, padding, query_keys], axis=1)
    return full_seq, target_vals


# =====================================================================
# Configurable Sequence Model Wrapper for All 7 Research Families
# =====================================================================

class UnifiedAblationWrapper(nn.Module):
    vocab_size: int = 64
    d_model: int = 64
    n_heads: int = 4
    d_k: int = 16
    d_v: int = 16
    d_ffc: int = 256
    n_layers: int = 2
    r: int = 4
    window_size: int = 16
    use_softplus_denom: bool = True
    model_type: str = "sdlg_adam"

    @nn.compact
    def __call__(self, seq):
        B, T = seq.shape
        emb = nn.Embed(self.vocab_size, self.d_model)(seq)
        terminations = jnp.zeros((B, T), dtype=jnp.float32)

        if self.model_type in ["sdlg_exact", "sdlg_delta", "sdlg_adam", "sdlg_channeldecay", "sdlg_adam_matched", "sdlg_window32"]:
            variant_map = {
                "sdlg_exact": "additive",
                "sdlg_delta": "delta",
                "sdlg_adam": "adam",
                "sdlg_adam_matched": "adam",
                "sdlg_channeldecay": "channel_decay",
                "sdlg_window32": "adam",
            }
            d_ffc_cur = 224 if self.model_type == "sdlg_adam_matched" else self.d_ffc
            win_size_cur = 32 if self.model_type == "sdlg_window32" else self.window_size

            unbatched = UnbatchedSdLG.initialize_carry(n_layers=self.n_layers, n_heads=self.n_heads, d_k=self.d_k, d_v=self.d_v, window_size=win_size_cur)
            carry_init = jax.tree_util.tree_map(lambda x: jnp.repeat(jnp.expand_dims(x, 0), B, axis=0), unbatched)
            
            BatchedModel = nn.vmap(
                UnbatchedSdLGADAM, in_axes=(0, 0, 0, None), out_axes=(0, 0, 0),
                variable_axes={'params': None}, split_rngs={'params': False}
            )
            model_inst = BatchedModel(
                n_layers=self.n_layers, d_model=self.d_model, n_heads=self.n_heads, d_k=self.d_k, d_v=self.d_v,
                d_ffc=d_ffc_cur, window_size=win_size_cur, variant=variant_map[self.model_type]
            )
            outs, _, _ = model_inst(emb, terminations, carry_init, True)
            
        elif self.model_type in ["sdlg_approx_adam_r4", "sdlg_approx_adam_r16", "sdlg_lineardenom", "sdlg_depth4_r4"]:
            r_cur = 16 if self.model_type == "sdlg_approx_adam_r16" else self.r
            n_layers_cur = 4 if self.model_type == "sdlg_depth4_r4" else self.n_layers
            softplus_cur = False if self.model_type == "sdlg_lineardenom" else self.use_softplus_denom

            from src.models.common.sdlg_hybrid import BatchedSdLGHybrid
            unbatched = BatchedSdLGHybrid.initialize_carry(batch_size=1, n_layers=n_layers_cur, d_k=self.d_k, d_v=self.d_v, r=r_cur, window_size=16)
            carry_init = jax.tree_util.tree_map(lambda x: jnp.repeat(jnp.expand_dims(x, 0), B, axis=0), unbatched)

            BatchedHybrid = nn.vmap(
                BatchedSdLGHybrid, in_axes=(0, 0, 0, None), out_axes=((0, 0), 0),
                variable_axes={'params': None}, split_rngs={'params': False}
            )
            hybrid_inst = BatchedHybrid(
                n_layers=n_layers_cur, d_model=self.d_model, d_k=self.d_k, d_v=self.d_v, d_ffc=self.d_ffc,
                r=r_cur, window_size=16, return_split=False, use_softplus_denominator=softplus_cur
            )
            ((outs, _), _) = hybrid_inst(emb, terminations, carry_init, True)

        logits = nn.Dense(self.vocab_size)(outs)
        return logits


# =====================================================================
# Supervised Fast Training & Evaluation (100 steps per model)
# =====================================================================

def train_and_eval_model(model_type: str, task: str = "mqar", num_kv_pairs: int = 16, num_steps: int = 120) -> Dict[str, float]:
    rng = jax.random.PRNGKey(42)
    batch_size = 32
    seq_len = 128
    vocab_size = 64
    
    model = UnifiedAblationWrapper(vocab_size=vocab_size, d_model=64, n_heads=4, d_k=16, d_v=16, d_ffc=256, model_type=model_type)
    
    dummy_seq = jnp.zeros((batch_size, seq_len), dtype=np.int32)
    params = model.init(rng, dummy_seq)
    
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=1e-3, weight_decay=1e-2)
    )
    opt_state = tx.init(params)

    @jax.jit
    def train_step(params, opt_state, seq, targets):
        def loss_fn(p):
            logits = model.apply(p, seq)
            query_logits = logits[:, -num_kv_pairs:, :]
            loss = optax.softmax_cross_entropy_with_integer_labels(query_logits, targets)
            preds = jnp.argmax(query_logits, axis=-1)
            acc = jnp.mean(preds == targets)
            return jnp.mean(loss), acc

        (loss, acc), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, new_opt_state = tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss, acc

    start_time = time.time()
    for step in range(num_steps):
        rng, step_rng = jax.random.split(rng)
        if task == "mqar":
            seq, targets = generate_mqar_batch(step_rng, batch_size, seq_len, num_kv_pairs, vocab_size)
        else:
            seq, targets = generate_overwrite_batch(step_rng, batch_size, seq_len, num_kv_pairs, vocab_size)
            
        params, opt_state, loss, acc = train_step(params, opt_state, seq, targets)

    elapsed = time.time() - start_time
    
    # Evaluate over 5 test batches
    eval_accs = []
    eval_losses = []
    for eval_step in range(5):
        rng, eval_rng = jax.random.split(rng)
        if task == "mqar":
            seq, targets = generate_mqar_batch(eval_rng, batch_size, seq_len, num_kv_pairs, vocab_size)
        else:
            seq, targets = generate_overwrite_batch(eval_rng, batch_size, seq_len, num_kv_pairs, vocab_size)
            
        logits = model.apply(params, seq)
        query_logits = logits[:, -num_kv_pairs:, :]
        loss_val = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(query_logits, targets))
        preds = jnp.argmax(query_logits, axis=-1)
        acc_val = float(jnp.mean(preds == targets))
        
        eval_losses.append(float(loss_val))
        eval_accs.append(acc_val)

    return {
        "model": model_type,
        "task": task,
        "N": num_kv_pairs,
        "eval_acc": float(np.mean(eval_accs)),
        "eval_loss": float(np.mean(eval_losses)),
        "time_sec": float(elapsed),
    }


# =====================================================================
# Main Suite Execution
# =====================================================================

def main():
    print("=" * 85, flush=True)
    print("⚡ FAST UNIFIED MEMORY ABLATION DIAGNOSTIC SUITE (Families A - G)", flush=True)
    print("=========================================================================", flush=True)
    
    models = [
        "sdlg_exact",            # Fam A baseline (Additive)
        "sdlg_delta",            # Fam A (Delta rule)
        "sdlg_adam",             # Fam A (Gated Delta with +3.0 erase bias)
        "sdlg_adam_matched",     # Fam E (Param matched control d_ffc=448)
        "sdlg_channeldecay",     # Fam B (Decoupled Erase Gate + Additive write)
        "sdlg_window32",         # Fam C (SWA Window size W=32)
        "sdlg_approx_adam_r4",   # Fam D (Rank r=4)
        "sdlg_approx_adam_r16",  # Fam D (Rank r=16)
        "sdlg_depth4_r4",        # Fam F (Depth L=4, r=4)
        "sdlg_lineardenom",      # Fam G (Linear denominator)
    ]
    
    suite_start = time.time()

    # ── Benchmark 1: Multi-Query Associative Recall (MQAR, N=16 pairs) ─────────────
    print("\n▶ [1/3] Running Multi-Query Associative Recall (MQAR, N=16 pairs)...", flush=True)
    mqar_results = []
    for m in models:
        res = train_and_eval_model(m, task="mqar", num_kv_pairs=16, num_steps=120)
        mqar_results.append(res)
        print(f"  [{m:<22}] Accuracy: {res['eval_acc']*100:>5.1f}% | Loss: {res['eval_loss']:>6.3f} | Time: {res['time_sec']:>4.1f}s", flush=True)

    df_mqar = pd.DataFrame(mqar_results)
    df_mqar.to_csv(os.path.join(OUTPUT_DIR, "fast_mqar_n16.csv"), index=False)

    # ── Benchmark 2: Selective Overwrite & Collateral Damage (N=16) ──────────
    print("\n▶ [2/3] Running Selective Overwrite & Collateral Damage (N=16)...", flush=True)
    overwrite_results = []
    for m in models:
        res = train_and_eval_model(m, task="overwrite", num_kv_pairs=16, num_steps=120)
        overwrite_results.append(res)
        print(f"  [{m:<22}] Accuracy: {res['eval_acc']*100:>5.1f}% | Loss: {res['eval_loss']:>6.3f} | Time: {res['time_sec']:>4.1f}s", flush=True)

    df_overwrite = pd.DataFrame(overwrite_results)
    df_overwrite.to_csv(os.path.join(OUTPUT_DIR, "fast_overwrite_n16.csv"), index=False)

    # ── Benchmark 3: Capacity Saturation Sweep (N ∈ {8, 16, 32}) ──────────
    print("\n▶ [3/3] Running Capacity Saturation Sweep (MQAR across N ∈ {8, 16, 32})...", flush=True)
    capacity_results = []
    for N in [8, 16, 32]:
        print(f"\n --- N = {N} Key-Value Pairs ---", flush=True)
        for m in ["sdlg_exact", "sdlg_delta", "sdlg_adam", "sdlg_approx_adam_r4", "sdlg_approx_adam_r16"]:
            res = train_and_eval_model(m, task="mqar", num_kv_pairs=N, num_steps=120)
            capacity_results.append(res)
            print(f"  [{m:<22}] N={N:>2} | Accuracy: {res['eval_acc']*100:>5.1f}% | Loss: {res['eval_loss']:>6.3f} | Time: {res['time_sec']:>4.1f}s", flush=True)

    df_capacity = pd.DataFrame(capacity_results)
    df_capacity.to_csv(os.path.join(OUTPUT_DIR, "fast_capacity_sweep.csv"), index=False)

    total_suite_time = time.time() - suite_start
    print("\n" + "=" * 85, flush=True)
    print(f"🎉 FAST UNIFIED MEMORY ABLATION SUITE COMPLETE (Total Time: {total_suite_time:.1f}s)", flush=True)
    print("=" * 85, flush=True)


if __name__ == "__main__":
    main()
