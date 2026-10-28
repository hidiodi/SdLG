# scripts/diagnostics/exact_sdlg/verify_param_budget.py
import jax
import jax.numpy as jnp
import flax.linen as nn
from src.models.purejaxrl.wrappers import (
    ActorCriticExactSdLG,
    ActorCriticSdLGDelta,
    ActorCriticSdLGADAM,
    ActorCriticSdLGHybrid,
    ActorCriticSdLGApproxDelta,
    ActorCriticSdLGApproxADAM,
)
from src.models.common.exact_sdlg import UnbatchedExactSdLG
from src.models.common.sdlg_hybrid import BatchedSdLGHybrid

def count_params(model_cls, is_exact=True):
    config = {
        "RNN": {
            "D_MODEL": 64,
            "N_LAYERS": 3,
            "N_HEADS": 4,
            "D_K": 32,
            "D_V": 32,
            "D_FFC": 512,
            "R": 4,
            "WINDOW_SIZE": 16,
        }
    }
    dummy_obs = jnp.zeros((16, 1, 64))  # (time, batch, obs_dim)
    dummy_dones = jnp.zeros((16, 1))    # (time, batch)
    
    if is_exact:
        unbatched = UnbatchedExactSdLG.initialize_carry(n_layers=3, n_heads=4, d_k=32, d_v=32, window_size=16)
        carry = jax.tree_util.tree_map(lambda x: jnp.repeat(jnp.expand_dims(x, 0), 1, axis=0), unbatched)
    else:
        carry = BatchedSdLGHybrid.initialize_carry(batch_size=1, n_layers=3, d_k=32, d_v=32, r=4, window_size=16)
        carry = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, 0), carry)

    model = model_cls(action_dim=2, config=config)
    rng = jax.random.PRNGKey(42)
    params = model.init(rng, carry, (dummy_obs, dummy_dones))
    
    flat_params, _ = jax.tree_util.tree_flatten(params)
    total_count = sum(p.size for p in flat_params)
    return total_count

if __name__ == "__main__":
    print("=" * 70)
    print("      SDLG PARAMETER BUDGET VERIFICATION (2x3 Factorial Matrix)   ")
    print("=" * 70)
    
    exact_baseline = count_params(ActorCriticExactSdLG, is_exact=True)
    exact_delta = count_params(ActorCriticSdLGDelta, is_exact=True)
    exact_adam = count_params(ActorCriticSdLGADAM, is_exact=True)
    
    # Import the Matched and ChannelDecay wrappers internally for this check
    from src.models.purejaxrl.wrappers import ActorCriticSdLGADAMMatched
    exact_adam_matched = count_params(ActorCriticSdLGADAMMatched, is_exact=True)

    approx_baseline = count_params(ActorCriticSdLGHybrid, is_exact=False)
    approx_delta = count_params(ActorCriticSdLGApproxDelta, is_exact=False)
    approx_adam = count_params(ActorCriticSdLGApproxADAM, is_exact=False)

    print(f"[Exact Family]")
    print(f"  - SdLG-Exact         (Additive)   : {exact_baseline:,} parameters")
    print(f"  - SdLG-Delta         (Delta)      : {exact_delta:,} parameters | Overhead: {((exact_delta-exact_baseline)/exact_baseline)*100:+.2f}%")
    print(f"  - SdLG-ADAM          (Gated Delta): {exact_adam:,} parameters | Overhead: {((exact_adam-exact_baseline)/exact_baseline)*100:+.2f}%")
    print(f"  - SdLG-ADAM-Matched  (Ctrl)       : {exact_adam_matched:,} parameters | Overhead: {((exact_adam_matched-exact_baseline)/exact_baseline)*100:+.2f}%")
    print(f"  - SdLG-ChannelDecay  (Family B)   : {exact_delta:,} parameters | Overhead: {((exact_delta-exact_baseline)/exact_baseline)*100:+.2f}%")

    print(f"\n[Approx Family (Low-Rank r=4)]")
    print(f"  - SdLG-Approx       (Additive)   : {approx_baseline:,} parameters")
    print(f"  - SdLG-Approx-Delta (Delta)      : {approx_delta:,} parameters | Overhead: {((approx_delta-approx_baseline)/approx_baseline)*100:+.2f}%")
    print(f"  - SdLG-Approx-ADAM  (Gated Delta): {approx_adam:,} parameters | Overhead: {((approx_adam-approx_baseline)/approx_baseline)*100:+.2f}%")
    print("=" * 70)
