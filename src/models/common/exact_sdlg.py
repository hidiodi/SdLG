# src/models/common/exact_sdlg.py

import jax
import jax.numpy as jnp
import flax.linen as nn
import sys
from typing import Tuple, NamedTuple, Dict

# Import your existing, battle-tested components
from src.models.common.sdlg_hybrid import GatedSlidingWindowAttention, GTrXLGRUGate
from flax.linen.initializers import constant, orthogonal, zeros


class ExactSdLGCarry(NamedTuple):
    M: jnp.ndarray  # [n_heads, d_k, d_v]
    z: jnp.ndarray  # [n_heads, d_k]
    swa_k: jnp.ndarray  # [W, d_k]
    swa_v: jnp.ndarray  # [W, d_v]
    step_count: jnp.ndarray  # []


class ExactMultiScaleGLA(nn.Module):
    d_model: int
    n_heads: int
    d_k: int
    d_v: int
    rope_variant: str = "none"

    def setup(self):
        # Base timescales for exponential decay (log-spaced half-lives)
        base_timescales = jnp.array([16.0, 64.0, 256.0, 1024.0, 4096.0, 16384.0])

        if self.n_heads > len(base_timescales):
            pad_size = self.n_heads - len(base_timescales)
            extra_scales = base_timescales[-1] * (4.0 ** jnp.arange(1, pad_size + 1))
            timescales = jnp.concatenate([base_timescales, extra_scales])
        else:
            timescales = base_timescales[:self.n_heads]

        # lambda_h = ln(2) / Half_Life
        lambda_h = jnp.log(2.0) / timescales

        # Inverse softplus for correct initialization: log(exp(lambda_h) - 1)
        self.gamma_bias = jnp.log(jnp.exp(lambda_h) - 1.0)

        # Projections
        self.q_proj = nn.Dense(self.n_heads * self.d_k, use_bias=False, name="q_proj")
        self.k_proj = nn.Dense(self.n_heads * self.d_k, use_bias=False, name="k_proj")
        self.v_proj = nn.Dense(self.n_heads * self.d_v, use_bias=False, name="v_proj")
        self.out_proj = nn.Dense(self.d_model, use_bias=False, name="out_proj")

        # Initialize gate kernels to 0.0 so the mathematical bias is cleanly preserved at step 0
        self.delta_lambda_proj = nn.Dense(self.n_heads, use_bias=False, kernel_init=zeros, name="gamma_gate")
        self.beta_proj = nn.Dense(self.n_heads, use_bias=False, kernel_init=zeros, name="beta_gate")

    def __call__(self, inputs: jnp.ndarray, terminations: jnp.ndarray, carry: Tuple[jnp.ndarray, jnp.ndarray], step_counts: jnp.ndarray = None) -> Tuple[
        jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray], Dict[str, jnp.ndarray]]:
        T, _ = inputs.shape
        M_prev, z_prev = carry

        # Projections
        q = self.q_proj(inputs).reshape((T, self.n_heads, self.d_k))
        k = self.k_proj(inputs).reshape((T, self.n_heads, self.d_k))
        v = self.v_proj(inputs).reshape((T, self.n_heads, self.d_v))

        delta_lambda = self.delta_lambda_proj(inputs)
        beta_raw = self.beta_proj(inputs)

        # RMSNorm on q and k before ELU to prevent vanishing query/key gradients
        q_rms = q / (jnp.sqrt(jnp.mean(q**2, axis=-1, keepdims=True)) + 1e-5) * 2.0
        k_rms = k / (jnp.sqrt(jnp.mean(k**2, axis=-1, keepdims=True)) + 1e-5) * 2.0

        if self.rope_variant == "fast_head" and step_counts is not None:
            from src.models.common.sdlg_hybrid import rope_rotate
            omegas = jnp.linspace(-jnp.pi, jnp.pi, self.n_heads)
            theta = (step_counts[..., None] * omegas) % (2 * jnp.pi)
            theta = theta[..., None] # Shape: (T, n_heads, 1)
            q_rot = rope_rotate(q_rms[..., None, :], theta).squeeze(-2)
            k_rot = rope_rotate(k_rms[..., None, :], theta).squeeze(-2)
            phi_q = jax.nn.elu(q_rot) + 1.0
            phi_k = jax.nn.elu(k_rot) + 1.0
        else:
            # Positive Feature Map
            phi_q = jax.nn.elu(q_rms) + 1.0
            phi_k = jax.nn.elu(k_rms) + 1.0

        # Gating
        gamma = jnp.exp(-jax.nn.softplus(delta_lambda + self.gamma_bias))
        beta = jax.nn.sigmoid(beta_raw)

        # Construct inputs for the parallel associative scan
        write_M = beta[..., None, None] * (phi_k[..., None] * v[..., None, :])  # [T, n_heads, d_k, d_v]
        write_z = beta[..., None] * phi_k  # [T, n_heads, d_k]

        # Apply termination masks directly to discounts to cleanly clear historical carry states at boundaries
        discount_gamma = gamma * (1.0 - terminations[:, None])
        discount_M = discount_gamma[..., None, None]  # [T, n_heads, 1, 1]
        discount_z = discount_gamma[..., None]  # [T, n_heads, 1]

        # --- PARALLEL RECURRENT SCAN (ASSOCIATIVE OPERATOR) ---
        def combine(a, b):
            w_a, d_a = a
            w_b, d_b = b
            return w_b + d_b * w_a, d_a * d_b

        # Step 0 Carry Injection
        write_M_0 = write_M[0] + discount_M[0] * M_prev
        write_M = write_M.at[0].set(write_M_0)

        write_z_0 = write_z[0] + discount_z[0] * z_prev
        write_z = write_z.at[0].set(write_z_0)

        # Associative Scan over the sequence dimension (axis 0) with O(log T) parallel depth
        all_M, _ = jax.lax.associative_scan(combine, (write_M, discount_M), axis=0)
        all_z, _ = jax.lax.associative_scan(combine, (write_z, discount_z), axis=0)

        # Retrieval
        num = jnp.matmul(jnp.expand_dims(phi_q, -2), all_M).squeeze(-2)
        den = jnp.sum(phi_q * all_z, axis=-1, keepdims=True) + 1e-6
        y = num / den

        y_flat = y.reshape((T, self.n_heads * self.d_v))
        out = self.out_proj(y_flat)

        # Telemetry averaged over sequence dimension
        telemetry = {
            "beta_mean": jnp.mean(beta),
            "gamma_head_0": jnp.mean(gamma[:, 0]),
            "gamma_head_n": jnp.mean(gamma[:, -1]),
        }

        new_carry = (all_M[-1], all_z[-1])
        return out, new_carry, telemetry


class ExactSdLGHybridBlock(nn.Module):
    d_model: int
    n_heads: int
    d_k: int
    d_v: int
    d_ffc: int
    window_size: int
    gru_bias: float = 2.0
    rope_variant: str = "none"

    @nn.compact
    def __call__(self, inputs, terminations, carry: ExactSdLGCarry, deterministic=True):
        T, _ = inputs.shape

        # Step 1: Local Sliding Window Attention
        norm_inputs_swa = nn.LayerNorm()(inputs)
        swa_out, new_swa_k, new_swa_v = GatedSlidingWindowAttention(
            d_model=self.d_model, d_k=self.d_k, d_v=self.d_v, window_size=self.window_size, name="swa"
        )(norm_inputs_swa, terminations, carry.swa_k, carry.swa_v, carry.step_count, deterministic=deterministic)

        x_swa = GTrXLGRUGate(self.d_model, self.gru_bias, name="gru_swa")(inputs, swa_out)

        # Update step count
        if T > 1:
            def compute_step_count(carry_sc, term):
                next_step = (carry_sc + 1.0) * (1.0 - term)
                return next_step, next_step

            _, step_counts = jax.lax.scan(compute_step_count, carry.step_count, terminations)
            new_step_count = step_counts[-1]
        else:
            new_step_count = (carry.step_count + 1.0) * (1.0 - terminations[0])

        # Step 2: Global Exact GLA
        norm_inputs_gla = nn.LayerNorm()(x_swa)
        gla_out, (new_M, new_z), telemetry = ExactMultiScaleGLA(
            d_model=self.d_model, n_heads=self.n_heads, d_k=self.d_k, d_v=self.d_v, rope_variant=self.rope_variant, name="exact_gla"
        )(norm_inputs_gla, terminations, (carry.M, carry.z), step_counts=new_step_count)

        x_gla = GTrXLGRUGate(self.d_model, self.gru_bias, name="gru_gla")(x_swa, gla_out)

        # Step 3: Feed-Forward
        norm_inputs_ffn = nn.LayerNorm()(x_gla)
        ffn_out = nn.Sequential([
            nn.Dense(self.d_ffc, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)),
            nn.gelu,
            nn.Dense(self.d_model, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))
        ])(norm_inputs_ffn)

        out = GTrXLGRUGate(self.d_model, self.gru_bias, name="gru_ffn")(x_gla, ffn_out)

        new_carry = ExactSdLGCarry(M=new_M, z=new_z, swa_k=new_swa_k, swa_v=new_swa_v, step_count=new_step_count)

        return out, new_carry, telemetry


class UnbatchedExactSdLG(nn.Module):
    n_layers: int
    d_model: int
    n_heads: int
    d_k: int
    d_v: int
    d_ffc: int
    window_size: int
    rope_variant: str = "none"

    @nn.compact
    def __call__(self, inputs, terminations, carry, deterministic=True):
        x = nn.Dense(self.d_model)(inputs)
        x = nn.relu(x)

        new_carry = []
        all_telemetry = {}

        for l in range(self.n_layers):
            x, next_carry_layer, telemetry = ExactSdLGHybridBlock(
                d_model=self.d_model, n_heads=self.n_heads, d_k=self.d_k, d_v=self.d_v,
                d_ffc=self.d_ffc, window_size=self.window_size, rope_variant=self.rope_variant, name=f"block_{l}"
            )(x, terminations, carry[l], deterministic)

            new_carry.append(next_carry_layer)
            if l == self.n_layers - 1:  # Extract telemetry from the final reasoning layer
                all_telemetry = telemetry

        return x, tuple(new_carry), all_telemetry

    @staticmethod
    def initialize_carry(n_layers: int, n_heads: int, d_k: int, d_v: int, window_size: int) -> Tuple[
        ExactSdLGCarry, ...]:
        carry_list = []
        for _ in range(n_layers):
            M = jnp.zeros((n_heads, d_k, d_v), dtype=jnp.float32)
            z = jnp.zeros((n_heads, d_k), dtype=jnp.float32)
            swa_k = jnp.zeros((window_size, d_k), dtype=jnp.float32)
            swa_v = jnp.zeros((window_size, d_v), dtype=jnp.float32)
            step_count = jnp.zeros((), dtype=jnp.float32)

            carry_list.append(ExactSdLGCarry(M=M, z=z, swa_k=swa_k, swa_v=swa_v, step_count=step_count))
        return tuple(carry_list)


# =======================================================================
# THE FUNCTIONAL SMOKE TESTS
# =======================================================================
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  EXACT GLA MATHEMATICS VERIFICATION PROTOCOL  ")
    print("=" * 50)

    rng = jax.random.PRNGKey(42)
    B, T, D_IN = 4, 128, 64
    CONFIG = {"n_layers": 2, "d_model": 128, "n_heads": 4, "d_k": 32, "d_v": 32, "d_ffc": 256, "window_size": 16}

    model = UnbatchedExactSdLG(**CONFIG)

    BatchedModel = nn.vmap(
        UnbatchedExactSdLG, in_axes=(0, 0, 0, None), out_axes=(0, 0, 0),
        variable_axes={'params': None}, split_rngs={'params': False}
    )

    dummy_obs = jnp.ones((B, T, D_IN))
    dummy_terms = jnp.zeros((B, T))

    unbatched_carry = UnbatchedExactSdLG.initialize_carry(
        CONFIG["n_layers"], CONFIG["n_heads"], CONFIG["d_k"], CONFIG["d_v"], CONFIG["window_size"]
    )
    batched_carry = jax.tree_util.tree_map(lambda x: jnp.repeat(x[None, ...], B, axis=0), unbatched_carry)

    variables = model.init(rng, dummy_obs[0], dummy_terms[0], unbatched_carry, True)

    print("\n[TEST A: JIT & Shape Compilation]")
    try:
        @jax.jit
        def compile_test(obs, term, carry):
            return BatchedModel(**CONFIG).apply(variables, obs, term, carry, True)


        out, new_c, telemetry = compile_test(dummy_obs, dummy_terms, batched_carry)
        print("  ✅ PASS: VMAP Isolation successful. No BatchTracer leaks. Output Shape:", out.shape)
        del out, new_c
    except Exception as e:
        print("  ❌ FAIL: Shape compilation crashed:", e)
        sys.exit(1)

    print("\n[TEST B: FP32 Bound Check (NaN Trap)]")
    try:
        long_noise = jax.random.normal(rng, (B, 10000, D_IN))
        long_terms = jnp.zeros((B, 10000))
        _, final_carry, _ = compile_test(long_noise, long_terms, batched_carry)

        last_M = final_carry[-1].M
        max_val = jnp.max(jnp.abs(last_M))

        if jnp.isnan(max_val) or jnp.isinf(max_val):
            print("  ❌ FAIL: Memory Matrix M_t exploded to NaN/Inf!")
            sys.exit(1)
        elif max_val > 1e15:
            print(f"  ❌ FAIL: Memory Matrix M_t bounds exploded beyond Float32 limits ({max_val})")
            sys.exit(1)
        else:
            print(f"  ✅ PASS: ELU+1 & Gamma Bounds stable after 10,000 steps. Max Value: {max_val:.2f}")
    except Exception as e:
        print("  ❌ FAIL: Bound Check crashed:", e)
        sys.exit(1)

    print("\n[TEST C: Logarithmic Gate Initialization]")
    try:
        beta = float(jnp.mean(telemetry["beta_mean"]))
        g0 = float(jnp.mean(telemetry["gamma_head_0"]))
        gN = float(jnp.mean(telemetry["gamma_head_n"]))

        if not (0.45 < beta < 0.55):
            print(f"  ❌ FAIL: Write Gate not initialized near 0.5. Got: {beta:.3f}")
            sys.exit(1)
        elif g0 >= gN:
            print(f"  ❌ FAIL: Gamma timescale array reversed or broken. Head 0: {g0:.3f}, Head N: {gN:.3f}")
            sys.exit(1)
        else:
            print(
                f"  ✅ PASS: Gates healthy. Beta Init: {beta:.2f} | Rapid Decay H0: {g0:.4f} | Slow Decay HN: {gN:.4f}")
    except Exception as e:
        print("  ❌ FAIL: Gate Check crashed:", e)
        sys.exit(1)

    print("\n" + "=" * 50)
    print("  ALL TESTS PASSED. CORE IS STABLE FOR RL.  ")
    print("=" * 50 + "\n")
