"""Check the Exact associative scan against sequential recurrence semantics.

The checks cover a carried state, a chunk boundary, and asynchronous episode
terminations. They validate the reference operator printed in Appendix G.
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def associative_scan(gamma, writes_m, writes_z, dones, carry):
    """Parallel recurrence with carried state and episode resets."""
    gamma_eff = gamma * (1.0 - dones[..., None])
    writes_m = writes_m.at[:, 0].add(gamma_eff[:, 0, :, None, None] * carry[0])
    writes_z = writes_z.at[:, 0].add(gamma_eff[:, 0, :, None] * carry[1])

    def combine(a, b):
        gamma_a, memory_a, normalizer_a = a
        gamma_b, memory_b, normalizer_b = b
        return (
            gamma_b * gamma_a,
            gamma_b[..., None, None] * memory_a + memory_b,
            gamma_b[..., None] * normalizer_a + normalizer_b,
        )

    _, memory, normalizer = jax.lax.associative_scan(
        combine, (gamma_eff, writes_m, writes_z), axis=1
    )
    return memory, normalizer, (memory[:, -1], normalizer[:, -1])


def sequential(gamma, writes_m, writes_z, dones, carry):
    """Step-by-step reference recurrence."""
    memory, normalizer = carry
    memories, normalizers = [], []
    for step in range(gamma.shape[1]):
        effective = gamma[:, step] * (1.0 - dones[:, step, None])
        memory = effective[..., None, None] * memory + writes_m[:, step]
        normalizer = effective[..., None] * normalizer + writes_z[:, step]
        memories.append(memory)
        normalizers.append(normalizer)
    return jnp.stack(memories, axis=1), jnp.stack(normalizers, axis=1), (memory, normalizer)


def assert_close(name, observed, expected):
    error = float(jnp.max(jnp.abs(observed - expected)))
    if error > 1e-5:
        raise AssertionError(f"{name} max abs error {error:.3e}")
    print(f"PASS {name}: max abs error {error:.3e}")


def check_factorized_carry_semantics():
    """Exercise the factorized recurrent scan used by SdLG-Approx."""
    from src.models.common.sdlg_hybrid import parallel_recurrent_scan

    key = jax.random.PRNGKey(19)
    key, write_key, discount_key, carry_key = jax.random.split(key, 4)
    writes = jax.random.normal(write_key, (37, 3, 5))
    discounts = 0.1 + 0.8 * jax.random.uniform(discount_key, (37, 3, 1))
    carry = jax.random.normal(carry_key, (3, 5))

    def sequential_factorized(start, ws, ds):
        state, states = start, []
        for write, discount in zip(ws, ds):
            state = discount * state + write
            states.append(state)
        return jnp.stack(states), state

    seq, seq_final = sequential_factorized(carry, writes, discounts)
    scanned = parallel_recurrent_scan(carry, writes, discounts)
    assert_close("factorized-full", scanned, seq)
    split = 17
    first = parallel_recurrent_scan(carry, writes[:split], discounts[:split])[-1]
    tail = parallel_recurrent_scan(first, writes[split:], discounts[split:])
    assert_close("factorized-chunked", tail, seq[split:])
    assert_close("factorized-final", tail[-1], seq_final)


def check_reset_and_tbptt_semantics():
    """Check reset-before-write semantics and a detached chunk boundary."""
    carry = jnp.array([9.0]).reshape(1, 1, 1, 1)
    writes = jnp.array([1.0, 2.0]).reshape(1, 2, 1, 1, 1)
    writes_z = jnp.array([1.0, 2.0]).reshape(1, 2, 1, 1)
    gamma = jnp.ones((1, 2, 1))
    dones = jnp.array([[1.0, 0.0]])
    states, _, _ = sequential(gamma, writes, writes_z, dones, (carry, carry[..., 0]))
    # At the new-episode observation, the old carry is cleared but the current
    # write remains. This matches the runner's last_done convention.
    assert_close("reset-before-current-write", states[0, 0], writes[0, 0])

    def loss(x_first, x_second, weight):
        carried = jax.lax.stop_gradient(weight * x_first)
        return jnp.sum(weight * carried * x_second)

    grad_x_first = jax.grad(loss, argnums=0)(jnp.array([2.0]), jnp.array([3.0]), jnp.array([4.0]))
    grad_weight = jax.grad(loss, argnums=2)(jnp.array([2.0]), jnp.array([3.0]), jnp.array([4.0]))
    assert_close("tbptt-detach-input", grad_x_first, jnp.zeros_like(grad_x_first))
    if float(jnp.max(jnp.abs(grad_weight))) == 0.0:
        raise AssertionError("TBPTT detach incorrectly removed parameter gradients")
    print("PASS tbptt-detach-parameters: nonzero gradient")


def check_normalized_delta_eigenvalue():
    """Validate the Appendix F.6 rank-one feedback eigenvalue formula."""
    gamma, beta, epsilon = 0.91, 0.37, 1e-4
    key = jax.random.PRNGKey(23)
    vector = jax.random.normal(key, (7,))
    feedback = gamma * jnp.eye(vector.size) - beta * jnp.outer(vector, vector) / (
        jnp.dot(vector, vector) + epsilon
    )
    observed = jnp.linalg.eigvals(feedback)
    expected_key = gamma - beta * jnp.dot(vector, vector) / (jnp.dot(vector, vector) + epsilon)
    expected = jnp.sort(jnp.concatenate([jnp.full((vector.size - 1,), gamma), expected_key[None]]))
    assert_close("normalized-delta-eigenvalues", jnp.sort(jnp.real(observed)), expected)


def main():
    batch, time, heads, d_k, d_v = 3, 11, 2, 3, 4
    key = jax.random.PRNGKey(7)
    key, k1, k2, k3, k4, k5 = jax.random.split(key, 6)
    gamma = 0.1 + 0.8 * jax.random.uniform(k1, (batch, time, heads))
    writes_m = jax.random.normal(k2, (batch, time, heads, d_k, d_v))
    writes_z = jax.random.normal(k3, (batch, time, heads, d_k))
    carry = (
        jax.random.normal(k4, (batch, heads, d_k, d_v)),
        jax.random.normal(k5, (batch, heads, d_k)),
    )
    dones = jnp.array(
        [[0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]],
        dtype=jnp.float32,
    )

    scan_m, scan_z, scan_carry = associative_scan(gamma, writes_m, writes_z, dones, carry)
    seq_m, seq_z, seq_carry = sequential(gamma, writes_m, writes_z, dones, carry)
    assert_close("full-memory", scan_m, seq_m)
    assert_close("full-normalizer", scan_z, seq_z)
    assert_close("full-carry-memory", scan_carry[0], seq_carry[0])
    assert_close("full-carry-normalizer", scan_carry[1], seq_carry[1])

    split = 5
    _, _, first_carry = associative_scan(
        gamma[:, :split], writes_m[:, :split], writes_z[:, :split], dones[:, :split], carry
    )
    tail_m, tail_z, tail_carry = associative_scan(
        gamma[:, split:], writes_m[:, split:], writes_z[:, split:], dones[:, split:], first_carry
    )
    assert_close("chunked-memory", tail_m, seq_m[:, split:])
    assert_close("chunked-normalizer", tail_z, seq_z[:, split:])
    assert_close("chunked-carry-memory", tail_carry[0], seq_carry[0])
    assert_close("chunked-carry-normalizer", tail_carry[1], seq_carry[1])
    check_factorized_carry_semantics()
    check_reset_and_tbptt_semantics()
    check_normalized_delta_eigenvalue()


if __name__ == "__main__":
    main()
