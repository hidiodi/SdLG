"""Regression checks for carried/reset T-Maze rollout versus PPO recomputation."""
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.diagnostics.exact_sdlg.exact_sdlg_tmaze_ablations import (
    ActorCriticAblation, ExactSdLGCarry, MATCHED_ABLATIONS,
)
from src.models.common.sdlg_hybrid import BatchedSdLGHybrid, GatedSlidingWindowAttention


def close(a, b, atol=2e-5):
    errors = []
    for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)):
        np.testing.assert_allclose(x, y, rtol=2e-4, atol=atol)
        errors.append(float(jnp.max(jnp.abs(x - y))))
    return max(errors, default=0.0)


def run():
    results = {}
    # Independent scalar oracle: mean over this episode's last W values.
    mod = GatedSlidingWindowAttention(1, 1, 1, 4)
    x = jnp.arange(1, 13, dtype=jnp.float32)[:, None]
    done = jnp.zeros(12).at[6].set(1).at[7].set(1)
    zero = jnp.zeros((4, 1))
    params = mod.init(jax.random.PRNGKey(0), x, done, zero, zero, jnp.array(0.0))
    params['params']['joint_swa_proj']['kernel'] = jnp.array([[0., 0., 1., 0.]])
    params['params']['joint_swa_proj']['bias'] = jnp.zeros(4)
    params['params']['out_proj']['kernel'] = jnp.ones((1, 1))
    params['params']['out_proj']['bias'] = jnp.zeros(1)
    expected, history = [], []
    for value, reset in zip(np.asarray(x[:, 0]), np.asarray(done)):
        if reset:
            history = []
        history.append(float(value))
        expected.append(.5 * np.mean(history[-4:]))
    for lengths in ([1] * 12, [3, 4, 5], [12]):
        k, v, count, start, outputs = zero, zero, jnp.array(0.), 0, []
        for length in lengths:
            stop = start + length
            out, k, v = mod.apply(params, x[start:stop], done[start:stop], k, v, count)
            for reset in done[start:stop]:
                count = count * (1 - reset) + 1
            outputs.append(out)
            start = stop
        close(jnp.concatenate(outputs)[:, 0], jnp.array(expected))
    results['attention_scalar_oracle'] = 'passed'

    for name, ablation in MATCHED_ABLATIONS.items():
        cfg = {'ABLATION': ablation, 'RNN': {
            'RNN_TYPE': 'sdlg_hybrid', 'D_MODEL': 128, 'N_HEADS': 4,
            'N_LAYERS': 3, 'D_FFC': 512, 'HIDDEN_DIM': 128, 'R': 16, 'ALPHA': 1.0,
        }}
        if ablation['USE_EXACT']:
            raw = tuple(ExactSdLGCarry(
                M=jnp.zeros((4, 32, 32)), z=jnp.zeros((4, 32)),
                swa_k=jnp.zeros((16, 32)), swa_v=jnp.zeros((16, 32)),
                step_count=jnp.array(0.)) for _ in range(3))
        else:
            raw = BatchedSdLGHybrid.initialize_carry(2, 3, 32, 32, 16, 16)
        initial = jax.tree_util.tree_map(lambda z: jnp.repeat(z[None], 2, axis=0), raw)
        network = ActorCriticAblation(2, cfg)
        obs = jax.random.normal(jax.random.PRNGKey(19), (40, 2, 2))
        dones = jnp.zeros((40, 2)).at[6, 0].set(1).at[17, 1].set(1).at[18, 1].set(1).at[35, 0].set(1)
        p = network.init(jax.random.PRNGKey(7), initial, (obs[:1], dones[:1]))
        apply = jax.jit(network.apply)

        def execute(lengths):
            carry, outputs, start = initial, [], 0
            for length in lengths:
                logits, values, bonus, carry = apply(p, carry, (obs[start:start+length], dones[start:start+length]))
                outputs.append((logits, values, bonus))
                start += length
            return jax.tree_util.tree_map(lambda *z: jnp.concatenate(z), *outputs), carry

        reference = execute([1] * 40)
        errors, carry_errors = [], []
        for lengths in ([40], [3, 16, 16, 5], [16, 16, 8]):
            candidate = execute(lengths)
            errors.append(close(reference[0], candidate[0]))
            # Parallel and serial float32 associative reductions accumulate
            # roundoff in the memory state; keep policy/value checks tighter.
            carry_errors.append(close(reference[1], candidate[1], atol=1e-4))
        # A reset must be equivalent to a fresh start, including counter/phase.
        fresh = apply(p, initial, (obs[:1], jnp.zeros((1, 2))))
        reset = apply(p, reference[1], (obs[:1], jnp.ones((1, 2))))
        errors.append(close(fresh, reset))
        results[name] = {'max_output_difference': max(errors),
                         'max_carry_difference': max(carry_errors), 'passed': True}
        print(json.dumps({name: results[name]}), flush=True)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    run()
