import jax
import jax.numpy as jnp
import functools

def build_chunked_evaluator(env, env_params, network, rnn_carry_init, eval_steps, is_deterministic: bool = False):
    """
    Generic chunked evaluator designed to evaluate sequence architectures
    at arbitrary corridor lengths with stochastic step sampling.
    """
    @functools.partial(jax.jit, static_argnums=(3,))
    def eval_chunk(params, start_seed, base_rng, chunk_size):
        keys = jax.random.split(jax.random.PRNGKey(start_seed), chunk_size)
        reset_fn = jax.vmap(env.reset, in_axes=(0, None))
        step_fn = jax.vmap(env.step, in_axes=(0, 0, 0, None))

        obs, env_state = reset_fn(keys, env_params)
        hstate = rnn_carry_init(chunk_size)

        def step_env(carry, step_idx):
            obs, env_state, hstate, rng, cum_reward, success = carry

            # Use same fixed length base_rng sequence across models for fair comparison
            rng, step_rng = jax.random.split(rng)
            action_keys = jax.random.split(step_rng, chunk_size)

            # Predict
            pi, val, _ = jax.vmap(network.apply, in_axes=(None, 0, 0))(
                params, hstate, obs
            )

            if is_deterministic:
                actions = pi.mode()
            else:
                actions = pi.sample(seed=action_keys)

            # Step environment
            next_obs, next_env_state, reward, done, info = step_fn(action_keys, env_state, actions, env_params)

            # Next carry
            next_hstate = jax.vmap(network.initialize_carry, in_axes=(0, 0))(hstate, done)

            new_cum_reward = cum_reward + reward
            new_success = success | info.get("returned_episode_returns", done)

            return (next_obs, next_env_state, next_hstate, rng, new_cum_reward, new_success), (reward, info, done)

        init_carry = (obs, env_state, hstate, base_rng, jnp.zeros(chunk_size), jnp.zeros(chunk_size, dtype=jnp.bool_))

        # Unroll the evaluation
        final_carry, _ = jax.lax.scan(step_env, init_carry, jnp.arange(eval_steps))

        _, _, _, _, final_cum_reward, final_success = final_carry
        return jnp.mean(final_success)

    return eval_chunk
