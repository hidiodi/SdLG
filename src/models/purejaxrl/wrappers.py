# src/models/purejaxrl/wrappers.py

import jax
import jax.numpy as jnp
from flax import linen as nn
import distrax
import functools
from typing import Tuple, Any, Dict, Sequence


class RepresentationNetwork(nn.Module):
    d_model: int

    @nn.compact
    def __call__(self, obs):
        T, B = obs.shape[0], obs.shape[1]
        obs_flat = obs.reshape(T * B, *obs.shape[2:])

        if len(obs.shape) > 3:
            # Convolutional encoder for 2D spatial/pixel/grid observations (Craftax, Minigrid)
            x = nn.Conv(32, kernel_size=(8, 8), strides=(4, 4), padding="VALID")(obs_flat)
            x = nn.relu(x)
            x = nn.Conv(64, kernel_size=(4, 4), strides=(2, 2), padding="VALID")(x)
            x = nn.relu(x)
            x = nn.Conv(64, kernel_size=(3, 3), strides=(1, 1), padding="VALID")(x)
            x = nn.relu(x)
            flat = x.reshape(x.shape[0], -1)
            out = nn.Dense(self.d_model)(flat)
        else:
            # Vector / MLP Observation Trunk for 1D vector observations (POPGym environments)
            # VectorObservation -> Dense(features=d_model) -> ReLU -> Dense(features=d_model) -> Sequence Layer
            flat = obs_flat.reshape(obs_flat.shape[0], -1)
            x = nn.Dense(self.d_model)(flat)
            x = nn.relu(x)
            out = nn.Dense(self.d_model)(x)

        return out.reshape(T, B, self.d_model)


class ActorCriticSdLGHybrid(nn.Module):
    rnn_module: Any = None
    action_dim: int = 2
    config: Dict = None

    @nn.compact
    def __call__(self, carry, x):
        obs, dones = x
        emb = RepresentationNetwork(d_model=self.config["RNN"]["D_MODEL"], name="repr_network")(obs)

        # Transpose to Batch-First (B, T, d_model) for sequence model vmap
        emb_trans = jnp.transpose(emb, (1, 0, 2))
        dones_trans = jnp.transpose(dones, (1, 0))

        # Import standard shared common layers
        from src.models.common.sdlg_hybrid import BatchedSdLGHybrid

        # Instantiate vmapped submodule mapping over Batch axis (dimension 0)
        VmappedSdLG = nn.vmap(
            BatchedSdLGHybrid,
            in_axes=(0, 0, 0, None),
            out_axes=0,
            variable_axes={'params': None},
            split_rngs={'params': False}
        )

        sdlg_seq = VmappedSdLG(
            n_layers=self.config["RNN"]["N_LAYERS"],
            d_model=self.config["RNN"]["D_MODEL"],
            d_k=self.config["RNN"].get("D_K", 32),
            d_v=self.config["RNN"].get("D_V", 32),
            d_ffc=self.config["RNN"].get("D_FFC", 512),
            r=self.config["RNN"].get("R", 16),
            window_size=self.config["RNN"].get("WINDOW_SIZE", 16),
            alpha=self.config["RNN"].get("ALPHA", 0.1),
            eta=self.config["RNN"].get("ETA", 0.0),
            use_rope_phase=self.config["RNN"].get("USE_ROPE_PHASE", True),
            use_softplus_denominator=self.config["RNN"].get("USE_SOFTPLUS_DENOMINATOR", True),
            use_per_rank_norm=self.config["RNN"].get("USE_PER_RANK_NORM", True),
            return_split=not self.config["RNN"].get("COUPLED_GGS", False),
            name="sdlg_hybrid"
        )

        out, new_carry = sdlg_seq(emb_trans, dones_trans, carry, True)

        if self.config["RNN"].get("COUPLED_GGS", False):
            (X_shared, intrinsic_bonus_trans) = out
            X_actor_trans = X_shared
            X_critic_trans = X_shared
        else:
            ((X_actor_trans, X_critic_trans), intrinsic_bonus_trans) = out

        X_actor = jnp.transpose(X_actor_trans, (1, 0, 2))
        X_critic = jnp.transpose(X_critic_trans, (1, 0, 2))

        # Actor Logits & Critic Values heads
        raw_logits = nn.Dense(self.action_dim, kernel_init=nn.initializers.orthogonal(0.01), name="actor_head")(X_actor)
        action_logits = 5.0 * jnp.tanh(raw_logits / 5.0)
        action_logits = jnp.nan_to_num(action_logits, nan=0.0, posinf=5.0, neginf=-5.0)
        critic_values = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0), name="critic_head")(X_critic)
        critic_values = jnp.squeeze(critic_values, axis=-1)
        critic_values = jnp.nan_to_num(critic_values, nan=0.0, posinf=10.0, neginf=-10.0)

        return action_logits, critic_values, jnp.transpose(intrinsic_bonus_trans, (1, 0)), new_carry


class ActorCriticExactSdLG(nn.Module):
    rnn_module: Any = None
    action_dim: int = 2
    config: Dict = None

    @nn.compact
    def __call__(self, carry, x):
        obs, dones = x
        emb = RepresentationNetwork(d_model=self.config["RNN"]["D_MODEL"], name="repr_network")(obs)
        emb_b_first = jnp.swapaxes(emb, 0, 1)
        dones_b_first = jnp.swapaxes(dones, 0, 1)

        from src.models.common.exact_sdlg import UnbatchedExactSdLG
        BatchedExactSdLG = nn.vmap(
            UnbatchedExactSdLG, in_axes=(0, 0, 0, None), out_axes=(0, 0, 0),
            variable_axes={'params': None}, split_rngs={'params': False}
        )

        exact_sdlg = BatchedExactSdLG(
            n_layers=self.config["RNN"].get("N_LAYERS", 3),
            d_model=self.config["RNN"]["D_MODEL"],
            n_heads=self.config["RNN"].get("N_HEADS", 4),
            d_k=self.config["RNN"].get("D_K", 32),
            d_v=self.config["RNN"].get("D_V", 32),
            d_ffc=self.config["RNN"].get("D_FFC", 512),
            window_size=self.config["RNN"].get("WINDOW_SIZE", 16),
            rope_variant=self.config["RNN"].get("ROPE_VARIANT", "none")
        )

        outs_b_first, new_carry, _ = exact_sdlg(emb_b_first, dones_b_first, carry, True)
        outs = jnp.swapaxes(outs_b_first, 0, 1)

        raw_logits = nn.Dense(self.action_dim, kernel_init=nn.initializers.orthogonal(0.01))(outs)
        action_logits = 5.0 * jnp.tanh(raw_logits / 5.0)
        action_logits = jnp.nan_to_num(action_logits, nan=0.0, posinf=5.0, neginf=-5.0)
        critic_values = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(outs)
        critic_values = jnp.squeeze(critic_values, axis=-1)
        critic_values = jnp.nan_to_num(critic_values, nan=0.0, posinf=10.0, neginf=-10.0)

        pi = distrax.Categorical(logits=action_logits)
        return new_carry, pi, critic_values


class ActorCriticSdLGDelta(nn.Module):
    rnn_module: Any = None
    action_dim: int = 2
    config: Dict = None

    @nn.compact
    def __call__(self, carry, x):
        obs, dones = x
        emb = RepresentationNetwork(d_model=self.config["RNN"]["D_MODEL"], name="repr_network")(obs)
        emb_b_first = jnp.swapaxes(emb, 0, 1)
        dones_b_first = jnp.swapaxes(dones, 0, 1)

        from src.models.common.sdlg_adam import UnbatchedSdLGADAM
        BatchedSdLGDelta = nn.vmap(
            UnbatchedSdLGADAM, in_axes=(0, 0, 0, None), out_axes=(0, 0, 0),
            variable_axes={'params': None}, split_rngs={'params': False}
        )

        sdlg_delta = BatchedSdLGDelta(
            n_layers=self.config["RNN"].get("N_LAYERS", 3),
            d_model=self.config["RNN"]["D_MODEL"],
            n_heads=self.config["RNN"].get("N_HEADS", 4),
            d_k=self.config["RNN"].get("D_K", 32),
            d_v=self.config["RNN"].get("D_V", 32),
            d_ffc=self.config["RNN"].get("D_FFC", 512),
            window_size=self.config["RNN"].get("WINDOW_SIZE", 16),
            variant="delta"
        )

        outs_b_first, new_carry, _ = sdlg_delta(emb_b_first, dones_b_first, carry, True)
        outs = jnp.swapaxes(outs_b_first, 0, 1)

        raw_logits = nn.Dense(self.action_dim, kernel_init=nn.initializers.orthogonal(0.01))(outs)
        action_logits = 5.0 * jnp.tanh(raw_logits / 5.0)
        action_logits = jnp.nan_to_num(action_logits, nan=0.0, posinf=5.0, neginf=-5.0)
        critic_values = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(outs)
        critic_values = jnp.squeeze(critic_values, axis=-1)
        critic_values = jnp.nan_to_num(critic_values, nan=0.0, posinf=10.0, neginf=-10.0)

        pi = distrax.Categorical(logits=action_logits)
        return new_carry, pi, critic_values


class ActorCriticSdLGADAM(nn.Module):
    rnn_module: Any = None
    action_dim: int = 2
    config: Dict = None

    @nn.compact
    def __call__(self, carry, x):
        obs, dones = x
        emb = RepresentationNetwork(d_model=self.config["RNN"]["D_MODEL"], name="repr_network")(obs)
        emb_b_first = jnp.swapaxes(emb, 0, 1)
        dones_b_first = jnp.swapaxes(dones, 0, 1)

        from src.models.common.sdlg_adam import UnbatchedSdLGADAM
        BatchedSdLGADAM = nn.vmap(
            UnbatchedSdLGADAM, in_axes=(0, 0, 0, None), out_axes=(0, 0, 0),
            variable_axes={'params': None}, split_rngs={'params': False}
        )

        sdlg_adam = BatchedSdLGADAM(
            n_layers=self.config["RNN"].get("N_LAYERS", 3),
            d_model=self.config["RNN"]["D_MODEL"],
            n_heads=self.config["RNN"].get("N_HEADS", 4),
            d_k=self.config["RNN"].get("D_K", 32),
            d_v=self.config["RNN"].get("D_V", 32),
            d_ffc=self.config["RNN"].get("D_FFC", 512),
            window_size=self.config["RNN"].get("WINDOW_SIZE", 16),
            variant="adam"
        )

        outs_b_first, new_carry, _ = sdlg_adam(emb_b_first, dones_b_first, carry, True)
        outs = jnp.swapaxes(outs_b_first, 0, 1)

        raw_logits = nn.Dense(self.action_dim, kernel_init=nn.initializers.orthogonal(0.01))(outs)
        action_logits = 5.0 * jnp.tanh(raw_logits / 5.0)
        action_logits = jnp.nan_to_num(action_logits, nan=0.0, posinf=5.0, neginf=-5.0)
        critic_values = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(outs)
        critic_values = jnp.squeeze(critic_values, axis=-1)
        critic_values = jnp.nan_to_num(critic_values, nan=0.0, posinf=10.0, neginf=-10.0)

        pi = distrax.Categorical(logits=action_logits)
        return new_carry, pi, critic_values


class ActorCriticSdLGADAMMatched(nn.Module):
    """Parameter-matched ADAM control.

    SdLG-ADAM adds n_heads*d_k erase-gate params per layer.
    With defaults (n_heads=4, d_k=32, d_model=64, n_layers=3):
      Extra = d_model*(n_heads*d_k)*n_layers = 64*128*3 = 24,576

    Reduce d_ffc: 512→448 saves 2*64*(512-448)*3 = 24,576 params.
    Target: identical 574,179 params to SdLG-Exact/Delta.
    """
    rnn_module: Any = None
    action_dim: int = 2
    config: Dict = None

    @nn.compact
    def __call__(self, carry, x):
        obs, dones = x
        emb = RepresentationNetwork(d_model=self.config["RNN"]["D_MODEL"], name="repr_network")(obs)
        emb_b_first = jnp.swapaxes(emb, 0, 1)
        dones_b_first = jnp.swapaxes(dones, 0, 1)

        from src.models.common.sdlg_adam import UnbatchedSdLGADAM
        BatchedMatched = nn.vmap(
            UnbatchedSdLGADAM, in_axes=(0, 0, 0, None), out_axes=(0, 0, 0),
            variable_axes={'params': None}, split_rngs={'params': False}
        )

        d_ffc_matched = self.config["RNN"].get("D_FFC", 512) - 64  # 512→448

        sdlg_matched = BatchedMatched(
            n_layers=self.config["RNN"].get("N_LAYERS", 3),
            d_model=self.config["RNN"]["D_MODEL"],
            n_heads=self.config["RNN"].get("N_HEADS", 4),
            d_k=self.config["RNN"].get("D_K", 32),
            d_v=self.config["RNN"].get("D_V", 32),
            d_ffc=d_ffc_matched,
            window_size=self.config["RNN"].get("WINDOW_SIZE", 16),
            variant="adam"
        )

        outs_b_first, new_carry, _ = sdlg_matched(emb_b_first, dones_b_first, carry, True)
        outs = jnp.swapaxes(outs_b_first, 0, 1)

        raw_logits = nn.Dense(self.action_dim, kernel_init=nn.initializers.orthogonal(0.01))(outs)
        action_logits = 5.0 * jnp.tanh(raw_logits / 5.0)
        action_logits = jnp.nan_to_num(action_logits, nan=0.0, posinf=5.0, neginf=-5.0)
        critic_values = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(outs)
        critic_values = jnp.squeeze(critic_values, axis=-1)
        critic_values = jnp.nan_to_num(critic_values, nan=0.0, posinf=10.0, neginf=-10.0)

        pi = distrax.Categorical(logits=action_logits)
        return new_carry, pi, critic_values


class ActorCriticSdLGApproxDelta(nn.Module):
    rnn_module: Any = None
    action_dim: int = 2
    config: Dict = None

    @nn.compact
    def __call__(self, carry, x):
        obs, dones = x
        emb = RepresentationNetwork(d_model=self.config["RNN"]["D_MODEL"], name="repr_network")(obs)
        emb_b_first = jnp.swapaxes(emb, 0, 1)
        dones_b_first = jnp.swapaxes(dones, 0, 1)

        from src.models.common.sdlg_approx_adam import UnbatchedSdLGApproxADAM
        BatchedSdLGApproxDelta = nn.vmap(
            UnbatchedSdLGApproxADAM, in_axes=(0, 0, 0, None), out_axes=(0, 0),
            variable_axes={'params': None}, split_rngs={'params': False}
        )

        sdlg_approx_delta = BatchedSdLGApproxDelta(
            n_layers=self.config["RNN"].get("N_LAYERS", 3),
            d_model=self.config["RNN"]["D_MODEL"],
            d_k=self.config["RNN"].get("D_K", 32),
            d_v=self.config["RNN"].get("D_V", 32),
            d_ffc=self.config["RNN"].get("D_FFC", 512),
            r=self.config["RNN"].get("R", 4),
            window_size=self.config["RNN"].get("WINDOW_SIZE", 16),
            variant="delta"
        )

        outs_b_first, new_carry = sdlg_approx_delta(emb_b_first, dones_b_first, carry, True)
        outs = jnp.swapaxes(outs_b_first, 0, 1)

        raw_logits = nn.Dense(self.action_dim, kernel_init=nn.initializers.orthogonal(0.01))(outs)
        action_logits = 5.0 * jnp.tanh(raw_logits / 5.0)
        action_logits = jnp.nan_to_num(action_logits, nan=0.0, posinf=5.0, neginf=-5.0)
        critic_values = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(outs)
        critic_values = jnp.squeeze(critic_values, axis=-1)
        critic_values = jnp.nan_to_num(critic_values, nan=0.0, posinf=10.0, neginf=-10.0)

        pi = distrax.Categorical(logits=action_logits)
        return new_carry, pi, critic_values


class ActorCriticSdLGApproxADAM(nn.Module):
    rnn_module: Any = None
    action_dim: int = 2
    config: Dict = None

    @nn.compact
    def __call__(self, carry, x):
        obs, dones = x
        emb = RepresentationNetwork(d_model=self.config["RNN"]["D_MODEL"], name="repr_network")(obs)
        emb_b_first = jnp.swapaxes(emb, 0, 1)
        dones_b_first = jnp.swapaxes(dones, 0, 1)

        from src.models.common.sdlg_approx_adam import UnbatchedSdLGApproxADAM
        BatchedSdLGApproxADAM = nn.vmap(
            UnbatchedSdLGApproxADAM, in_axes=(0, 0, 0, None), out_axes=(0, 0),
            variable_axes={'params': None}, split_rngs={'params': False}
        )

        sdlg_approx_adam = BatchedSdLGApproxADAM(
            n_layers=self.config["RNN"].get("N_LAYERS", 3),
            d_model=self.config["RNN"]["D_MODEL"],
            d_k=self.config["RNN"].get("D_K", 32),
            d_v=self.config["RNN"].get("D_V", 32),
            d_ffc=self.config["RNN"].get("D_FFC", 512),
            r=self.config["RNN"].get("R", 4),
            window_size=self.config["RNN"].get("WINDOW_SIZE", 16),
            variant="adam"
        )

        outs_b_first, new_carry = sdlg_approx_adam(emb_b_first, dones_b_first, carry, True)
        outs = jnp.swapaxes(outs_b_first, 0, 1)

        raw_logits = nn.Dense(self.action_dim, kernel_init=nn.initializers.orthogonal(0.01))(outs)
        action_logits = 5.0 * jnp.tanh(raw_logits / 5.0)
        action_logits = jnp.nan_to_num(action_logits, nan=0.0, posinf=5.0, neginf=-5.0)
        critic_values = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(outs)
        critic_values = jnp.squeeze(critic_values, axis=-1)
        critic_values = jnp.nan_to_num(critic_values, nan=0.0, posinf=10.0, neginf=-10.0)

        pi = distrax.Categorical(logits=action_logits)
        return new_carry, pi, critic_values


class ActorCriticGRU(nn.Module):
    rnn_module: Any = None
    action_dim: int = 2
    config: Dict = None

    @nn.compact
    def __call__(self, carry, x):
        obs, dones = x
        emb = RepresentationNetwork(d_model=self.config["RNN"]["D_MODEL"], name="repr_network")(obs)
        from src.models.common.gru import BatchedGRU
        new_carry, outs = BatchedGRU()(carry, (emb, dones))

        raw_logits = nn.Dense(self.action_dim, kernel_init=nn.initializers.orthogonal(0.01))(outs)
        action_logits = 5.0 * jnp.tanh(raw_logits / 5.0)
        action_logits = jnp.nan_to_num(action_logits, nan=0.0, posinf=5.0, neginf=-5.0)
        critic_values = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(outs)
        critic_values = jnp.squeeze(critic_values, axis=-1)
        critic_values = jnp.nan_to_num(critic_values, nan=0.0, posinf=10.0, neginf=-10.0)

        pi = distrax.Categorical(logits=action_logits)
        return new_carry, pi, critic_values


class BatchedMamba(nn.Module):
    d_model: int

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        emb_t, done_t = x
        carry = jax.tree_util.tree_map(lambda leaf: leaf * (1.0 - done_t[:, None, None]), carry)
        from src.models.baselines.mamba import MambaCell
        return MambaCell(d_model=self.d_model)(emb_t, carry)


class ActorCriticMamba(nn.Module):
    rnn_module: Any = None
    action_dim: int = 2
    config: Dict = None

    @nn.compact
    def __call__(self, carry, x):
        obs, dones = x
        emb = RepresentationNetwork(d_model=self.config["RNN"]["D_MODEL"], name="repr_network")(obs)
        new_carry, outs = BatchedMamba(d_model=self.config["RNN"]["D_MODEL"])(carry, (emb, dones))

        raw_logits = nn.Dense(self.action_dim, kernel_init=nn.initializers.orthogonal(0.01))(outs)
        action_logits = 5.0 * jnp.tanh(raw_logits / 5.0)
        action_logits = jnp.nan_to_num(action_logits, nan=0.0, posinf=5.0, neginf=-5.0)
        critic_values = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(outs)
        critic_values = jnp.squeeze(critic_values, axis=-1)
        critic_values = jnp.nan_to_num(critic_values, nan=0.0, posinf=10.0, neginf=-10.0)

        pi = distrax.Categorical(logits=action_logits)
        return new_carry, pi, critic_values


class ActorCriticGLA(nn.Module):
    rnn_module: Any = None
    action_dim: int = 2
    config: Dict = None

    @nn.compact
    def __call__(self, carry, x):
        obs, dones = x
        emb = RepresentationNetwork(d_model=self.config["RNN"]["D_MODEL"], name="repr_network")(obs)
        emb_b_first = jnp.swapaxes(emb, 0, 1)
        dones_b_first = jnp.swapaxes(dones, 0, 1)

        from src.models.common.exact_sdlg import ExactMultiScaleGLA

        class UnbatchedGLA(nn.Module):
            config: dict

            @nn.compact
            def __call__(self, inputs, terminations, carry, deterministic=True):
                x = inputs
                new_carry = []
                for l in range(self.config["RNN"].get("N_LAYERS", 3)):
                    norm_gla = nn.LayerNorm(epsilon=1e-5)(x)
                    gla_out, next_c, _ = ExactMultiScaleGLA(
                        d_model=self.config["RNN"]["D_MODEL"],
                        n_heads=self.config["RNN"].get("N_HEADS", 4),
                        d_k=32, d_v=32
                    )(norm_gla, terminations, carry[l])
                    x = x + gla_out
                    new_carry.append(next_c)
                return x, tuple(new_carry)

        BatchedGLA = nn.vmap(UnbatchedGLA, in_axes=(0, 0, 0, None), out_axes=(0, 0),
                             variable_axes={'params': None}, split_rngs={'params': False})

        outs_b_first, new_carry = BatchedGLA(self.config)(emb_b_first, dones_b_first, carry, True)
        outs = jnp.swapaxes(outs_b_first, 0, 1)

        raw_logits = nn.Dense(self.action_dim, kernel_init=nn.initializers.orthogonal(0.01))(outs)
        action_logits = 5.0 * jnp.tanh(raw_logits / 5.0)
        action_logits = jnp.nan_to_num(action_logits, nan=0.0, posinf=5.0, neginf=-5.0)
        critic_values = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(outs)
        critic_values = jnp.squeeze(critic_values, axis=-1)
        critic_values = jnp.nan_to_num(critic_values, nan=0.0, posinf=10.0, neginf=-10.0)

        pi = distrax.Categorical(logits=action_logits)
        return new_carry, pi, critic_values


class ActorCriticDeltaGLA(nn.Module):
    rnn_module: Any = None
    action_dim: int = 2
    config: Dict = None

    @nn.compact
    def __call__(self, carry, x):
        obs, dones = x
        emb = RepresentationNetwork(d_model=self.config["RNN"]["D_MODEL"], name="repr_network")(obs)
        emb_b_first = jnp.swapaxes(emb, 0, 1)
        dones_b_first = jnp.swapaxes(dones, 0, 1)

        from src.models.common.adam_gla import ADAMMultiScaleGLA

        class UnbatchedDeltaGLA(nn.Module):
            config: dict

            @nn.compact
            def __call__(self, inputs, terminations, carry, deterministic=True):
                x = inputs
                new_carry = []
                for l in range(self.config["RNN"].get("N_LAYERS", 3)):
                    norm_gla = nn.LayerNorm(epsilon=1e-5)(x)
                    gla_out, next_c, _ = ADAMMultiScaleGLA(
                        d_model=self.config["RNN"]["D_MODEL"],
                        n_heads=self.config["RNN"].get("N_HEADS", 4),
                        d_k=32, d_v=32, variant="delta"
                    )(norm_gla, terminations, carry[l])
                    x = x + gla_out
                    new_carry.append(next_c)
                return x, tuple(new_carry)

        BatchedDeltaGLA = nn.vmap(UnbatchedDeltaGLA, in_axes=(0, 0, 0, None), out_axes=(0, 0),
                                   variable_axes={'params': None}, split_rngs={'params': False})

        outs_b_first, new_carry = BatchedDeltaGLA(self.config)(emb_b_first, dones_b_first, carry, True)
        outs = jnp.swapaxes(outs_b_first, 0, 1)

        raw_logits = nn.Dense(self.action_dim, kernel_init=nn.initializers.orthogonal(0.01))(outs)
        action_logits = 5.0 * jnp.tanh(raw_logits / 5.0)
        action_logits = jnp.nan_to_num(action_logits, nan=0.0, posinf=5.0, neginf=-5.0)
        critic_values = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(outs)
        critic_values = jnp.squeeze(critic_values, axis=-1)
        critic_values = jnp.nan_to_num(critic_values, nan=0.0, posinf=10.0, neginf=-10.0)

        pi = distrax.Categorical(logits=action_logits)
        return new_carry, pi, critic_values


class ActorCriticADAM(nn.Module):
    rnn_module: Any = None
    action_dim: int = 2
    config: Dict = None

    @nn.compact
    def __call__(self, carry, x):
        obs, dones = x
        emb = RepresentationNetwork(d_model=self.config["RNN"]["D_MODEL"], name="repr_network")(obs)
        emb_b_first = jnp.swapaxes(emb, 0, 1)
        dones_b_first = jnp.swapaxes(dones, 0, 1)

        from src.models.common.adam_gla import ADAMMultiScaleGLA

        class UnbatchedADAM(nn.Module):
            config: dict

            @nn.compact
            def __call__(self, inputs, terminations, carry, deterministic=True):
                x = inputs
                new_carry = []
                for l in range(self.config["RNN"].get("N_LAYERS", 3)):
                    norm_gla = nn.LayerNorm(epsilon=1e-5)(x)
                    gla_out, next_c, _ = ADAMMultiScaleGLA(
                        d_model=self.config["RNN"]["D_MODEL"],
                        n_heads=self.config["RNN"].get("N_HEADS", 4),
                        d_k=32, d_v=32, variant="adam"
                    )(norm_gla, terminations, carry[l])
                    x = x + gla_out
                    new_carry.append(next_c)
                return x, tuple(new_carry)

        BatchedADAM = nn.vmap(UnbatchedADAM, in_axes=(0, 0, 0, None), out_axes=(0, 0),
                              variable_axes={'params': None}, split_rngs={'params': False})

        outs_b_first, new_carry = BatchedADAM(self.config)(emb_b_first, dones_b_first, carry, True)
        outs = jnp.swapaxes(outs_b_first, 0, 1)

        raw_logits = nn.Dense(self.action_dim, kernel_init=nn.initializers.orthogonal(0.01))(outs)
        action_logits = 5.0 * jnp.tanh(raw_logits / 5.0)
        action_logits = jnp.nan_to_num(action_logits, nan=0.0, posinf=5.0, neginf=-5.0)
        critic_values = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(outs)
        critic_values = jnp.squeeze(critic_values, axis=-1)
        critic_values = jnp.nan_to_num(critic_values, nan=0.0, posinf=10.0, neginf=-10.0)

        pi = distrax.Categorical(logits=action_logits)
        return new_carry, pi, critic_values