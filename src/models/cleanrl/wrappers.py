import jax
import jax.numpy as jnp
from flax import linen as nn
import distrax
from typing import Tuple, Any, Callable
from src.models.common.hybrid import BatchedHybrid, HybridHiddenState
from src.models.common.agalite import BatchedAGaLiTe, AGaLiTeHiddenState
from src.models.common.gtrxl import BatchedGTrXL, GTrXLHiddenState


class CleanRLActorCritic(nn.Module):
    """CleanRL compatible joint Actor-Critic wrapper."""
    action_dim: int
    rnn_module: nn.Module

    @nn.compact
    def __call__(self, carry: Any, x: Tuple[jnp.ndarray, jnp.ndarray]) -> Tuple[Any, Any, Any]:
        """
        x is typically (obs, done)
        Returns: carry, pi (distrax distribution), value
        """
        obs, done = x
        carry, hidden = self.rnn_module(carry, (obs, done))

        # Actor
        actor_logits = nn.Dense(self.action_dim)(hidden)
        # Clip logits to prevent softmax probability from exactly reaching zero, avoiding NaN entropies.
        actor_logits = 20.0 * nn.tanh(actor_logits)
        
        pi = distrax.Categorical(logits=actor_logits)

        # Critic
        value = nn.Dense(1)(hidden)

        return carry, pi, jnp.squeeze(value, axis=-1)

def get_cleanrl_model(rnn_type: str, action_dim: int, **kwargs) -> nn.Module:
    if rnn_type == "hybrid":
        rnn = BatchedHybrid(**kwargs)
    elif rnn_type == "agalite":
        rnn = BatchedAGaLiTe(**kwargs)
    elif rnn_type == "gtrxl":
        rnn = BatchedGTrXL(**kwargs)
    else:
        raise ValueError(f"Unknown RNN type: {rnn_type}")

    return CleanRLActorCritic(action_dim=action_dim, rnn_module=rnn)
