"""
Unified Model Instantiation Factory
Strictly abstracts the instantiations for PPO baselines (LSTM, GRU, GTrXL) and Exact-SdLG,
enforcing shared hyperparameter parsing.
"""

def make_model(model_name: str, config: dict):
    """
    Instantiates a JAX/Flax module given its name and configuration.

    Args:
        model_name: String identifier for the architecture (e.g., 'Exact_SdLG_Dense', 'PPO-LSTM').
        config: A dictionary of network configuration hyperparameters.

    Returns:
        An instantiated Flax nn.Module.
    """
    model_name_lower = model_name.lower()

    if "exact_sdlg_dense" in model_name_lower:
        from src.models.architectures.sdlg import Exact_SdLG_Dense
        return Exact_SdLG_Dense(config)
    elif "exact_sdlg_sparse_moe" in model_name_lower:
        from src.models.architectures.sdlg import Exact_SdLG_Sparse_MoE
        return Exact_SdLG_Sparse_MoE(config)
    elif "lstm" in model_name_lower:
        # Note: Placeholder for explicit LSTM implementation (Stage 2/3 alignment)
        from src.models.baselines.lstm import LSTM_Module
        return LSTM_Module(config)
    elif "gru" in model_name_lower:
        # Note: Placeholder for explicit GRU implementation
        from src.models.baselines.gru import GRU_Module
        return GRU_Module(config)
    elif "gtrxl" in model_name_lower:
        # Note: Placeholder for GTrXL
        from src.models.baselines.gtrxl import GTrXL_Module
        return GTrXL_Module(config)
    elif "s5" in model_name_lower:
        from src.models.baselines.s5 import S5_Module
        return S5_Module(config)
    elif "markovian" in model_name_lower or "amnesia" in model_name_lower:
        from src.models.baselines.markovian import Markovian_Module
        return Markovian_Module(config)
    else:
        raise ValueError(f"Unknown architecture identifier: {model_name}")
