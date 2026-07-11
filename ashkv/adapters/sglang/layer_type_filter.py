"""Layer Type Filter for SGLang Hybrid Models.

Identifies which layers in a hybrid model (e.g., Jamba, DeltaNet) are standard
Attention layers that can be compressed by ASH-KV. Non-attention layers
are skipped during INT8 Shadow Cache operations.
"""
from typing import List, Any


def get_compressible_layers(model_config: Any) -> List[int]:
    """Parse the model config and return a list of compressible layer indices.
    
    Args:
        model_config: A transformers.PretrainedConfig or SGLang ModelConfig object.
        
    Returns:
        List of integers representing the layer indices that use Attention.
    """
    num_hidden_layers = getattr(model_config, "num_hidden_layers", getattr(model_config, "n_layer", 0))
    if num_hidden_layers == 0:
        # Fallback if config is wrapped
        hf_config = getattr(model_config, "hf_config", None)
        if hf_config:
            num_hidden_layers = getattr(hf_config, "num_hidden_layers", getattr(hf_config, "n_layer", 0))
            
    if num_hidden_layers == 0:
        raise ValueError("Could not determine number of layers from config.")

    compressible_layers = []
    
    # 1. Check for explicit layer types list (like in Jamba)
    layer_types = getattr(model_config, "layers_block_type", None)
    if layer_types is not None:
        for idx, l_type in enumerate(layer_types):
            if "attention" in str(l_type).lower():
                compressible_layers.append(idx)
        return compressible_layers

    # 2. Check if the model is purely attention
    is_hybrid = getattr(model_config, "is_hybrid", False)
    has_mamba = hasattr(model_config, "mamba_config") or "mamba" in str(type(model_config)).lower()
    has_deltanet = hasattr(model_config, "deltanet_config") or "deltanet" in str(type(model_config)).lower()
    
    if not is_hybrid and not has_mamba and not has_deltanet:
        return list(range(num_hidden_layers))

    # 3. Handle specific hybrid models with stride patterns
    attn_layer_offset = getattr(model_config, "attn_layer_offset", 0)
    attn_layer_period = getattr(model_config, "attn_layer_period", 0)
    
    if attn_layer_period > 0:
        for i in range(num_hidden_layers):
            if (i - attn_layer_offset) % attn_layer_period == 0:
                compressible_layers.append(i)
        return compressible_layers

    # 4. Fallback
    return list(range(num_hidden_layers))
