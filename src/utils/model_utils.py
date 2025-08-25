import json
from typing import Any, Dict, Tuple

import aiohttp
from huggingface_hub import get_safetensors_metadata

from src.config.settings import DEFAULT_CONFIG_FILENAME, HUGGINGFACE_TOKEN


async def download_config(
    model_id: str, hf_token: str = None, filename: str = DEFAULT_CONFIG_FILENAME
) -> Dict[str, Any]:
    """
    Asynchronously download model configuration from HuggingFace.

    Args:
        model_id: HuggingFace model ID
        hf_token: HuggingFace API token (optional, uses default if not provided)
        filename: Configuration filename

    Returns:
        Model configuration dictionary
    """
    url = f"https://huggingface.co/{model_id}/raw/main/{filename}"
    headers = {
        "Authorization": f"Bearer {hf_token or HUGGINGFACE_TOKEN}",
        "Accept": "application/json",
    }

    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, headers=headers) as response:
                if response.status == 404:
                    raise Exception(f"Model or file not found: {model_id}/{filename}")
                elif response.status == 401:
                    raise Exception(
                        "Unauthorized: Please check your Hugging Face token"
                    )
                elif response.status != 200:
                    raise Exception(
                        f"Error downloading {filename} from model {model_id}. HTTP {response.status}"
                    )

                # Try to parse as JSON first
                try:
                    return await response.json()
                except Exception as _:
                    # If JSON parsing fails, try to parse the text content
                    text_content = await response.text()
                    try:
                        return json.loads(text_content)
                    except Exception as _:
                        raise Exception(
                            f"Failed to parse response as JSON. Response content: {text_content[:200]}..."
                        )
        except aiohttp.ClientError as e:
            raise Exception(f"Network error while accessing Hugging Face: {str(e)}")


# llm-mem's byte mapping
BYTES_PER_DTYPE = {
    "int4": 0.5,
    "int8": 1,
    "float8": 1,
    "float16": 2,
    "float32": 4,
    "bfloat16": 2,
}


def estimate_parameters(model_id: str, hf_token: str) -> int:
    """Get actual parameter count from safetensors metadata (llm-mem approach)"""
    try:
        metadata = get_safetensors_metadata(model_id, token=hf_token)
        if not metadata or not metadata.parameter_count:
            print("Could not fetch metadata")
            return 0

        # Sum all parameter counts (handles multi-file models)
        total_params = sum(int(count) for count in metadata.parameter_count.values())
        return total_params
    except Exception as e:
        print(f"Error getting safetensors metadata: {e}")
        return 0


def estimate_vram(parameters_billions: float, dtype: str) -> float:
    """Apply llm-mem's exact formula for model memory"""
    bytes_val = BYTES_PER_DTYPE[dtype]
    # llm-mem's formula: (params * 4) / (32/(bytes*8)) * 1.18
    # return round((parameters_billions * 4) / (32 / (bytes_val * 8)) * 1.18, 2)
    MAGIC_KNOB_FOR_ACTIVATION_OVERHEAD = 1.21
    return round(
        (parameters_billions * 4) / (32 / (bytes_val * 8))
        + MAGIC_KNOB_FOR_ACTIVATION_OVERHEAD,
        2,
    )


# def estimate_parameters(config: Dict[str, Any]) -> int:
#     """
#     Estimate the total number of parameters of a Transformer model based on its config.

#     Args:
#         config: Model configuration dictionary

#     Returns:
#         Total number of parameters
#     """
#     vocab_size = config.get("vocab_size")
#     hidden_size = config.get("hidden_size")
#     num_hidden_layers = config.get("num_hidden_layers")
#     num_attention_heads = config.get("num_attention_heads")
#     head_dim = config.get("head_dim")
#     if head_dim is None:
#         head_dim = hidden_size // num_attention_heads
#     num_key_value_heads = config.get("num_key_value_heads")
#     intermediate_size = config.get("intermediate_size")

#     if None in [vocab_size, hidden_size, num_hidden_layers, num_attention_heads, num_key_value_heads, intermediate_size]:
#         raise ValueError("The config is missing one or more required parameters.")

#     embed_params = vocab_size * hidden_size
#     q_params = hidden_size * (num_attention_heads * head_dim)
#     k_params = hidden_size * (num_key_value_heads * head_dim)
#     v_params = hidden_size * (num_key_value_heads * head_dim)
#     o_params = (num_attention_heads * head_dim) * hidden_size
#     attn_params = q_params + k_params + v_params + o_params
#     mlp_params = 2 * hidden_size * intermediate_size
#     layer_params = attn_params + mlp_params
#     total_params = embed_params + num_hidden_layers * layer_params
#     return total_params

# def estimate_vram(total_params: int, q_bits: int) -> float:
#     """
#     Estimate VRAM requirements for a model.

#     Args:
#         total_params: Total number of model parameters
#         q_bits: Quantization bits (e.g., 16 for FP16)

#     Returns:
#         Estimated VRAM requirement in GB
#     """
#     P_in_billions = total_params / 1e9
#     return P_in_billions * 4 * (q_bits / 32) * 1.2


# [HETARTH] TODO: This is a hack to account for the activation overhead. THIS WILL FUCK US UP SOMEDAY - SO PROFILE IT!!
def estimate_layer_vram(
    config: Dict[str, Any],
    q_bits: int,
    batch_size: int = 1,
    seq_length: int = 2048,
    include_kv_cache: bool = False,
) -> Tuple[float, float, float]:
    """
    Estimate VRAM requirements per layer and for non-layer components (weights + activations).

    Notes:
        - This function estimates both weights memory and activation memory.
        - Embeddings include the input embedding matrix plus LM head if weights are not tied.
        - Per-layer parameters include attention, 3-projection MLP (gate/up/down), and layer norms.
        - Final output norm is included in the total but not in per-layer or embedding buckets.
        - Activation memory is calculated based on batch size and sequence length.

    Args:
        config: Model configuration dictionary
        q_bits: Quantization bits (e.g., 16 for FP16)
        batch_size: Batch size for activation calculation (default: 1)
        seq_length: Sequence length for activation calculation (default: 2048)
        include_kv_cache: Whether to include KV cache (not implemented yet, default: False)

    Returns:
        Tuple of (vram_per_layer_gb, embedding_vram_gb, total_vram_gb)
    """
    # Get model parameters
    vocab_size = config.get("vocab_size")
    hidden_size = config.get("hidden_size")
    num_hidden_layers = config.get("num_hidden_layers")
    num_attention_heads = config.get("num_attention_heads")
    head_dim = config.get("head_dim")
    if head_dim is None and hidden_size is not None and num_attention_heads:
        head_dim = hidden_size // num_attention_heads
    num_key_value_heads = config.get("num_key_value_heads")
    if num_key_value_heads is None:
        num_key_value_heads = num_attention_heads
    intermediate_size = config.get("intermediate_size")
    tie_word_embeddings = bool(config.get("tie_word_embeddings", False))

    required = [
        vocab_size,
        hidden_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        intermediate_size,
        head_dim,
    ]
    # print("Required parameters:", required)
    if any(x is None for x in required):
        raise ValueError("The config is missing one or more required parameters.")

    # Embeddings (input) + LM head if not tied
    embed_params = vocab_size * hidden_size
    lm_head_params = 0 if tie_word_embeddings else vocab_size * hidden_size
    embedding_params_total = embed_params + lm_head_params

    # Parameters per transformer layer
    # Attention projections
    q_params = hidden_size * (num_attention_heads * head_dim)
    k_params = hidden_size * (num_key_value_heads * head_dim)
    v_params = hidden_size * (num_key_value_heads * head_dim)
    o_params = (num_attention_heads * head_dim) * hidden_size
    attn_params = q_params + k_params + v_params + o_params

    # MLP: gate, up, down (3 projections for most modern LLMs, e.g., SwiGLU)
    mlp_params = 3 * hidden_size * intermediate_size

    # Layer norms per layer (pre-attn and pre-mlp)
    layer_norm_params = 2 * hidden_size

    layer_params = attn_params + mlp_params + layer_norm_params

    # Final output norm (applied once at the end of the stack)
    final_norm_params = hidden_size

    # Convert to VRAM (GB) for WEIGHTS
    bytes_per_param = q_bits / 8
    bytes_to_gb = 1024**3

    # Calculate WEIGHTS memory
    embedding_vram_gb = (embedding_params_total * bytes_per_param) / bytes_to_gb
    layer_weights_vram_gb = (layer_params * bytes_per_param) / bytes_to_gb

    # Calculate ACTIVATION memory per layer
    # Based on article formula: batch_size * seq_length * hidden_dim * K
    # K is a heuristic factor (using conservative estimate of 12)
    # This represents intermediate activations in attention and MLP blocks
    K_FACTOR = 12  # Conservative heuristic for activation multiplier

    # Activation memory per layer (in bytes)
    activation_bytes_per_layer = (
        batch_size * seq_length * hidden_size * K_FACTOR * (q_bits / 8)
    )
    activation_vram_per_layer_gb = activation_bytes_per_layer / bytes_to_gb

    # Total per-layer VRAM (weights + activations)
    vram_per_layer_gb = layer_weights_vram_gb + activation_vram_per_layer_gb

    # Total model VRAM
    total_vram_gb = (
        embedding_vram_gb
        + (num_hidden_layers * vram_per_layer_gb)
        + (final_norm_params * bytes_per_param) / bytes_to_gb
    )

    return (
        round(vram_per_layer_gb, 6),
        round(embedding_vram_gb, 6),
        round(total_vram_gb, 6),
    )


def calculate_max_layers_for_peer(
    config: Dict[str, Any],
    available_vram_gb: float,
    q_bits: int,
    safety_margin: float = 0.1,
    batch_size: int = 1,
    seq_length: int = 2048,
) -> Dict[str, Any]:
    """
    Calculate the maximum number of layers that can fit in a peer's available VRAM.

    Args:
        config: Model configuration dictionary
        available_vram_gb: Available VRAM in GB for this peer
        q_bits: Quantization bits (e.g., 16 for FP16)
        safety_margin: Safety margin as fraction of available VRAM (default: 10%)
        batch_size: Batch size for activation calculation (default: 1)
        seq_length: Sequence length for activation calculation (default: 2048)

    Returns:
        Dictionary containing layer calculation details
    """
    # Get VRAM requirements with activation overhead
    vram_per_layer_gb, embedding_vram_gb, total_model_vram_gb = estimate_layer_vram(
        config, q_bits, batch_size, seq_length, include_kv_cache=False
    )

    # Apply safety margin
    usable_vram_gb = available_vram_gb * (1 - safety_margin)

    # Calculate max layers (assuming embedding layers are handled separately)
    max_layers = int(usable_vram_gb / vram_per_layer_gb)

    # If this peer needs to handle embeddings too, subtract that from available space
    max_layers_with_embeddings = int(
        (usable_vram_gb - embedding_vram_gb) / vram_per_layer_gb
    )
    max_layers_with_embeddings = max(0, max_layers_with_embeddings)

    # Calculate actual VRAM usage
    vram_used_layers_only = max_layers * vram_per_layer_gb
    vram_used_with_embeddings = embedding_vram_gb + (
        max_layers_with_embeddings * vram_per_layer_gb
    )

    return {
        "max_layers_only": max_layers,
        "max_layers_with_embeddings": max_layers_with_embeddings,
        "vram_per_layer_gb": round(vram_per_layer_gb, 3),
        "embedding_vram_gb": round(embedding_vram_gb, 3),
        "available_vram_gb": available_vram_gb,
        "usable_vram_gb": round(usable_vram_gb, 3),
        "safety_margin": safety_margin,
        "vram_used_layers_only": round(vram_used_layers_only, 3),
        "vram_used_with_embeddings": round(vram_used_with_embeddings, 3),
        "total_model_layers": config.get("num_hidden_layers"),
        "total_model_vram_gb": round(total_model_vram_gb, 3),
        "batch_size": batch_size,
        "seq_length": seq_length,
    }


def distribute_layers_across_peers(
    config: Dict[str, Any],
    peers_vram: Dict[str, float],  # peer_id -> available_vram_gb
    q_bits: int,
    safety_margin: float = 0.1,
    batch_size: int = 1,
    seq_length: int = 2048,
) -> Dict[str, Any]:
    """
    Distribute model layers optimally across multiple GPU peers.

    Args:
        config: Model configuration dictionary
        peers_vram: Dictionary mapping peer_id to available VRAM in GB
        q_bits: Quantization bits
        safety_margin: Safety margin as fraction of available VRAM
        batch_size: Batch size for activation calculation (default: 1)
        seq_length: Sequence length for activation calculation (default: 2048)

    Returns:
        Dictionary containing the distribution plan
    """
    total_layers = config.get("num_hidden_layers")
    vram_per_layer_gb, embedding_vram_gb, total_model_vram_gb = estimate_layer_vram(
        config, q_bits, batch_size, seq_length, include_kv_cache=False
    )
    print(
        f"🔍 Model {config.get('model_name')} has {total_layers} layers and requires {total_model_vram_gb} GB of VRAM"
    )

    # Sort peers by available VRAM (descending)
    sorted_peers = sorted(peers_vram.items(), key=lambda x: x[1], reverse=True)

    distribution = {}
    remaining_layers = total_layers
    total_available_vram = sum(peers_vram.values())
    embedding_assigned = False

    for i, (peer_id, available_vram) in enumerate(sorted_peers):
        if remaining_layers <= 0:
            break

        # HACK: For the peer with highest VRAM (first in sorted list), simulate only 4GB
        # if i == 0:
        #     print(f"🔧 [HACK] Simulating 4GB VRAM for highest VRAM peer {peer_id} (actual: {available_vram}GB)")
        #     available_vram = 0.25

        # Calculate max layers for this peer
        peer_calculation = calculate_max_layers_for_peer(
            config, available_vram, q_bits, safety_margin, batch_size, seq_length
        )

        # Determine how many layers to assign
        if not embedding_assigned and available_vram >= embedding_vram_gb:
            # First peer with enough space gets embeddings + layers
            max_layers_for_peer = peer_calculation["max_layers_with_embeddings"]
            will_handle_embeddings = True
            embedding_assigned = True
        else:
            # Other peers get only layers
            max_layers_for_peer = peer_calculation["max_layers_only"]
            will_handle_embeddings = False

        # Don't assign more layers than remaining
        assigned_layers = min(max_layers_for_peer, remaining_layers)

        if assigned_layers > 0:
            distribution[peer_id] = {
                "assigned_layers": assigned_layers,
                "handles_embeddings": will_handle_embeddings,
                "available_vram_gb": available_vram,
                "estimated_vram_usage": round(
                    (embedding_vram_gb if will_handle_embeddings else 0)
                    + (assigned_layers * vram_per_layer_gb),
                    3,
                ),
                "vram_utilization_percent": round(
                    (
                        (embedding_vram_gb if will_handle_embeddings else 0)
                        + (assigned_layers * vram_per_layer_gb)
                    )
                    / available_vram
                    * 100,
                    1,
                ),
            }
            remaining_layers -= assigned_layers

    # Check if we can fit the entire model
    total_assigned_layers = sum(
        peer["assigned_layers"] for peer in distribution.values()
    )
    can_fit_model = total_assigned_layers >= total_layers and embedding_assigned

    return {
        "distribution": distribution,
        "model_info": {
            "total_layers": total_layers,
            "total_assigned_layers": total_assigned_layers,
            "vram_per_layer_gb": round(vram_per_layer_gb, 3),
            "embedding_vram_gb": round(embedding_vram_gb, 3),
            "total_model_vram_gb": round(total_model_vram_gb, 3),
            "batch_size": batch_size,
            "seq_length": seq_length,
        },
        "can_fit_model": can_fit_model,
        "remaining_layers": remaining_layers,
        "total_peers": len(peers_vram),
        "utilized_peers": len(distribution),
        "total_available_vram_gb": round(total_available_vram, 3),
    }
