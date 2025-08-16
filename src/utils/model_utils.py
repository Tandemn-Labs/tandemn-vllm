import json
import aiohttp
from typing import Dict, Any, Tuple
from src.config.settings import HUGGINGFACE_TOKEN, DEFAULT_CONFIG_FILENAME
from huggingface_hub import get_safetensors_metadata

async def download_config(model_id: str, hf_token: str = None, filename: str = DEFAULT_CONFIG_FILENAME) -> Dict[str, Any]:
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
        "Accept": "application/json"
    }
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, headers=headers) as response:
                if response.status == 404:
                    raise Exception(f"Model or file not found: {model_id}/{filename}")
                elif response.status == 401:
                    raise Exception("Unauthorized: Please check your Hugging Face token")
                elif response.status != 200:
                    raise Exception(f"Error downloading {filename} from model {model_id}. HTTP {response.status}")
                
                # Try to parse as JSON first
                try:
                    return await response.json()
                except:
                    # If JSON parsing fails, try to parse the text content
                    text_content = await response.text()
                    try:
                        return json.loads(text_content)
                    except:
                        raise Exception(f"Failed to parse response as JSON. Response content: {text_content[:200]}...")
        except aiohttp.ClientError as e:
            raise Exception(f"Network error while accessing Hugging Face: {str(e)}")

# llm-mem's byte mapping
BYTES_PER_DTYPE = {
        "int4": 0.5,
        "int8": 1,
        "float8": 1,
        "float16": 2,
        "float32": 4,
        "bfloat16": 2}

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
    return round((parameters_billions * 4) / (32 / (bytes_val * 8)) + MAGIC_KNOB_FOR_ACTIVATION_OVERHEAD, 2)


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

def estimate_layer_vram(config: Dict[str, Any], q_bits: int) -> Tuple[float, float, float]:
    """
    Estimate VRAM requirements per layer and for non-layer components (weights only).

    Notes:
        - This function estimates WEIGHTS memory only (no activations / KV cache / framework overhead),
          to align with vLLM's "model weights take X GiB" measurement.
        - Embeddings include the input embedding matrix plus LM head if weights are not tied.
        - Per-layer parameters include attention, 3-projection MLP (gate/up/down), and layer norms.
        - Final output norm is included in the total but not in per-layer or embedding buckets.

    Args:
        config: Model configuration dictionary
        q_bits: Quantization bits (e.g., 16 for FP16)

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


    required = [vocab_size, hidden_size, num_hidden_layers, num_attention_heads,
                num_key_value_heads, intermediate_size, head_dim]
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

    # Convert to VRAM (GB) for WEIGHTS only
    bytes_per_param = q_bits / 8
    bytes_to_gb = (1024 ** 3)

    # [HETARTH] TODO: This is a hack to account for the activation overhead. THIS WILL FUCK US UP SOMEDAY - SO PROFILE IT!!
    MAGIC_KNOB_FOR_ACTIVATION_OVERHEAD = 1.2 # this is the activation overhead for the ENTIRE MODEL WHEN ITS LOADED
    # So the activation overhead for each layer is MAGIC_KNOB_FOR_ACTIVATION_OVERHEAD / num_hidden_layers
    per_layer_activation_overhead = MAGIC_KNOB_FOR_ACTIVATION_OVERHEAD / num_hidden_layers # this is the activation overhead for each layer

    embedding_vram_gb = (embedding_params_total * bytes_per_param) / bytes_to_gb
    vram_per_layer_gb = ((layer_params * bytes_per_param) / bytes_to_gb) + per_layer_activation_overhead  # make sure you profile the activation overhead
    total_vram_gb = (
        embedding_vram_gb
        + (num_hidden_layers * vram_per_layer_gb)
        + (final_norm_params * bytes_per_param) / bytes_to_gb
    )

    return round(vram_per_layer_gb, 6), round(embedding_vram_gb, 6), round(total_vram_gb, 6)

def calculate_max_layers_for_peer(
    config: Dict[str, Any], 
    available_vram_gb: float, 
    q_bits: int,
    safety_margin: float = 0.1
) -> Dict[str, Any]:
    """
    Calculate the maximum number of layers that can fit in a peer's available VRAM.
    
    Args:
        config: Model configuration dictionary
        available_vram_gb: Available VRAM in GB for this peer
        q_bits: Quantization bits (e.g., 16 for FP16)
        safety_margin: Safety margin as fraction of available VRAM (default: 10%)
        
    Returns:
        Dictionary containing layer calculation details
    """
    # Get VRAM requirements
    vram_per_layer_gb, embedding_vram_gb, total_model_vram_gb = estimate_layer_vram(config, q_bits)
    
    # Apply safety margin
    usable_vram_gb = available_vram_gb * (1 - safety_margin)
    
    # Calculate max layers (assuming embedding layers are handled separately)
    max_layers = int(usable_vram_gb / vram_per_layer_gb)
    
    # If this peer needs to handle embeddings too, subtract that from available space
    max_layers_with_embeddings = int((usable_vram_gb - embedding_vram_gb) / vram_per_layer_gb)
    max_layers_with_embeddings = max(0, max_layers_with_embeddings)
    
    # Calculate actual VRAM usage
    vram_used_layers_only = max_layers * vram_per_layer_gb
    vram_used_with_embeddings = embedding_vram_gb + (max_layers_with_embeddings * vram_per_layer_gb)
    
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
        "total_model_vram_gb": round(total_model_vram_gb, 3)
    }

def distribute_layers_across_peers(
    config: Dict[str, Any],
    peers_vram: Dict[str, float],  # peer_id -> available_vram_gb
    q_bits: int,
    safety_margin: float = 0.1
) -> Dict[str, Any]:
    """
    Distribute model layers optimally across multiple GPU peers.
    
    Args:
        config: Model configuration dictionary
        peers_vram: Dictionary mapping peer_id to available VRAM in GB
        q_bits: Quantization bits
        safety_margin: Safety margin as fraction of available VRAM
        
    Returns:
        Dictionary containing the distribution plan
    """
    total_layers = config.get("num_hidden_layers")
    vram_per_layer_gb, embedding_vram_gb, total_model_vram_gb = estimate_layer_vram(config, q_bits)
    print(f"🔍 Model {config.get('model_name')} has {total_layers} layers and requires {total_model_vram_gb} GB of VRAM")
    
    # Sort peers by available VRAM (descending)
    sorted_peers = sorted(peers_vram.items(), key=lambda x: x[1], reverse=True)

    distribution = {}
    remaining_layers = total_layers
    total_available_vram = sum(peers_vram.values())
    embedding_assigned = False
    
    for i, (peer_id, available_vram) in enumerate(sorted_peers):
        if remaining_layers <= 0:
            break
            
        # # HACK: For the peer with highest VRAM (first in sorted list), simulate only 4GB
        # if i == 0:
        #     print(f"🔧 [HACK] Simulating 4GB VRAM for highest VRAM peer {peer_id} (actual: {available_vram}GB)")
        #     available_vram = 0.25
            
        # Calculate max layers for this peer
        peer_calculation = calculate_max_layers_for_peer(config, available_vram, q_bits, safety_margin)
        
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
                    (embedding_vram_gb if will_handle_embeddings else 0) + 
                    (assigned_layers * vram_per_layer_gb), 3
                ),
                "vram_utilization_percent": round(
                    ((embedding_vram_gb if will_handle_embeddings else 0) + 
                     (assigned_layers * vram_per_layer_gb)) / available_vram * 100, 1
                )
            }
            remaining_layers -= assigned_layers
    
    # Check if we can fit the entire model
    total_assigned_layers = sum(peer["assigned_layers"] for peer in distribution.values())
    can_fit_model = total_assigned_layers >= total_layers and embedding_assigned
    
    return {
        "distribution": distribution,
        "model_info": {
            "total_layers": total_layers,
            "total_assigned_layers": total_assigned_layers,
            "vram_per_layer_gb": round(vram_per_layer_gb, 3),
            "embedding_vram_gb": round(embedding_vram_gb, 3),
            "total_model_vram_gb": round(total_model_vram_gb, 3)
        },
        "can_fit_model": can_fit_model,
        "remaining_layers": remaining_layers,
        "total_peers": len(peers_vram),
        "utilized_peers": len(distribution),
        "total_available_vram_gb": round(total_available_vram, 3)
    }
