"""
Layer sharding utilities for breaking down models into individual layer files.
This module provides functionality to shard models layer by layer with vLLM compatibility.
"""

import os
import json
from typing import Optional, Dict, Any, List
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from safetensors.torch import save_file, load_file
from huggingface_hub import hf_hub_download, list_repo_files

os.environ["HF_TRANSFER"] = "0"


def rgetattr(obj, attr_path):
    """Recursively get attribute from object using dot notation."""
    attrs = attr_path.split('.')
    current = obj
    
    for attr in attrs:
        if not hasattr(current, attr):
            raise AttributeError(f"'{type(current).__name__}' object has no attribute '{attr}'")
        current = getattr(current, attr)
    
    return current

def fuse_qkv_weights(q_weight, k_weight, v_weight):
    """Fuse separate Q, K, V weights into single QKV weight for vLLM."""
    return torch.cat([q_weight, k_weight, v_weight], dim=0)

def fuse_gate_up_weights(gate_weight, up_weight):
    """Fuse separate gate and up weights into single gate_up weight for vLLM."""
    return torch.cat([gate_weight, up_weight], dim=0)

def save_layer_weights_vllm_format(layer: nn.Module, output_path: str, layer_name: str, layer_idx: int):
    """Save a single layer's weights to safetensors format with vLLM-compatible naming and fusion."""
    layer_state_dict = {}
    
    # Map original layer name to vLLM format
    vllm_layer_name = f"model.layers.{layer_idx}"
    
    # Handle attention weights - fuse Q, K, V into QKV
    if hasattr(layer, 'self_attn'):
        self_attn = layer.self_attn
        
        # Get Q, K, V weights
        q_weight = self_attn.q_proj.weight.detach().cpu()
        k_weight = self_attn.k_proj.weight.detach().cpu() 
        v_weight = self_attn.v_proj.weight.detach().cpu()
        
        # Fuse into QKV
        qkv_weight = fuse_qkv_weights(q_weight, k_weight, v_weight)
        layer_state_dict[f"{vllm_layer_name}.self_attn.qkv_proj.weight"] = qkv_weight
        
        # Output projection
        layer_state_dict[f"{vllm_layer_name}.self_attn.o_proj.weight"] = self_attn.o_proj.weight.detach().cpu()
    
    # Handle MLP weights - fuse gate and up into gate_up
    if hasattr(layer, 'mlp'):
        mlp = layer.mlp
        
        # Get gate and up weights
        gate_weight = mlp.gate_proj.weight.detach().cpu()
        up_weight = mlp.up_proj.weight.detach().cpu()
        
        # Fuse into gate_up
        gate_up_weight = fuse_gate_up_weights(gate_weight, up_weight)
        layer_state_dict[f"{vllm_layer_name}.mlp.gate_up_proj.weight"] = gate_up_weight
        
        # Down projection
        layer_state_dict[f"{vllm_layer_name}.mlp.down_proj.weight"] = mlp.down_proj.weight.detach().cpu()
    
    # Layer norms
    if hasattr(layer, 'input_layernorm'):
        layer_state_dict[f"{vllm_layer_name}.input_layernorm.weight"] = layer.input_layernorm.weight.detach().cpu()
    
    if hasattr(layer, 'post_attention_layernorm'):
        layer_state_dict[f"{vllm_layer_name}.post_attention_layernorm.weight"] = layer.post_attention_layernorm.weight.detach().cpu()
    
    print(f"Saving vLLM-compatible layer {layer_idx} with {len(layer_state_dict)} fused parameters")
    
    # Save to safetensors
    save_file(layer_state_dict, output_path)

def save_embedding_weights_vllm_format(embed_layer: nn.Module, output_path: str):
    """Save embedding weights in vLLM format."""
    layer_state_dict = {
        "model.embed_tokens.weight": embed_layer.weight.detach().cpu()
    }
    save_file(layer_state_dict, output_path)

# def create_dummy_model_file(config_dir: str):
#     """
#     Create a minimal dummy model.safetensors that vLLM can find.
#     Our custom loader will override these anyway.
    
#     Args:
#         config_dir: Path to the config directory where the dummy model should be saved
#     """
#     config_path = Path(config_dir)
    
#     # Load config to get model dimensions
#     with open(config_path / "config.json", "r") as f:
#         config = json.load(f)
    
#     vocab_size = config["vocab_size"]
#     hidden_size = config["hidden_size"]
#     num_layers = config["num_hidden_layers"]
    
#     print(f"Creating minimal dummy model file for {num_layers} layer model...")
    
#     # Create minimal dummy weights (will be overridden by our loader)
#     dummy_weights = {}
    
#     # Embedding
#     dummy_weights["model.embed_tokens.weight"] = torch.zeros(vocab_size, hidden_size, dtype=torch.float16)
    
#     # LM head  
#     dummy_weights["lm_head.weight"] = torch.zeros(vocab_size, hidden_size, dtype=torch.float16)
    
#     # Model norm
#     dummy_weights["model.norm.weight"] = torch.ones(hidden_size, dtype=torch.float16)
    
#     # Create dummy weights for ALL layers (not just layer 0)
#     intermediate_size = config.get("intermediate_size", hidden_size * 4)
    
#     for layer_idx in range(num_layers):
#         layer_prefix = f"model.layers.{layer_idx}"
        
#         # Attention
#         dummy_weights[f"{layer_prefix}.self_attn.qkv_proj.weight"] = torch.zeros(hidden_size * 3, hidden_size, dtype=torch.float16)
#         dummy_weights[f"{layer_prefix}.self_attn.o_proj.weight"] = torch.zeros(hidden_size, hidden_size, dtype=torch.float16)
        
#         # MLP  
#         dummy_weights[f"{layer_prefix}.mlp.gate_up_proj.weight"] = torch.zeros(intermediate_size * 2, hidden_size, dtype=torch.float16)
#         dummy_weights[f"{layer_prefix}.mlp.down_proj.weight"] = torch.zeros(hidden_size, intermediate_size, dtype=torch.float16)
        
#         # Layer norms
#         dummy_weights[f"{layer_prefix}.input_layernorm.weight"] = torch.ones(hidden_size, dtype=torch.float16)
#         dummy_weights[f"{layer_prefix}.post_attention_layernorm.weight"] = torch.ones(hidden_size, dtype=torch.float16)
    
#     # Save dummy model
#     model_path = config_path / "model.safetensors"
#     save_file(dummy_weights, str(model_path))
    
#     print(f"Created minimal dummy model.safetensors with {len(dummy_weights)} weights")
#     print(f"Saved to: {model_path}")
#     print("NOTE: These are dummy weights that will be overridden by selective loader")

def get_model_safetensors_files(model_name: str, hf_token: Optional[str] = None) -> List[str]:
    """
    Get list of safetensors files for a model from HuggingFace.
    
    Args:
        model_name: HuggingFace model name
        hf_token: HuggingFace API token
        
    Returns:
        List of safetensors file paths
    """
    try:
        files = list_repo_files(model_name, token=hf_token)
        safetensors_files = [f for f in files if f.endswith('.safetensors')]
        print(f"Found {len(safetensors_files)} safetensors files for {model_name}")
        return safetensors_files
    except Exception as e:
        print(f"Error listing files for {model_name}: {e}")
        # Note: list_repo_files doesn't support trust_remote_code parameter
        # This is only needed for model/config loading, not file listing
        return []

def extract_layer_weights_from_safetensors(
    model_name: str,
    layer_idx: int,
    hf_token: Optional[str] = None,
    cache_dir: Optional[str] = None
) -> Dict[str, torch.Tensor]:
    """
    Extract weights for a specific layer directly from safetensors files.
    
    Args:
        model_name: HuggingFace model name
        layer_idx: Layer index to extract
        hf_token: HuggingFace API token
        cache_dir: Cache directory
        
    Returns:
        Dictionary of layer weights in vLLM format
    """
    layer_weights = {}
    
    # Get safetensors files
    safetensors_files = get_model_safetensors_files(model_name, hf_token)
    if not safetensors_files:
        raise ValueError(f"No safetensors files found for {model_name}")
    
    # Download and process each safetensors file
    for file_path in safetensors_files:
        try:
            # Download the file
            local_path = hf_hub_download(
                repo_id=model_name,
                filename=file_path,
                token=hf_token,
                cache_dir=cache_dir
            )
            
            print(f"Processing {file_path} for layer {layer_idx}...")
            
            # Load the safetensors file
            weights = load_file(local_path)
            
            # Extract layer-specific weights
            layer_prefix = f"model.layers.{layer_idx}"
            
            # Attention weights
            q_key = f"{layer_prefix}.self_attn.q_proj.weight"
            k_key = f"{layer_prefix}.self_attn.k_proj.weight"
            v_key = f"{layer_prefix}.self_attn.v_proj.weight"
            o_key = f"{layer_prefix}.self_attn.o_proj.weight"
            
            if q_key in weights and k_key in weights and v_key in weights:
                # Fuse Q, K, V into QKV
                qkv_weight = fuse_qkv_weights(weights[q_key], weights[k_key], weights[v_key])
                layer_weights[f"{layer_prefix}.self_attn.qkv_proj.weight"] = qkv_weight
                
                # Output projection
                if o_key in weights:
                    layer_weights[f"{layer_prefix}.self_attn.o_proj.weight"] = weights[o_key]
            
            # MLP weights
            gate_key = f"{layer_prefix}.mlp.gate_proj.weight"
            up_key = f"{layer_prefix}.mlp.up_proj.weight"
            down_key = f"{layer_prefix}.mlp.down_proj.weight"
            
            if gate_key in weights and up_key in weights:
                # Fuse gate and up into gate_up
                gate_up_weight = fuse_gate_up_weights(weights[gate_key], weights[up_key])
                layer_weights[f"{layer_prefix}.mlp.gate_up_proj.weight"] = gate_up_weight
                
                # Down projection
                if down_key in weights:
                    layer_weights[f"{layer_prefix}.mlp.down_proj.weight"] = weights[down_key]
            
            # Layer norms
            input_norm_key = f"{layer_prefix}.input_layernorm.weight"
            post_norm_key = f"{layer_prefix}.post_attention_layernorm.weight"
            
            if input_norm_key in weights:
                layer_weights[f"{layer_prefix}.input_layernorm.weight"] = weights[input_norm_key]
            
            if post_norm_key in weights:
                layer_weights[f"{layer_prefix}.post_attention_layernorm.weight"] = weights[post_norm_key]
                
        except Exception as e:
            print(f"Warning: Error processing {file_path}: {e}")
            continue
    
    return layer_weights

def extract_embedding_weights_from_safetensors(
    model_name: str,
    hf_token: Optional[str] = None,
    cache_dir: Optional[str] = None
) -> Dict[str, torch.Tensor]:
    """
    Extract embedding weights directly from safetensors files.
    
    Args:
        model_name: HuggingFace model name
        hf_token: HuggingFace API token
        cache_dir: Cache directory
        
    Returns:
        Dictionary of embedding weights in vLLM format
    """
    embedding_weights = {}
    
    # Get safetensors files
    safetensors_files = get_model_safetensors_files(model_name, hf_token)
    if not safetensors_files:
        raise ValueError(f"No safetensors files found for {model_name}")
    
    # Download and process each safetensors file
    for file_path in safetensors_files:
        try:
            # Download the file
            local_path = hf_hub_download(
                repo_id=model_name,
                filename=file_path,
                token=hf_token,
                cache_dir=cache_dir
            )
            
            print(f"Processing {file_path} for embedding...")
            
            # Load the safetensors file
            weights = load_file(local_path)
            
            # Extract embedding weights
            embed_key = "model.embed_tokens.weight"
            if embed_key in weights:
                embedding_weights["model.embed_tokens.weight"] = weights[embed_key]
                break  # Found embedding, no need to check other files
                
        except Exception as e:
            print(f"Warning: Error processing {file_path}: {e}")
            continue
    
    return embedding_weights

def extract_lm_head_weights_from_safetensors(
    model_name: str,
    hf_token: Optional[str] = None,
    cache_dir: Optional[str] = None
) -> Dict[str, torch.Tensor]:
    """
    Extract LM head weights directly from safetensors files.
    Handles tied weights by falling back to embed_tokens.weight if lm_head.weight is not found.
    
    Args:
        model_name: HuggingFace model name
        hf_token: HuggingFace API token
        cache_dir: Cache directory
        
    Returns:
        Dictionary of LM head weights
    """
    lm_head_weights = {}
    
    # Get safetensors files
    safetensors_files = get_model_safetensors_files(model_name, hf_token)
    if not safetensors_files:
        raise ValueError(f"No safetensors files found for {model_name}")
    
    # Download and process each safetensors file
    for file_path in safetensors_files:
        try:
            # Download the file
            local_path = hf_hub_download(
                repo_id=model_name,
                filename=file_path,
                token=hf_token,
                cache_dir=cache_dir
            )
            
            print(f"Processing {file_path} for LM head...")
            
            # Load the safetensors file
            weights = load_file(local_path)
            
            # Try to extract LM head weights
            lm_head_key = "lm_head.weight"
            embed_key = "model.embed_tokens.weight"
            
            if lm_head_key in weights:
                lm_head_weights["lm_head.weight"] = weights[lm_head_key]
                print("Found explicit lm_head.weight")
                break  # Found LM head, no need to check other files
            elif embed_key in weights:
                # Handle tied weights - use embed_tokens as lm_head
                lm_head_weights["lm_head.weight"] = weights[embed_key]
                print("Using tied weights: model.embed_tokens.weight as lm_head.weight")
                break  # Found embedding, can use as LM head
                
        except Exception as e:
            print(f"Warning: Error processing {file_path}: {e}")
            continue
    
    return lm_head_weights

def extract_norm_weights_from_safetensors(
    model_name: str,
    hf_token: Optional[str] = None,
    cache_dir: Optional[str] = None
) -> Dict[str, torch.Tensor]:
    """
    Extract model norm weights directly from safetensors files.
    
    Args:
        model_name: HuggingFace model name
        hf_token: HuggingFace API token
        cache_dir: Cache directory
        
    Returns:
        Dictionary of norm weights
    """
    norm_weights = {}
    
    # Get safetensors files
    safetensors_files = get_model_safetensors_files(model_name, hf_token)
    if not safetensors_files:
        raise ValueError(f"No safetensors files found for {model_name}")
    
    # Download and process each safetensors file
    for file_path in safetensors_files:
        try:
            # Download the file
            local_path = hf_hub_download(
                repo_id=model_name,
                filename=file_path,
                token=hf_token,
                cache_dir=cache_dir
            )
            
            print(f"Processing {file_path} for model norm...")
            
            # Load the safetensors file
            weights = load_file(local_path)
            
            # Extract norm weights
            norm_key = "model.norm.weight"
            if norm_key in weights:
                norm_weights["model.norm.weight"] = weights[norm_key]
                break  # Found norm, no need to check other files
                
        except Exception as e:
            print(f"Warning: Error processing {file_path}: {e}")
            continue
    
    return norm_weights

def shard_model_by_layers_safetensors(
    model_name: str,
    output_dir: str,
    hf_token: Optional[str] = None,
    cache_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Shard a model by individual layers using direct safetensors processing.
    This approach avoids loading the entire model into memory.
    
    Args:
        model_name: HuggingFace model name or path
        output_dir: Directory to save sharded layers
        hf_token: HuggingFace API token
        cache_dir: Cache directory for model downloads
        
    Returns:
        Dictionary with sharding metadata and results
    """
    
    print(f"🔪 Starting safetensors-based layer sharding for {model_name}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Download config and tokenizer (needed for all layers)
    config = AutoConfig.from_pretrained(
        model_name, 
        cache_dir=cache_dir,
        token=hf_token,
        trust_remote_code=True  # Allow custom model code
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        cache_dir=cache_dir,
        token=hf_token,
        trust_remote_code=True  # Allow custom tokenizer code
    )
    
    # Save config and tokenizer
    config_dir = output_path / "config"
    config.save_pretrained(config_dir)
    tokenizer.save_pretrained(config_dir)
    
    # Create minimal dummy model.safetensors for vLLM compatibility
    # create_dummy_model_file(str(config_dir))
    
    # Create a metadata file
    metadata = {
        "model_name": model_name,
        "model_type": config.model_type,
        "num_layers": getattr(config, "num_hidden_layers", getattr(config, "n_layer", None)),
        "hidden_size": getattr(config, "hidden_size", getattr(config, "n_embd", None)),
        "layer_components": []
    }
    
    print(f"Model has {metadata['num_layers']} hidden layers")
    
    # 1. Save embedding layer
    try:
        embed_weights = extract_embedding_weights_from_safetensors(model_name, hf_token, cache_dir)
        if embed_weights:
            embed_path = output_path / "embedding" / "layer.safetensors"
            embed_path.parent.mkdir(exist_ok=True)
            save_file(embed_weights, str(embed_path))
            metadata["layer_components"].append({
                "type": "embedding",
                "path": "embedding/layer.safetensors", 
                "component_name": "model.embed_tokens"
            })
            print("✅ Saved embedding weights")
        else:
            print("⚠️ No embedding weights found")
    except Exception as e:
        print(f"❌ Error extracting embedding: {e}")
    
    # 2. Save each transformer layer individually
    layers_dir = output_path / "layers"
    layers_dir.mkdir(exist_ok=True)
    
    for layer_idx in range(metadata['num_layers']):
        try:
            print(f"🔪 Processing layer {layer_idx}/{metadata['num_layers']-1}...")
            
            layer_weights = extract_layer_weights_from_safetensors(
                model_name, layer_idx, hf_token, cache_dir
            )
            
            if layer_weights:
                layer_path = layers_dir / f"layer_{layer_idx}.safetensors"
                save_file(layer_weights, str(layer_path))
                
                metadata["layer_components"].append({
                    "type": "transformer_layer",
                    "layer_index": layer_idx,
                    "path": f"layers/layer_{layer_idx}.safetensors",
                    "component_name": f"model.layers.{layer_idx}"
                })
                print(f"✅ Saved layer {layer_idx} with {len(layer_weights)} weights")
            else:
                print(f"⚠️ No weights found for layer {layer_idx}")
                
        except Exception as e:
            print(f"❌ Error processing layer {layer_idx}: {e}")
            continue
    
    # 3. Save lm_head
    try:
        lm_head_weights = extract_lm_head_weights_from_safetensors(model_name, hf_token, cache_dir)
        if lm_head_weights:
            lm_head_path = output_path / "lm_head" / "layer.safetensors"
            lm_head_path.parent.mkdir(exist_ok=True)
            save_file(lm_head_weights, str(lm_head_path))
            metadata["layer_components"].append({
                "type": "lm_head",
                "path": "lm_head/layer.safetensors",
                "component_name": "lm_head"
            })
            print("✅ Saved lm_head weights")
        else:
            print("⚠️ No lm_head weights found")
    except Exception as e:
        print(f"❌ Error extracting lm_head: {e}")
    
    # 4. Save model.norm
    try:
        norm_weights = extract_norm_weights_from_safetensors(model_name, hf_token, cache_dir)
        if norm_weights:
            norm_path = output_path / "norm" / "layer.safetensors"
            norm_path.parent.mkdir(exist_ok=True)
            save_file(norm_weights, str(norm_path))
            metadata["layer_components"].append({
                "type": "norm",
                "path": "norm/layer.safetensors",
                "component_name": "model.norm"
            })
            print("✅ Saved model.norm weights")
        else:
            print("⚠️ No model.norm weights found")
    except Exception as e:
        print(f"❌ Error extracting model.norm: {e}")
    
    # Save metadata
    metadata_path = output_path / "layer_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Successfully sharded model into {len(metadata['layer_components'])} components")
    print(f"📁 Saved to: {output_path}")
    
    return {
        "status": "success",
        "output_dir": str(output_path),
        "metadata": metadata,
        "total_components": len(metadata["layer_components"])
    }
