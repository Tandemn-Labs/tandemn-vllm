"""
Layer sharding utilities for breaking down models into individual layer files.
This module provides functionality to shard models layer by layer with vLLM compatibility.
"""

import os
import json
from typing import Optional, Dict, Any
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from safetensors.torch import save_file

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

def create_dummy_model_file(config_dir: str):
    """
    Create a minimal dummy model.safetensors that vLLM can find.
    Our custom loader will override these anyway.
    
    Args:
        config_dir: Path to the config directory where the dummy model should be saved
    """
    config_path = Path(config_dir)
    
    # Load config to get model dimensions
    with open(config_path / "config.json", "r") as f:
        config = json.load(f)
    
    vocab_size = config["vocab_size"]
    hidden_size = config["hidden_size"]
    num_layers = config["num_hidden_layers"]
    
    print(f"Creating dummy model file for {num_layers} layer model...")
    
    # Create minimal dummy weights (will be overridden by our loader)
    dummy_weights = {}
    
    # Embedding
    dummy_weights["model.embed_tokens.weight"] = torch.zeros(vocab_size, hidden_size, dtype=torch.float16)
    
    # LM head  
    dummy_weights["lm_head.weight"] = torch.zeros(vocab_size, hidden_size, dtype=torch.float16)
    
    # Model norm
    dummy_weights["model.norm.weight"] = torch.ones(hidden_size, dtype=torch.float16)
    
    # Dummy transformer layers (minimal weights)
    for i in range(num_layers):
        layer_prefix = f"model.layers.{i}"
        
        # Attention
        dummy_weights[f"{layer_prefix}.self_attn.qkv_proj.weight"] = torch.zeros(hidden_size * 3, hidden_size, dtype=torch.float16)
        dummy_weights[f"{layer_prefix}.self_attn.o_proj.weight"] = torch.zeros(hidden_size, hidden_size, dtype=torch.float16)
        
        # MLP  
        intermediate_size = config.get("intermediate_size", hidden_size * 4)
        dummy_weights[f"{layer_prefix}.mlp.gate_up_proj.weight"] = torch.zeros(intermediate_size * 2, hidden_size, dtype=torch.float16)
        dummy_weights[f"{layer_prefix}.mlp.down_proj.weight"] = torch.zeros(hidden_size, intermediate_size, dtype=torch.float16)
        
        # Layer norms
        dummy_weights[f"{layer_prefix}.input_layernorm.weight"] = torch.ones(hidden_size, dtype=torch.float16)
        dummy_weights[f"{layer_prefix}.post_attention_layernorm.weight"] = torch.ones(hidden_size, dtype=torch.float16)
    
    # Save dummy model
    model_path = config_path / "model.safetensors"
    save_file(dummy_weights, str(model_path))
    
    print(f"Created dummy model.safetensors with {len(dummy_weights)} weights")
    print(f"Saved to: {model_path}")
    print("NOTE: These are dummy weights that will be overridden by selective loader")

def shard_model_by_layers(
    model_name: str,
    output_dir: str,
    hf_token: Optional[str] = None,
    cache_dir: Optional[str] = None,
    model_layers_key: str = "model.layers"
) -> Dict[str, Any]:
    """
    Shard a model by individual layers for vLLM compatibility.
    
    Args:
        model_name: HuggingFace model name or path
        output_dir: Directory to save sharded layers
        hf_token: HuggingFace API token
        cache_dir: Cache directory for model downloads
        model_layers_key: Path to model layers in the model structure
        
    Returns:
        Dictionary with sharding metadata and results
    """
    
    print(f"Loading model {model_name}...")
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        cache_dir=cache_dir, 
        torch_dtype=torch.float16,
        device_map="cpu",  # Keep on CPU to avoid GPU memory issues
        token=hf_token
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        cache_dir=cache_dir,
        token=hf_token
    )
    config = AutoConfig.from_pretrained(
        model_name, 
        cache_dir=cache_dir,
        token=hf_token
    )
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save config and tokenizer (needed for all layers)
    config_dir = output_path / "config"
    config.save_pretrained(config_dir)
    tokenizer.save_pretrained(config_dir)
    
    # Create dummy model.safetensors for vLLM compatibility
    create_dummy_model_file(str(config_dir))
    
    # Create a metadata file
    metadata = {
        "model_name": model_name,
        "model_type": config.model_type,
        "num_layers": getattr(config, "num_hidden_layers", getattr(config, "n_layer", None)),
        "hidden_size": getattr(config, "hidden_size", getattr(config, "n_embd", None)),
        "layer_components": []
    }
    
    print(f"Model has {metadata['num_layers']} hidden layers")
    
    # 1. Save embedding layer (Llama-style only for TinyLlama)
    if hasattr(model, "model"):  # Llama-style
        embed_layer = model.model.embed_tokens
        embed_path = output_path / "embedding" / "layer.safetensors"
        embed_path.parent.mkdir(exist_ok=True)
        save_embedding_weights_vllm_format(embed_layer, str(embed_path))
        metadata["layer_components"].append({
            "type": "embedding",
            "path": "embedding/layer.safetensors", 
            "component_name": "model.embed_tokens"
        })
        
        # Get transformer layers
        transformer_layers = rgetattr(model, model_layers_key)
        layers_key = model_layers_key
        
    else:
        raise ValueError(f"Unknown model structure for {model_name}")
    
    # 2. Save each transformer layer individually
    layers_dir = output_path / "layers"
    layers_dir.mkdir(exist_ok=True)
    
    for layer_idx, layer in enumerate(transformer_layers):
        layer_path = layers_dir / f"layer_{layer_idx}.safetensors"
        save_layer_weights_vllm_format(layer, str(layer_path), f"layers.{layer_idx}", layer_idx)
        
        metadata["layer_components"].append({
            "type": "transformer_layer",
            "layer_index": layer_idx,
            "path": f"layers/layer_{layer_idx}.safetensors",
            "component_name": f"model.layers.{layer_idx}"
        })
    
    # 3. Save lm_head and model.norm (needed for complete model)
    if hasattr(model, "lm_head"):
        lm_head_path = output_path / "lm_head" / "layer.safetensors"
        lm_head_path.parent.mkdir(exist_ok=True)
        lm_head_state_dict = {"lm_head.weight": model.lm_head.weight.detach().cpu()}
        save_file(lm_head_state_dict, str(lm_head_path))
        metadata["layer_components"].append({
            "type": "lm_head",
            "path": "lm_head/layer.safetensors",
            "component_name": "lm_head"
        })
        print("Saved lm_head weights")
    
    if hasattr(model.model, "norm"):
        norm_path = output_path / "norm" / "layer.safetensors"
        norm_path.parent.mkdir(exist_ok=True)
        norm_state_dict = {"model.norm.weight": model.model.norm.weight.detach().cpu()}
        save_file(norm_state_dict, str(norm_path))
        metadata["layer_components"].append({
            "type": "norm",
            "path": "norm/layer.safetensors",
            "component_name": "model.norm"
        })
        print("Saved model.norm weights")
    
    # Save metadata
    metadata_path = output_path / "layer_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Successfully sharded model into {len(metadata['layer_components'])} components")
    print(f"Saved to: {output_path}")
    
    # Clean up model from memory
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return {
        "status": "success",
        "output_dir": str(output_path),
        "metadata": metadata,
        "total_components": len(metadata["layer_components"])
    } 