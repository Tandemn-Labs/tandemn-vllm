from __future__ import annotations

from typing import Dict, Type

from transformers import PretrainedConfig

from .base import ShardingAdapter
from .llama import LlamaShardingAdapter
from .qwen import QwenShardingAdapter
from .compressed_tensors_llama import CompressedTensorsLlamaShardingAdapter


# Simple registry for model-family specific sharding adapters
_ADAPTER_REGISTRY: Dict[str, Type[ShardingAdapter]] = {
    "llama": LlamaShardingAdapter,
    "qwen": QwenShardingAdapter,
    "qwen2": QwenShardingAdapter,
    "qwen3": QwenShardingAdapter,
}


def get_adapter_for_config(config: PretrainedConfig) -> ShardingAdapter:
    """
    Return a sharding adapter instance for a given HF config.
    
    Detects compressed-tensors quantization and uses specialized adapters
    for quantized models when needed.
    """
    model_type = getattr(config, "model_type", "").lower()
    
    # Check for compressed-tensors quantization
    quant_config = getattr(config, "quantization_config", None)
    if quant_config:
        quant_method = quant_config.get("quant_method", "") if isinstance(quant_config, dict) else ""
        
        # Use specialized compressed-tensors adapter for Llama-family models
        if quant_method == "compressed-tensors":
            if model_type.startswith("llama"):
                print(f"ℹ️ Detected compressed-tensors quantization for {model_type}, using CompressedTensorsLlamaShardingAdapter")
                return CompressedTensorsLlamaShardingAdapter(config)
            # Add more compressed-tensors adapters for other architectures as needed
            # elif model_type.startswith("qwen"):
            #     return CompressedTensorsQwenShardingAdapter(config)
    
    # Standard (non-quantized) model adapters
    for key, adapter_cls in _ADAPTER_REGISTRY.items():
        if model_type.startswith(key):
            return adapter_cls(config)
    
    # Default: try LLaMA-style
    return LlamaShardingAdapter(config) 