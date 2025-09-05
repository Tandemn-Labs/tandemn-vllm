from __future__ import annotations

from typing import Dict, Type

from transformers import PretrainedConfig

from .base import ShardingAdapter
from .llama import LlamaShardingAdapter
from .mistral import MistralShardingAdapter
from .qwen import QwenShardingAdapter

# Simple registry for model-family specific sharding adapters
_ADAPTER_REGISTRY: Dict[str, Type[ShardingAdapter]] = {
    "llama": LlamaShardingAdapter,
    "qwen": QwenShardingAdapter,
    "qwen2": QwenShardingAdapter,
    "qwen3": QwenShardingAdapter,
    "mistral": MistralShardingAdapter,
}


def get_adapter_for_config(config: PretrainedConfig) -> ShardingAdapter:
    """Return a sharding adapter instance for a given HF config.
    Falls back to LLaMA adapter for similar architectures if unknown.
    """
    model_type = getattr(config, "model_type", "").lower()
    for key, adapter_cls in _ADAPTER_REGISTRY.items():
        if model_type.startswith(key):
            return adapter_cls(config)
    # Default: try LLaMA-style
    return LlamaShardingAdapter(config)
