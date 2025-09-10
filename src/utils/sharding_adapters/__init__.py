from __future__ import annotations

from typing import Dict, Type

from transformers import PretrainedConfig

from .base import ShardingAdapter
from .llama import LlamaShardingAdapter

# from .mistral import MistralShardingAdapter
# from .mixtral_moe import MixtralMoEShardingAdapter
from .phi4 import Phi4ShardingAdapter
from .qwen import QwenShardingAdapter

# Simple registry for model-family specific sharding adapters
_ADAPTER_REGISTRY: Dict[str, Type[ShardingAdapter]] = {
    "llama": LlamaShardingAdapter,
    "qwen": QwenShardingAdapter,
    "qwen2": QwenShardingAdapter,
    "qwen3": QwenShardingAdapter,
    # "mistral": MistralShardingAdapter,
    # "mixtral": MixtralMoEShardingAdapter,
    "phi3": Phi4ShardingAdapter,  # Phi-4 uses phi3 model_type
    "phi4": Phi4ShardingAdapter,  # Also register as phi4 for future models
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
