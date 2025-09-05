from __future__ import annotations

from typing import Dict, Type

from transformers import PretrainedConfig

from .base import WeightLoadingAdapter
from .llama import LlamaWeightLoadingAdapter

_ADAPTER_REGISTRY: Dict[str, Type[WeightLoadingAdapter]] = {
    "llama": LlamaWeightLoadingAdapter
}


def get_adapter_for_config(config: PretrainedConfig) -> WeightLoadingAdapter:
    """Return a weight loading adapter instance for a given HF config.
    Falls back to LLaMA adapter for similar architectures if unknown.
    """
    model_type = getattr(config, "model_type", "").lower()
    for key, adapter_cls in _ADAPTER_REGISTRY.items():
        if model_type.startswith(key):
            return adapter_cls(config)
    # Default: try LLaMA-style
    return LlamaWeightLoadingAdapter(config)
