from __future__ import annotations

from typing import Dict, Type

from transformers import PretrainedConfig

from .base import WeightLoadingAdapter
from .llama import LlamaWeightLoadingAdapter

# from .mistral import MistralWeightLoadingAdapter

_ADAPTER_REGISTRY: Dict[str, Type[WeightLoadingAdapter]] = {
    "llama": LlamaWeightLoadingAdapter,
    "mistral": LlamaWeightLoadingAdapter,
}


def get_adapter_for_config(config: PretrainedConfig):
    """Get the appropriate weight loading adapter for a model config."""
    model_type = getattr(config, "model_type", "").lower()

    if "llama" in model_type or "mistral" in model_type:
        from .llama import LlamaWeightLoadingAdapter

        return LlamaWeightLoadingAdapter

    else:
        # Default to Llama adapter for unknown models
        from .llama import LlamaWeightLoadingAdapter

        return LlamaWeightLoadingAdapter
