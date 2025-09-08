from __future__ import annotations

from typing import Dict, Type

from transformers import PretrainedConfig

from .base import WeightLoadingAdapter
from .llama import LlamaWeightLoadingAdapter
from .mixtral_moe import MixtralMoEWeightLoadingAdapter
from .qwen import QwenWeightLoadingAdapter

# from .mistral import MistralWeightLoadingAdapter

_ADAPTER_REGISTRY: Dict[str, Type[WeightLoadingAdapter]] = {
    "llama": LlamaWeightLoadingAdapter,
    "mistral": LlamaWeightLoadingAdapter,
    "qwen3": QwenWeightLoadingAdapter,  # Add Qwen3 support
    "qwen2": QwenWeightLoadingAdapter,  # Also support Qwen2
    "mixtral": MixtralMoEWeightLoadingAdapter,
}


def get_adapter_for_config(config: PretrainedConfig):
    """Get the appropriate weight loading adapter for a model config."""
    model_type = getattr(config, "model_type", "").lower()

    if "qwen" in model_type:
        from .qwen import QwenWeightLoadingAdapter

        return QwenWeightLoadingAdapter
    elif "mixtral" in model_type:
        from .mixtral_moe import MixtralMoEWeightLoadingAdapter

        return MixtralMoEWeightLoadingAdapter
    elif "llama" in model_type or "mistral" in model_type:
        from .llama import LlamaWeightLoadingAdapter

        return LlamaWeightLoadingAdapter
    else:
        # Default to Llama adapter for unknown models
        from .llama import LlamaWeightLoadingAdapter

        return LlamaWeightLoadingAdapter
