from __future__ import annotations

from typing import Dict, Any, Iterable, Tuple
from dataclasses import dataclass

from transformers import PretrainedConfig
import torch


@dataclass
class LayerShard:
    """Represents the serialized weights for a single transformer layer in vLLM format."""
    weights: Dict[str, torch.Tensor]


class ShardingAdapter:
    """Abstract adapter for model-family-specific sharding to vLLM-compatible keys."""

    def __init__(self, config: PretrainedConfig) -> None:
        self.config = config

    # ---- High-level API the sharder will call ----
    def shard_embedding(self, hf_weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Return embedding weights keyed for vLLM."""
        raise NotImplementedError

    def shard_layer(self, layer_idx: int, hf_weights: Dict[str, torch.Tensor]) -> LayerShard:
        """Return a single layer shard in vLLM key format for a given layer index."""
        raise NotImplementedError

    def shard_lm_head(self, hf_weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Return lm_head weights (or empty dict if tied)."""
        raise NotImplementedError

    def shard_model_norm(self, hf_weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Return final model norm weights (if exists)."""
        return {}

    # ---- Utilities ----
    @staticmethod
    def fuse_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return torch.cat([q, k, v], dim=0)

    @staticmethod
    def fuse_gate_up(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        return torch.cat([gate, up], dim=0)