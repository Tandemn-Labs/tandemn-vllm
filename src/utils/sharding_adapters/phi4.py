from __future__ import annotations

from typing import Dict

import torch
from transformers import PretrainedConfig

from .base import LayerShard, ShardingAdapter


class Phi4ShardingAdapter(ShardingAdapter):
    """Sharding adapter for Phi-4 (Phi3ForCausalLM architecture) models."""

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__(config)
        self.is_awq = self._is_awq(config)
        if self.is_awq:
            print("🔧 AWQ quantization detected - using AWQ-specific fusion (dim=1)")

    def _is_awq(self, config: PretrainedConfig) -> bool:
        qc = getattr(config, "quantization_config", None)
        if isinstance(qc, dict):
            return qc.get("quant_method") == "awq"
        return False

    # ---- Helpers ----
    @staticmethod
    def _cat_rows(*tensors: torch.Tensor) -> torch.Tensor:
        # For float weights: shapes [out, in], fuse along dim=0 (rows).
        return torch.cat([t.detach().cpu() for t in tensors], dim=0)

    @staticmethod
    def _cat_cols(*tensors: torch.Tensor) -> torch.Tensor:
        # For AWQ packed tensors: shapes [K, N/pack] (or [G, N/pack] / [G, N]),
        # fuse along dim=1 (cols).
        return torch.cat([t.detach().cpu() for t in tensors], dim=1)

    # ---- Parts ----
    def shard_embedding(
        self, hf_weights: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        key = "model.embed_tokens.weight"
        if key in hf_weights:
            out[key] = hf_weights[key].detach().cpu()
        return out

    def shard_layer(
        self, layer_idx: int, hf_weights: Dict[str, torch.Tensor]
    ) -> LayerShard:
        """
        Shard a single Phi-4 transformer layer.

        Phi-4 uses:
        - Combined gate_up_proj (gate and up projections fused)
        - Combined qkv_proj (Q, K, V projections fused)
        - Separate down_proj and o_proj
        - Input and post-attention layer norms
        """
        prefix = f"model.layers.{layer_idx}"
        out: Dict[str, torch.Tensor] = {}

        # Layer norms
        for norm_name in ["input_layernorm", "post_attention_layernorm"]:
            key = f"{prefix}.{norm_name}.weight"
            if key in hf_weights:
                out[key] = hf_weights[key].detach().cpu()

        # MLP weights
        if not self.is_awq:
            # Regular (non-quantized) weights
            # gate_up_proj is already fused in Phi-4
            gate_up_key = f"{prefix}.mlp.gate_up_proj.weight"
            if gate_up_key in hf_weights:
                out[gate_up_key] = hf_weights[gate_up_key].detach().cpu()

            # down_proj
            down_key = f"{prefix}.mlp.down_proj.weight"
            if down_key in hf_weights:
                out[down_key] = hf_weights[down_key].detach().cpu()
        else:
            # AWQ quantized weights
            # gate_up_proj (already fused)
            for suffix in ["qweight", "qzeros", "scales"]:
                key = f"{prefix}.mlp.gate_up_proj.{suffix}"
                if key in hf_weights:
                    out[key] = hf_weights[key].detach().cpu()

            # down_proj
            for suffix in ["qweight", "qzeros", "scales"]:
                key = f"{prefix}.mlp.down_proj.{suffix}"
                if key in hf_weights:
                    out[key] = hf_weights[key].detach().cpu()

        # Attention weights
        if not self.is_awq:
            # Regular weights
            # qkv_proj is already fused in Phi-4
            qkv_key = f"{prefix}.self_attn.qkv_proj.weight"
            if qkv_key in hf_weights:
                out[qkv_key] = hf_weights[qkv_key].detach().cpu()

            # o_proj
            o_key = f"{prefix}.self_attn.o_proj.weight"
            if o_key in hf_weights:
                out[o_key] = hf_weights[o_key].detach().cpu()
        else:
            # AWQ quantized weights
            # qkv_proj (already fused)
            for suffix in ["qweight", "qzeros", "scales"]:
                key = f"{prefix}.self_attn.qkv_proj.{suffix}"
                if key in hf_weights:
                    out[key] = hf_weights[key].detach().cpu()

            # o_proj
            for suffix in ["qweight", "qzeros", "scales"]:
                key = f"{prefix}.self_attn.o_proj.{suffix}"
                if key in hf_weights:
                    out[key] = hf_weights[key].detach().cpu()

        return LayerShard(weights=out)

    def shard_lm_head(
        self, hf_weights: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Shard the language modeling head.
        Phi-4 has tie_word_embeddings=false, so lm_head is separate.
        """
        out: Dict[str, torch.Tensor] = {}

        if not self.is_awq:
            # Regular weight
            key = "lm_head.weight"
            if key in hf_weights:
                out[key] = hf_weights[key].detach().cpu()
        else:
            # AWQ quantized weights
            for suffix in ["qweight", "qzeros", "scales"]:
                key = f"lm_head.{suffix}"
                if key in hf_weights:
                    out[key] = hf_weights[key].detach().cpu()

        return out

    def shard_model_norm(
        self, hf_weights: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Shard the final model normalization layer."""
        out: Dict[str, torch.Tensor] = {}
        key = "model.norm.weight"
        if key in hf_weights:
            out[key] = hf_weights[key].detach().cpu()
        return out
