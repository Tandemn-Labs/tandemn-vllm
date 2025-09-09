# src/utils/sharding_adapters/mixtral_moe.py

from __future__ import annotations

from typing import Dict

import torch
from transformers import PretrainedConfig

from .base import LayerShard, ShardingAdapter


class MixtralMoEShardingAdapter(ShardingAdapter):
    """
    Sharding adapter for Mixtral MoE models.

    CRITICAL: Preserve original HF checkpoint names exactly!
    vLLM's weight loader expects:
    - block_sparse_moe.experts.{expert_id}.w{1,2,3}.{qweight,qzeros,scales}
    - block_sparse_moe.gate.weight

    vLLM internally converts these to FusedMoE format automatically.
    """

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__(config)

        # Detect quantization
        self.is_awq = self._is_awq(config)
        if self.is_awq:
            print(
                "🔧 Mixtral MoE: AWQ quantization detected - preserving HF checkpoint names"
            )

        # MoE configuration
        self.num_local_experts = getattr(config, "num_local_experts", 8)
        self.num_experts_per_tok = getattr(config, "num_experts_per_tok", 2)

        print("🏗️ Mixtral MoE Configuration:")
        print(f"   • Quantization: {'AWQ' if self.is_awq else 'None'}")
        print(
            f"   • MoE: {self.num_local_experts} experts, {self.num_experts_per_tok} per token"
        )
        print("   • Weight format: HF checkpoint (vLLM will convert internally)")

    def _is_awq(self, config: PretrainedConfig) -> bool:
        """Check if model uses AWQ quantization."""
        qc = getattr(config, "quantization_config", None)
        if isinstance(qc, dict):
            return qc.get("quant_method") == "awq"
        return False

    @staticmethod
    def _cat_rows(*tensors: torch.Tensor) -> torch.Tensor:
        """Concatenate tensors along dimension 0 (rows) for float weights."""
        return torch.cat([t.detach().cpu() for t in tensors], dim=0)

    @staticmethod
    def _cat_cols(*tensors: torch.Tensor) -> torch.Tensor:
        """Concatenate tensors along dimension 1 (cols) for AWQ quantized weights."""
        return torch.cat([t.detach().cpu() for t in tensors], dim=1)

    def shard_embedding(
        self, hf_weights: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Shard embedding layer weights."""
        out: Dict[str, torch.Tensor] = {}

        embed_key = "model.embed_tokens.weight"
        if embed_key in hf_weights:
            out[embed_key] = hf_weights[embed_key].detach().cpu()
            print(f"✅ Sharded embedding: {hf_weights[embed_key].shape}")

        return out

    def shard_layer(
        self, layer_idx: int, hf_weights: Dict[str, torch.Tensor]
    ) -> LayerShard:
        """
        Shard a single Mixtral MoE transformer layer.
        Preserves original HF checkpoint structure completely.
        """
        p = f"model.layers.{layer_idx}"
        out: Dict[str, torch.Tensor] = {}

        print(f"🔧 Sharding Mixtral MoE layer {layer_idx}")

        # === ATTENTION SHARDING ===
        self._shard_attention_weights(p, hf_weights, out)

        # === MoE SHARDING - PRESERVE ORIGINAL NAMES ===
        self._shard_moe_weights_original(p, hf_weights, out)

        # === LAYER NORMS ===
        self._shard_layer_norms(p, hf_weights, out)

        print(f"✅ Layer {layer_idx}: {len(out)} weight tensors sharded")
        return LayerShard(weights=out)

    def _shard_attention_weights(
        self,
        prefix: str,
        hf_weights: Dict[str, torch.Tensor],
        out: Dict[str, torch.Tensor],
    ) -> None:
        """Shard attention weights with QKV fusion for vLLM."""

        if self.is_awq:
            # AWQ quantized attention - fuse QKV
            q_qw = f"{prefix}.self_attn.q_proj.qweight"
            k_qw = f"{prefix}.self_attn.k_proj.qweight"
            v_qw = f"{prefix}.self_attn.v_proj.qweight"

            if all(k in hf_weights for k in (q_qw, k_qw, v_qw)):
                out[f"{prefix}.self_attn.qkv_proj.qweight"] = self._cat_cols(
                    hf_weights[q_qw],
                    hf_weights[k_qw],
                    hf_weights[v_qw],
                )

            # Fuse qzeros
            q_qz = f"{prefix}.self_attn.q_proj.qzeros"
            k_qz = f"{prefix}.self_attn.k_proj.qzeros"
            v_qz = f"{prefix}.self_attn.v_proj.qzeros"

            if all(k in hf_weights for k in (q_qz, k_qz, v_qz)):
                out[f"{prefix}.self_attn.qkv_proj.qzeros"] = self._cat_cols(
                    hf_weights[q_qz],
                    hf_weights[k_qz],
                    hf_weights[v_qz],
                )

            # Fuse scales
            q_sc = f"{prefix}.self_attn.q_proj.scales"
            k_sc = f"{prefix}.self_attn.k_proj.scales"
            v_sc = f"{prefix}.self_attn.v_proj.scales"

            if all(k in hf_weights for k in (q_sc, k_sc, v_sc)):
                out[f"{prefix}.self_attn.qkv_proj.scales"] = self._cat_cols(
                    hf_weights[q_sc],
                    hf_weights[k_sc],
                    hf_weights[v_sc],
                )

            # Output projection (no fusion)
            o_qw = f"{prefix}.self_attn.o_proj.qweight"
            o_qz = f"{prefix}.self_attn.o_proj.qzeros"
            o_sc = f"{prefix}.self_attn.o_proj.scales"

            if o_qw in hf_weights:
                out[o_qw] = hf_weights[o_qw].detach().cpu()
            if o_qz in hf_weights:
                out[o_qz] = hf_weights[o_qz].detach().cpu()
            if o_sc in hf_weights:
                out[o_sc] = hf_weights[o_sc].detach().cpu()

        else:
            # Float attention weights - fuse QKV
            q_key = f"{prefix}.self_attn.q_proj.weight"
            k_key = f"{prefix}.self_attn.k_proj.weight"
            v_key = f"{prefix}.self_attn.v_proj.weight"

            if all(k in hf_weights for k in (q_key, k_key, v_key)):
                out[f"{prefix}.self_attn.qkv_proj.weight"] = self._cat_rows(
                    hf_weights[q_key],
                    hf_weights[k_key],
                    hf_weights[v_key],
                )

            # Output projection (no fusion)
            o_key = f"{prefix}.self_attn.o_proj.weight"
            if o_key in hf_weights:
                out[o_key] = hf_weights[o_key].detach().cpu()

    def _shard_moe_weights_original(
        self,
        prefix: str,
        hf_weights: Dict[str, torch.Tensor],
        out: Dict[str, torch.Tensor],
    ) -> None:
        """Preserve original HF MoE structure - let vLLM handle conversion."""

        expert_count = 0

        # === MoE EXPERTS - Keep original individual expert structure ===
        for expert_idx in range(self.num_local_experts):
            if self._copy_expert_weights_original(prefix, expert_idx, hf_weights, out):
                expert_count += 1

        # === MoE GATE ===
        gate_key = f"{prefix}.block_sparse_moe.gate.weight"
        if gate_key in hf_weights:
            out[gate_key] = hf_weights[gate_key].detach().cpu()
            print(f"✅ Sharded MoE gate: {hf_weights[gate_key].shape}")

        print(
            f"   • Preserved {expert_count}/{self.num_local_experts} experts in HF format"
        )

    def _copy_expert_weights_original(
        self,
        prefix: str,
        expert_idx: int,
        hf_weights: Dict[str, torch.Tensor],
        out: Dict[str, torch.Tensor],
    ) -> bool:
        """Copy expert weights exactly as they are in HF checkpoint."""
        expert_prefix = f"{prefix}.block_sparse_moe.experts.{expert_idx}"
        found_expert = False

        if self.is_awq:
            # AWQ quantized expert weights - copy as-is
            quant_suffixes = [".qweight", ".qzeros", ".scales"]

            for suffix in quant_suffixes:
                # Keep exact HF names: w1, w2, w3
                w1_key = f"{expert_prefix}.w1{suffix}"
                w2_key = f"{expert_prefix}.w2{suffix}"
                w3_key = f"{expert_prefix}.w3{suffix}"

                # Copy weights exactly as-is - NO conversion!
                if w1_key in hf_weights:
                    out[w1_key] = hf_weights[w1_key].detach().cpu()
                    found_expert = True

                if w2_key in hf_weights:
                    out[w2_key] = hf_weights[w2_key].detach().cpu()
                    found_expert = True

                if w3_key in hf_weights:
                    out[w3_key] = hf_weights[w3_key].detach().cpu()
                    found_expert = True

        else:
            # Float expert weights - copy as-is
            w1_key = f"{expert_prefix}.w1.weight"
            w2_key = f"{expert_prefix}.w2.weight"
            w3_key = f"{expert_prefix}.w3.weight"

            # Copy weights exactly as-is - NO conversion!
            if w1_key in hf_weights:
                out[w1_key] = hf_weights[w1_key].detach().cpu()
                found_expert = True

            if w2_key in hf_weights:
                out[w2_key] = hf_weights[w2_key].detach().cpu()
                found_expert = True

            if w3_key in hf_weights:
                out[w3_key] = hf_weights[w3_key].detach().cpu()
                found_expert = True

        return found_expert

    def _shard_layer_norms(
        self,
        prefix: str,
        hf_weights: Dict[str, torch.Tensor],
        out: Dict[str, torch.Tensor],
    ) -> None:
        """Shard layer normalization weights."""

        # Input layer norm
        input_norm_key = f"{prefix}.input_layernorm.weight"
        if input_norm_key in hf_weights:
            out[input_norm_key] = hf_weights[input_norm_key].detach().cpu()

        # Post-attention layer norm
        post_norm_key = f"{prefix}.post_attention_layernorm.weight"
        if post_norm_key in hf_weights:
            out[post_norm_key] = hf_weights[post_norm_key].detach().cpu()

    def shard_lm_head(
        self, hf_weights: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Shard language model head weights."""
        out: Dict[str, torch.Tensor] = {}

        if self.is_awq:
            # Check for quantized lm_head (rare but possible)
            quant_suffixes = [".qweight", ".qzeros", ".scales"]
            for suffix in quant_suffixes:
                lm_head_key = f"lm_head{suffix}"
                if lm_head_key in hf_weights:
                    out[lm_head_key] = hf_weights[lm_head_key].detach().cpu()
                    print(
                        f"✅ Found quantized lm_head{suffix}: {hf_weights[lm_head_key].shape}"
                    )

        # Standard lm_head weight (most common in AWQ models)
        lm_head_key = "lm_head.weight"
        if lm_head_key in hf_weights:
            out[lm_head_key] = hf_weights[lm_head_key].detach().cpu()
            print(f"✅ Found lm_head.weight: {hf_weights[lm_head_key].shape}")

        return out

    def shard_model_norm(
        self, hf_weights: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Shard final model normalization weights."""
        out: Dict[str, torch.Tensor] = {}

        norm_key = "model.norm.weight"
        if norm_key in hf_weights:
            out[norm_key] = hf_weights[norm_key].detach().cpu()
            print(f"✅ Sharded model.norm: {hf_weights[norm_key].shape}")

        return out
