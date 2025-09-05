from __future__ import annotations

from typing import Dict

import torch
from transformers import PretrainedConfig

from .base import LayerShard, ShardingAdapter


class DeepSeekV3ShardingAdapter(ShardingAdapter):
    """
    Sharding adapter for DeepSeek V3 models with support for:
    - Multi-head Latent Attention (MLA) architecture
    - Mixture of Experts (MoE) with routed and shared experts
    - INT4 quantization (AWQ, GPTQ, etc.)
    - Variable layer configurations (dense vs MoE layers)

    DeepSeek V3 Architecture Overview:
    - 671B parameters total, 37B activated per token
    - 61 Transformer layers with 7168 hidden size
    - 257 experts (256 routed + 1 shared), 8 routed + shared activated per token
    - Multi-head Latent Attention with q_lora_rank and kv_lora_rank
    """

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__(config)

        # Detect quantization method
        self.quantization_config = getattr(config, "quantization_config", None)
        self.is_quantized = self.quantization_config is not None

        if self.is_quantized:
            self.quant_method = self.quantization_config.get("quant_method", "unknown")
            print(f"🔧 DeepSeek V3: {self.quant_method.upper()} quantization detected")

        # MoE configuration
        self.n_routed_experts = getattr(config, "n_routed_experts", None)
        self.n_shared_experts = getattr(config, "n_shared_experts", None)
        self.first_k_dense_replace = getattr(config, "first_k_dense_replace", 0)
        self.moe_layer_freq = getattr(config, "moe_layer_freq", 1)

        # MLA configuration
        self.q_lora_rank = getattr(config, "q_lora_rank", None)
        self.kv_lora_rank = getattr(config, "kv_lora_rank", None)

        print("🏗️ DeepSeek V3 Configuration:")
        print(
            f"   • Quantization: {self.quant_method if self.is_quantized else 'None'}"
        )
        print(
            f"   • MoE: {self.n_routed_experts} routed + {self.n_shared_experts} shared experts"
        )
        print(
            f"   • MLA: q_lora_rank={self.q_lora_rank}, kv_lora_rank={self.kv_lora_rank}"
        )

    def _is_moe_layer(self, layer_idx: int) -> bool:
        """Determine if a layer uses MoE based on DeepSeek configuration."""
        return (
            self.n_routed_experts is not None
            and layer_idx >= self.first_k_dense_replace
            and layer_idx % self.moe_layer_freq == 0
        )

    @staticmethod
    def _cat_rows(*tensors: torch.Tensor) -> torch.Tensor:
        """Concatenate tensors along dimension 0 (rows) for float weights."""
        return torch.cat([t.detach().cpu() for t in tensors], dim=0)

    @staticmethod
    def _cat_cols(*tensors: torch.Tensor) -> torch.Tensor:
        """Concatenate tensors along dimension 1 (cols) for quantized weights."""
        return torch.cat([t.detach().cpu() for t in tensors], dim=1)

    def shard_embedding(
        self, hf_weights: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Shard embedding layer weights."""
        out: Dict[str, torch.Tensor] = {}

        # Standard embedding
        embed_key = "model.embed_tokens.weight"
        if embed_key in hf_weights:
            out[embed_key] = hf_weights[embed_key].detach().cpu()
            print(f"✅ Sharded embedding: {hf_weights[embed_key].shape}")

        return out

    def shard_layer(
        self, layer_idx: int, hf_weights: Dict[str, torch.Tensor]
    ) -> LayerShard:
        """
        Shard a single DeepSeek V3 transformer layer.
        Handles both dense and MoE layers with MLA attention.
        """
        p = f"model.layers.{layer_idx}"
        out: Dict[str, torch.Tensor] = {}

        print(
            f"🔧 Sharding layer {layer_idx} ({'MoE' if self._is_moe_layer(layer_idx) else 'Dense'})"
        )

        # === ATTENTION SHARDING ===
        self._shard_attention_weights(layer_idx, p, hf_weights, out)

        # === MLP SHARDING ===
        if self._is_moe_layer(layer_idx):
            self._shard_moe_weights(layer_idx, p, hf_weights, out)
        else:
            self._shard_dense_mlp_weights(layer_idx, p, hf_weights, out)

        # === LAYER NORMS ===
        self._shard_layer_norms(p, hf_weights, out)

        print(f"✅ Layer {layer_idx}: {len(out)} weight tensors sharded")
        return LayerShard(weights=out)

    def _shard_attention_weights(
        self,
        layer_idx: int,
        prefix: str,
        hf_weights: Dict[str, torch.Tensor],
        out: Dict[str, torch.Tensor],
    ) -> None:
        """Shard Multi-head Latent Attention (MLA) weights."""

        if self.is_quantized:
            self._shard_quantized_attention(prefix, hf_weights, out)
        else:
            self._shard_float_attention(prefix, hf_weights, out)

    def _shard_float_attention(
        self,
        prefix: str,
        hf_weights: Dict[str, torch.Tensor],
        out: Dict[str, torch.Tensor],
    ) -> None:
        """Shard float attention weights."""

        # Check if using LoRA-based attention (MLA)
        if self.q_lora_rank is not None:
            # MLA: q_a_proj and q_b_proj
            q_a_key = f"{prefix}.self_attn.q_a_proj.weight"
            q_b_key = f"{prefix}.self_attn.q_b_proj.weight"
            if q_a_key in hf_weights:
                out[q_a_key] = hf_weights[q_a_key].detach().cpu()
            if q_b_key in hf_weights:
                out[q_b_key] = hf_weights[q_b_key].detach().cpu()
        else:
            # Standard attention: q_proj only
            q_key = f"{prefix}.self_attn.q_proj.weight"
            if q_key in hf_weights:
                out[q_key] = hf_weights[q_key].detach().cpu()

        # MLA: kv_a_proj_with_mqa and kv_b_proj (always present in MLA)
        kv_a_key = f"{prefix}.self_attn.kv_a_proj_with_mqa.weight"
        kv_b_key = f"{prefix}.self_attn.kv_b_proj.weight"
        if kv_a_key in hf_weights:
            out[kv_a_key] = hf_weights[kv_a_key].detach().cpu()
        if kv_b_key in hf_weights:
            out[kv_b_key] = hf_weights[kv_b_key].detach().cpu()

        # Output projection (always present)
        o_key = f"{prefix}.self_attn.o_proj.weight"
        if o_key in hf_weights:
            out[o_key] = hf_weights[o_key].detach().cpu()

        # MLA layer norms
        if self.q_lora_rank is not None:
            q_norm_key = f"{prefix}.self_attn.q_a_layernorm.weight"
            if q_norm_key in hf_weights:
                out[q_norm_key] = hf_weights[q_norm_key].detach().cpu()

        kv_norm_key = f"{prefix}.self_attn.kv_a_layernorm.weight"
        if kv_norm_key in hf_weights:
            out[kv_norm_key] = hf_weights[kv_norm_key].detach().cpu()

    def _shard_quantized_attention(
        self,
        prefix: str,
        hf_weights: Dict[str, torch.Tensor],
        out: Dict[str, torch.Tensor],
    ) -> None:
        """Shard quantized attention weights (qweight, qzeros, scales)."""

        # Define quantized weight suffixes
        quant_suffixes = [".qweight", ".qzeros", ".scales"]

        # MLA q projections
        if self.q_lora_rank is not None:
            for suffix in quant_suffixes:
                q_a_key = f"{prefix}.self_attn.q_a_proj{suffix}"
                q_b_key = f"{prefix}.self_attn.q_b_proj{suffix}"
                if q_a_key in hf_weights:
                    out[q_a_key] = hf_weights[q_a_key].detach().cpu()
                if q_b_key in hf_weights:
                    out[q_b_key] = hf_weights[q_b_key].detach().cpu()
        else:
            for suffix in quant_suffixes:
                q_key = f"{prefix}.self_attn.q_proj{suffix}"
                if q_key in hf_weights:
                    out[q_key] = hf_weights[q_key].detach().cpu()

        # MLA kv projections
        for suffix in quant_suffixes:
            kv_a_key = f"{prefix}.self_attn.kv_a_proj_with_mqa{suffix}"
            kv_b_key = f"{prefix}.self_attn.kv_b_proj{suffix}"
            o_key = f"{prefix}.self_attn.o_proj{suffix}"

            if kv_a_key in hf_weights:
                out[kv_a_key] = hf_weights[kv_a_key].detach().cpu()
            if kv_b_key in hf_weights:
                out[kv_b_key] = hf_weights[kv_b_key].detach().cpu()
            if o_key in hf_weights:
                out[o_key] = hf_weights[o_key].detach().cpu()

        # Layer norms (typically not quantized)
        if self.q_lora_rank is not None:
            q_norm_key = f"{prefix}.self_attn.q_a_layernorm.weight"
            if q_norm_key in hf_weights:
                out[q_norm_key] = hf_weights[q_norm_key].detach().cpu()

        kv_norm_key = f"{prefix}.self_attn.kv_a_layernorm.weight"
        if kv_norm_key in hf_weights:
            out[kv_norm_key] = hf_weights[kv_norm_key].detach().cpu()

    def _shard_dense_mlp_weights(
        self,
        layer_idx: int,
        prefix: str,
        hf_weights: Dict[str, torch.Tensor],
        out: Dict[str, torch.Tensor],
    ) -> None:
        """Shard dense MLP layer (non-MoE)."""

        if self.is_quantized:
            # Quantized dense MLP
            quant_suffixes = [".qweight", ".qzeros", ".scales"]

            # Process gate_proj, up_proj, down_proj with quantization
            for suffix in quant_suffixes:
                gate_key = f"{prefix}.mlp.gate_proj{suffix}"
                up_key = f"{prefix}.mlp.up_proj{suffix}"
                down_key = f"{prefix}.mlp.down_proj{suffix}"

                # Fuse gate_up_proj for quantized weights
                if gate_key in hf_weights and up_key in hf_weights:
                    if suffix == ".qweight":
                        # For qweight, concatenate along columns (dim=1)
                        out[f"{prefix}.mlp.gate_up_proj{suffix}"] = self._cat_cols(
                            hf_weights[gate_key], hf_weights[up_key]
                        )
                    else:
                        # For qzeros and scales, also concatenate along columns
                        out[f"{prefix}.mlp.gate_up_proj{suffix}"] = self._cat_cols(
                            hf_weights[gate_key], hf_weights[up_key]
                        )

                if down_key in hf_weights:
                    out[down_key] = hf_weights[down_key].detach().cpu()
        else:
            # Float dense MLP - fuse gate_up_proj
            gate_key = f"{prefix}.mlp.gate_proj.weight"
            up_key = f"{prefix}.mlp.up_proj.weight"
            down_key = f"{prefix}.mlp.down_proj.weight"

            if gate_key in hf_weights and up_key in hf_weights:
                out[f"{prefix}.mlp.gate_up_proj.weight"] = self._cat_rows(
                    hf_weights[gate_key], hf_weights[up_key]
                )

            if down_key in hf_weights:
                out[down_key] = hf_weights[down_key].detach().cpu()

    def _shard_moe_weights(
        self,
        layer_idx: int,
        prefix: str,
        hf_weights: Dict[str, torch.Tensor],
        out: Dict[str, torch.Tensor],
    ) -> None:
        """Shard MoE layer weights (routed experts + shared experts)."""

        # === ROUTED EXPERTS ===
        expert_count = 0
        for expert_idx in range(self.n_routed_experts or 0):
            if self._shard_single_expert(prefix, expert_idx, hf_weights, out):
                expert_count += 1

        # === SHARED EXPERTS ===
        if self.n_shared_experts and self.n_shared_experts > 0:
            self._shard_shared_experts(prefix, hf_weights, out)

        # === MoE GATE ===
        gate_key = f"{prefix}.mlp.gate.weight"
        if gate_key in hf_weights:
            out[gate_key] = hf_weights[gate_key].detach().cpu()

        # MoE gate bias (if present)
        gate_bias_key = f"{prefix}.mlp.gate.bias"
        if gate_bias_key in hf_weights:
            out[gate_bias_key] = hf_weights[gate_bias_key].detach().cpu()

        print(f"   • Sharded {expert_count}/{self.n_routed_experts} routed experts")

    def _shard_single_expert(
        self,
        prefix: str,
        expert_idx: int,
        hf_weights: Dict[str, torch.Tensor],
        out: Dict[str, torch.Tensor],
    ) -> bool:
        """Shard a single MoE expert."""
        expert_prefix = f"{prefix}.mlp.experts.{expert_idx}"
        found_expert = False

        if self.is_quantized:
            # Quantized expert weights
            quant_suffixes = [".qweight", ".qzeros", ".scales"]

            for suffix in quant_suffixes:
                gate_key = f"{expert_prefix}.gate_proj{suffix}"
                up_key = f"{expert_prefix}.up_proj{suffix}"
                down_key = f"{expert_prefix}.down_proj{suffix}"

                # Fuse gate_up_proj for expert
                if gate_key in hf_weights and up_key in hf_weights:
                    out[f"{expert_prefix}.gate_up_proj{suffix}"] = self._cat_cols(
                        hf_weights[gate_key], hf_weights[up_key]
                    )
                    found_expert = True

                if down_key in hf_weights:
                    out[down_key] = hf_weights[down_key].detach().cpu()
                    found_expert = True
        else:
            # Float expert weights
            gate_key = f"{expert_prefix}.gate_proj.weight"
            up_key = f"{expert_prefix}.up_proj.weight"
            down_key = f"{expert_prefix}.down_proj.weight"

            # Fuse gate_up_proj for expert
            if gate_key in hf_weights and up_key in hf_weights:
                out[f"{expert_prefix}.gate_up_proj.weight"] = self._cat_rows(
                    hf_weights[gate_key], hf_weights[up_key]
                )
                found_expert = True

            if down_key in hf_weights:
                out[down_key] = hf_weights[down_key].detach().cpu()
                found_expert = True

        return found_expert

    def _shard_shared_experts(
        self,
        prefix: str,
        hf_weights: Dict[str, torch.Tensor],
        out: Dict[str, torch.Tensor],
    ) -> None:
        """Shard shared expert weights."""
        shared_prefix = f"{prefix}.mlp.shared_experts"

        if self.is_quantized:
            # Quantized shared expert weights
            quant_suffixes = [".qweight", ".qzeros", ".scales"]

            for suffix in quant_suffixes:
                gate_key = f"{shared_prefix}.gate_proj{suffix}"
                up_key = f"{shared_prefix}.up_proj{suffix}"
                down_key = f"{shared_prefix}.down_proj{suffix}"

                # Fuse gate_up_proj for shared expert
                if gate_key in hf_weights and up_key in hf_weights:
                    out[f"{shared_prefix}.gate_up_proj{suffix}"] = self._cat_cols(
                        hf_weights[gate_key], hf_weights[up_key]
                    )

                if down_key in hf_weights:
                    out[down_key] = hf_weights[down_key].detach().cpu()
        else:
            # Float shared expert weights
            gate_key = f"{shared_prefix}.gate_proj.weight"
            up_key = f"{shared_prefix}.up_proj.weight"
            down_key = f"{shared_prefix}.down_proj.weight"

            # Fuse gate_up_proj for shared expert
            if gate_key in hf_weights and up_key in hf_weights:
                out[f"{shared_prefix}.gate_up_proj.weight"] = self._cat_rows(
                    hf_weights[gate_key], hf_weights[up_key]
                )

            if down_key in hf_weights:
                out[down_key] = hf_weights[down_key].detach().cpu()

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

        if self.is_quantized:
            # Check for quantized lm_head
            quant_suffixes = [".qweight", ".qzeros", ".scales"]
            for suffix in quant_suffixes:
                lm_head_key = f"lm_head{suffix}"
                if lm_head_key in hf_weights:
                    out[lm_head_key] = hf_weights[lm_head_key].detach().cpu()
                    print(
                        f"✅ Found quantized lm_head{suffix}: {hf_weights[lm_head_key].shape}"
                    )

        # Standard lm_head weight (may coexist with quantized in some models)
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
