from __future__ import annotations

from typing import Dict

import torch
from transformers import PretrainedConfig

from .base import LayerShard, ShardingAdapter


class LlamaShardingAdapter(ShardingAdapter):
    """Sharding adapter for LLaMA-family models into vLLM-compatible keys."""

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
        # fuse along dim=1 (cols). Supports 2 or 3 inputs.
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
        p = f"model.layers.{layer_idx}"
        out: Dict[str, torch.Tensor] = {}

        if self.is_awq:
            # Attention: qkv (AWQ) - concat along dim=1
            q_qw = f"{p}.self_attn.q_proj.qweight"
            k_qw = f"{p}.self_attn.k_proj.qweight"
            v_qw = f"{p}.self_attn.v_proj.qweight"
            if all(k in hf_weights for k in (q_qw, k_qw, v_qw)):
                out[f"{p}.self_attn.qkv_proj.qweight"] = self._cat_cols(
                    hf_weights[q_qw],
                    hf_weights[k_qw],
                    hf_weights[v_qw],
                )

            q_qz = f"{p}.self_attn.q_proj.qzeros"
            k_qz = f"{p}.self_attn.k_proj.qzeros"
            v_qz = f"{p}.self_attn.v_proj.qzeros"
            if all(k in hf_weights for k in (q_qz, k_qz, v_qz)):
                out[f"{p}.self_attn.qkv_proj.qzeros"] = self._cat_cols(
                    hf_weights[q_qz],
                    hf_weights[k_qz],
                    hf_weights[v_qz],
                )

            q_sc = f"{p}.self_attn.q_proj.scales"
            k_sc = f"{p}.self_attn.k_proj.scales"
            v_sc = f"{p}.self_attn.v_proj.scales"
            if all(k in hf_weights for k in (q_sc, k_sc, v_sc)):
                out[f"{p}.self_attn.qkv_proj.scales"] = self._cat_cols(
                    hf_weights[q_sc],
                    hf_weights[k_sc],
                    hf_weights[v_sc],
                )

            # Attention: o_proj (AWQ)
            o_qw = f"{p}.self_attn.o_proj.qweight"
            o_qz = f"{p}.self_attn.o_proj.qzeros"
            o_sc = f"{p}.self_attn.o_proj.scales"
            if o_qw in hf_weights:
                out[f"{p}.self_attn.o_proj.qweight"] = hf_weights[o_qw].detach().cpu()
            if o_qz in hf_weights:
                out[f"{p}.self_attn.o_proj.qzeros"] = hf_weights[o_qz].detach().cpu()
            if o_sc in hf_weights:
                out[f"{p}.self_attn.o_proj.scales"] = hf_weights[o_sc].detach().cpu()

            # MLP: gate_up fused (AWQ) - concat along dim=1
            gate_qw = f"{p}.mlp.gate_proj.qweight"
            up_qw = f"{p}.mlp.up_proj.qweight"
            if all(k in hf_weights for k in (gate_qw, up_qw)):
                out[f"{p}.mlp.gate_up_proj.qweight"] = self._cat_cols(
                    hf_weights[gate_qw],
                    hf_weights[up_qw],
                )
            gate_qz = f"{p}.mlp.gate_proj.qzeros"
            up_qz = f"{p}.mlp.up_proj.qzeros"
            if all(k in hf_weights for k in (gate_qz, up_qz)):
                out[f"{p}.mlp.gate_up_proj.qzeros"] = self._cat_cols(
                    hf_weights[gate_qz],
                    hf_weights[up_qz],
                )
            gate_sc = f"{p}.mlp.gate_proj.scales"
            up_sc = f"{p}.mlp.up_proj.scales"
            if all(k in hf_weights for k in (gate_sc, up_sc)):
                out[f"{p}.mlp.gate_up_proj.scales"] = self._cat_cols(
                    hf_weights[gate_sc],
                    hf_weights[up_sc],
                )

            # MLP: down_proj (AWQ)
            down_qw = f"{p}.mlp.down_proj.qweight"
            down_qz = f"{p}.mlp.down_proj.qzeros"
            down_sc = f"{p}.mlp.down_proj.scales"
            if down_qw in hf_weights:
                out[f"{p}.mlp.down_proj.qweight"] = hf_weights[down_qw].detach().cpu()
            if down_qz in hf_weights:
                out[f"{p}.mlp.down_proj.qzeros"] = hf_weights[down_qz].detach().cpu()
            if down_sc in hf_weights:
                out[f"{p}.mlp.down_proj.scales"] = hf_weights[down_sc].detach().cpu()

        else:
            # Float weights path
            # Attention: fuse qkv along dim=0 (rows)
            q_key = f"{p}.self_attn.q_proj.weight"
            k_key = f"{p}.self_attn.k_proj.weight"
            v_key = f"{p}.self_attn.v_proj.weight"
            if all(k in hf_weights for k in (q_key, k_key, v_key)):
                out[f"{p}.self_attn.qkv_proj.weight"] = self._cat_rows(
                    hf_weights[q_key],
                    hf_weights[k_key],
                    hf_weights[v_key],
                )
            o_key = f"{p}.self_attn.o_proj.weight"
            if o_key in hf_weights:
                out[f"{p}.self_attn.o_proj.weight"] = hf_weights[o_key].detach().cpu()

            # MLP: fuse gate_up along dim=0 (rows)
            gate_key = f"{p}.mlp.gate_proj.weight"
            up_key = f"{p}.mlp.up_proj.weight"
            down_key = f"{p}.mlp.down_proj.weight"
            if all(k in hf_weights for k in (gate_key, up_key)):
                out[f"{p}.mlp.gate_up_proj.weight"] = self._cat_rows(
                    hf_weights[gate_key],
                    hf_weights[up_key],
                )
            if down_key in hf_weights:
                out[f"{p}.mlp.down_proj.weight"] = hf_weights[down_key].detach().cpu()

        # Norms (same for both)
        in_ln = f"{p}.input_layernorm.weight"
        post_ln = f"{p}.post_attention_layernorm.weight"
        if in_ln in hf_weights:
            out[in_ln] = hf_weights[in_ln].detach().cpu()
        if post_ln in hf_weights:
            out[post_ln] = hf_weights[post_ln].detach().cpu()

        return LayerShard(weights=out)

    def shard_lm_head(
        self, hf_weights: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        if self.is_awq:
            # Check for quantized lm_head (rare for AWQ)
            if "lm_head.qweight" in hf_weights:
                out["lm_head.qweight"] = hf_weights["lm_head.qweight"].detach().cpu()
            if "lm_head.qzeros" in hf_weights:
                out["lm_head.qzeros"] = hf_weights["lm_head.qzeros"].detach().cpu()
            if "lm_head.scales" in hf_weights:
                out["lm_head.scales"] = hf_weights["lm_head.scales"].detach().cpu()

            # CRITICAL: Also check for unquantized lm_head (common in AWQ models!)
            if "lm_head.weight" in hf_weights:
                out["lm_head.weight"] = hf_weights["lm_head.weight"].detach().cpu()
                print(f"✅ Found unquantized lm_head.weight in AWQ model (size: {hf_weights['lm_head.weight'].shape})")
        else:
            if "lm_head.weight" in hf_weights:
                out["lm_head.weight"] = hf_weights["lm_head.weight"].detach().cpu()
        return out

    def shard_model_norm(
        self, hf_weights: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        if "model.norm.weight" in hf_weights:
            out["model.norm.weight"] = hf_weights["model.norm.weight"].detach().cpu()
        return out
