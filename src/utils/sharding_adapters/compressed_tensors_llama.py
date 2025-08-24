from __future__ import annotations

from typing import Dict
from transformers import PretrainedConfig
import torch

from .base import ShardingAdapter, LayerShard


class CompressedTensorsLlamaShardingAdapter(ShardingAdapter):
    """
    Sharding adapter for compressed-tensors quantized Llama models.
    
    Key difference from standard LlamaShardingAdapter:
    - Does NOT fuse QKV projections (keeps q_proj, k_proj, v_proj separate)
    - Does NOT fuse gate_up projections (keeps gate_proj, up_proj separate)
    - Preserves quantization scale tensors
    - Normalizes weight orientation to match vLLM parameter expectations
    
    This is because vLLM's compressed-tensors loader expects the original
    weight names and will handle fusion during model loading.
    """

    def _maybe_transpose(self, weight: torch.Tensor, expected_out: int, expected_in: int) -> torch.Tensor:
        """Return weight transposed if it appears in [in, out] layout.
        If weight matches [expected_out, expected_in], return as is.
        """
        if weight.dim() == 2:
            h0, h1 = weight.shape
            # If it looks flipped, transpose once
            if h0 == expected_in and h1 == expected_out:
                return weight.t()
        return weight

    @staticmethod
    def _store(t: torch.Tensor) -> torch.Tensor:
        """Detach, move to CPU, and ensure contiguous memory for safetensors save."""
        return t.detach().cpu().contiguous()

    def shard_embedding(self, hf_weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        if "model.embed_tokens.weight" in hf_weights:
            out["model.embed_tokens.weight"] = self._store(hf_weights["model.embed_tokens.weight"])
        
        # Also preserve any quantization scales for embeddings
        if "model.embed_tokens.weight_scale" in hf_weights:
            out["model.embed_tokens.weight_scale"] = self._store(hf_weights["model.embed_tokens.weight_scale"])
        if "model.embed_tokens.input_scale" in hf_weights:
            out["model.embed_tokens.input_scale"] = self._store(hf_weights["model.embed_tokens.input_scale"])
        if "model.embed_tokens.input_zero_point" in hf_weights:
            out["model.embed_tokens.input_zero_point"] = self._store(hf_weights["model.embed_tokens.input_zero_point"])
        if "model.embed_tokens.azp_adj" in hf_weights:
            out["model.embed_tokens.azp_adj"] = self._store(hf_weights["model.embed_tokens.azp_adj"])
            
        return out

    def shard_layer(self, layer_idx: int, hf_weights: Dict[str, torch.Tensor]) -> LayerShard:
        p = f"model.layers.{layer_idx}"
        out: Dict[str, torch.Tensor] = {}

        hidden_size = int(getattr(self.config, "hidden_size", getattr(self.config, "n_embd", 0)))
        intermediate_size = int(getattr(self.config, "intermediate_size", max(1, hidden_size * 4)))

        # FUSE Q, K, V projections into qkv_proj for vLLM compatibility
        q_weight = hf_weights.get(f"{p}.self_attn.q_proj.weight")
        k_weight = hf_weights.get(f"{p}.self_attn.k_proj.weight")
        v_weight = hf_weights.get(f"{p}.self_attn.v_proj.weight")
        
        if q_weight is not None and k_weight is not None and v_weight is not None:
            # Ensure correct orientation before fusion
            q_weight = self._maybe_transpose(q_weight, expected_out=q_weight.shape[0], expected_in=hidden_size)
            k_weight = self._maybe_transpose(k_weight, expected_out=k_weight.shape[0], expected_in=hidden_size)
            v_weight = self._maybe_transpose(v_weight, expected_out=v_weight.shape[0], expected_in=hidden_size)
            
            # Fuse weights [q_out + k_out + v_out, hidden_size]
            fused_qkv = torch.cat([q_weight, k_weight, v_weight], dim=0)
            
            # vLLM expects [in_features, out_features] which is [hidden_size, 6144]
            # But we have [6144, hidden_size], so transpose
            out[f"{p}.self_attn.qkv_proj.weight"] = self._store(fused_qkv.t())
            
            # Fuse weight scales if present
            q_scale = hf_weights.get(f"{p}.self_attn.q_proj.weight_scale")
            k_scale = hf_weights.get(f"{p}.self_attn.k_proj.weight_scale")
            v_scale = hf_weights.get(f"{p}.self_attn.v_proj.weight_scale")
            if q_scale is not None and k_scale is not None and v_scale is not None:
                out[f"{p}.self_attn.qkv_proj.weight_scale"] = self._store(
                    torch.cat([q_scale, k_scale, v_scale], dim=0)
                )

        # O projection stays separate
        o_weight = hf_weights.get(f"{p}.self_attn.o_proj.weight")
        if o_weight is not None:
            o_weight = self._maybe_transpose(o_weight, expected_out=hidden_size, expected_in=hidden_size)
            out[f"{p}.self_attn.o_proj.weight"] = self._store(o_weight)
            
            # Copy o_proj scale
            o_scale = hf_weights.get(f"{p}.self_attn.o_proj.weight_scale")
            if o_scale is not None:
                out[f"{p}.self_attn.o_proj.weight_scale"] = self._store(o_scale)

        # FUSE gate and up projections into gate_up_proj for vLLM compatibility
        gate_weight = hf_weights.get(f"{p}.mlp.gate_proj.weight")
        up_weight = hf_weights.get(f"{p}.mlp.up_proj.weight")
        
        if gate_weight is not None and up_weight is not None:
            # Ensure correct orientation before fusion
            gate_weight = self._maybe_transpose(gate_weight, expected_out=intermediate_size, expected_in=hidden_size)
            up_weight = self._maybe_transpose(up_weight, expected_out=intermediate_size, expected_in=hidden_size)
            
            # Fuse weights [gate_out + up_out, hidden_size]
            fused_gate_up = torch.cat([gate_weight, up_weight], dim=0)
            
            # vLLM expects [in_features, out_features] which is [hidden_size, 28672]
            # But we have [28672, hidden_size], so transpose
            out[f"{p}.mlp.gate_up_proj.weight"] = self._store(fused_gate_up.t())
            
            # Fuse weight scales if present
            gate_scale = hf_weights.get(f"{p}.mlp.gate_proj.weight_scale")
            up_scale = hf_weights.get(f"{p}.mlp.up_proj.weight_scale")
            if gate_scale is not None and up_scale is not None:
                out[f"{p}.mlp.gate_up_proj.weight_scale"] = self._store(
                    torch.cat([gate_scale, up_scale], dim=0)
                )

        # Down projection stays separate but needs transposition
        down_weight = hf_weights.get(f"{p}.mlp.down_proj.weight")
        if down_weight is not None:
            # vLLM expects down_proj as [intermediate_size, hidden_size] not [hidden_size, intermediate_size]
            # So we transpose it to match
            down_weight = self._maybe_transpose(down_weight, expected_out=hidden_size, expected_in=intermediate_size)
            # But vLLM actually expects it as [intermediate_size, hidden_size] for compressed tensors
            # So transpose it back to the correct shape
            if down_weight.shape == (hidden_size, intermediate_size):
                down_weight = down_weight.t()
            out[f"{p}.mlp.down_proj.weight"] = self._store(down_weight)
            
            # Copy down_proj scale
            down_scale = hf_weights.get(f"{p}.mlp.down_proj.weight_scale")
            if down_scale is not None:
                out[f"{p}.mlp.down_proj.weight_scale"] = self._store(down_scale)

        # Layer norms
        for norm in ["input_layernorm", "post_attention_layernorm"]:
            weight_key = f"{p}.{norm}.weight"
            if weight_key in hf_weights:
                out[weight_key] = self._store(hf_weights[weight_key])

        return LayerShard(weights=out)

    def shard_lm_head(self, hf_weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        
        if "lm_head.weight" in hf_weights:
            out["lm_head.weight"] = self._store(hf_weights["lm_head.weight"])
        
        # Preserve quantization scales for lm_head
        for suffix in ("weight_scale", "input_scale", "input_zero_point", "azp_adj"):
            k = f"lm_head.{suffix}"
            if k in hf_weights:
                out[k] = self._store(hf_weights[k])
            
        return out

    def shard_model_norm(self, hf_weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        if "model.norm.weight" in hf_weights:
            out["model.norm.weight"] = self._store(hf_weights["model.norm.weight"])
        return out