from __future__ import annotations

from typing import Dict

from transformers import PretrainedConfig
import torch

from .base import ShardingAdapter, LayerShard


class LlamaShardingAdapter(ShardingAdapter):
    """Sharding adapter for LLaMA-family models into vLLM-compatible keys."""

    def shard_embedding(self, hf_weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        if "model.embed_tokens.weight" in hf_weights:
            out["model.embed_tokens.weight"] = hf_weights["model.embed_tokens.weight"].detach().cpu()
        return out

    def shard_layer(self, layer_idx: int, hf_weights: Dict[str, torch.Tensor]) -> LayerShard:
        p = f"model.layers.{layer_idx}"
        out: Dict[str, torch.Tensor] = {}

        # Attention: fuse qkv
        q_key = f"{p}.self_attn.q_proj.weight"
        k_key = f"{p}.self_attn.k_proj.weight"
        v_key = f"{p}.self_attn.v_proj.weight"
        if all(k in hf_weights for k in (q_key, k_key, v_key)):
            out[f"{p}.self_attn.qkv_proj.weight"] = self.fuse_qkv(
                hf_weights[q_key].detach().cpu(),
                hf_weights[k_key].detach().cpu(),
                hf_weights[v_key].detach().cpu(),
            )
        o_key = f"{p}.self_attn.o_proj.weight"
        if o_key in hf_weights:
            out[f"{p}.self_attn.o_proj.weight"] = hf_weights[o_key].detach().cpu()

        # MLP: fuse gate_up
        gate_key = f"{p}.mlp.gate_proj.weight"
        up_key = f"{p}.mlp.up_proj.weight"
        down_key = f"{p}.mlp.down_proj.weight"
        if all(k in hf_weights for k in (gate_key, up_key)):
            out[f"{p}.mlp.gate_up_proj.weight"] = self.fuse_gate_up(
                hf_weights[gate_key].detach().cpu(),
                hf_weights[up_key].detach().cpu(),
            )
        if down_key in hf_weights:
            out[f"{p}.mlp.down_proj.weight"] = hf_weights[down_key].detach().cpu()

        # Layer norms
        in_ln = f"{p}.input_layernorm.weight"
        post_ln = f"{p}.post_attention_layernorm.weight"
        if in_ln in hf_weights:
            out[in_ln] = hf_weights[in_ln].detach().cpu()
        if post_ln in hf_weights:
            out[post_ln] = hf_weights[post_ln].detach().cpu()

        return LayerShard(weights=out)

    def shard_lm_head(self, hf_weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        # If tied, lm_head may not be present
        if "lm_head.weight" in hf_weights:
            out["lm_head.weight"] = hf_weights["lm_head.weight"].detach().cpu()
        return out

    def shard_model_norm(self, hf_weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        if "model.norm.weight" in hf_weights:
            out["model.norm.weight"] = hf_weights["model.norm.weight"].detach().cpu()
        return out 