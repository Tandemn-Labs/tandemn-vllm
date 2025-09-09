# src/utils/weight_loading_adapters/mixtral_moe.py

from __future__ import annotations

import time

import torch
from safetensors import safe_open
from transformers import PretrainedConfig

from .base import WeightLoadingAdapter


class MixtralMoEWeightLoadingAdapter(WeightLoadingAdapter):
    """Weight loading adapter for Mixtral MoE models with vLLM internal format."""

    def __init__(
        self,
        config: PretrainedConfig,
        model,
        assigned_layers,
        model_dir,
        quantization: str,
    ):
        super().__init__(config, model, assigned_layers, model_dir, quantization)
        self.all_weights = {}
        self.is_awq = self._is_awq(config)

        # MoE configuration
        self.num_local_experts = getattr(config, "num_local_experts", 8)
        self.num_experts_per_tok = getattr(config, "num_experts_per_tok", 2)

        if self.is_awq:
            print("🔧 Mixtral MoE Weight Loader: AWQ quantization detected")

        print(
            f"🏗️ Mixtral MoE Weight Loader: {self.num_local_experts} experts, {self.num_experts_per_tok} per token"
        )
        print("🏗️ Loading weights in vLLM internal format (w13_weight, w2_weight)")

    def _is_awq(self, config: PretrainedConfig) -> bool:
        """Check if model uses AWQ quantization."""
        qc = getattr(config, "quantization_config", None)
        if isinstance(qc, dict):
            return qc.get("quant_method") == "awq"
        return False

    def load_safetensors_file(self, path):
        """Load safetensors file into the weights dictionary."""
        if path.exists():
            with safe_open(str(path), framework="pt", device="cpu") as f:
                for key in f.keys():
                    self.all_weights[key] = f.get_tensor(key)
                    # Log vLLM internal MoE weights as they're loaded
                    if "block_sparse_moe" in key:
                        if "gate.weight" in key:
                            print(
                                f"✅ Loaded MoE gate: {key} (shape: {self.all_weights[key].shape})"
                            )
                        elif "experts.w13" in key:
                            print(
                                f"✅ Loaded fused gate+up: {key} (shape: {self.all_weights[key].shape})"
                            )
                        elif "experts.w2" in key:
                            print(
                                f"✅ Loaded down proj: {key} (shape: {self.all_weights[key].shape})"
                            )
        return self.all_weights

    def load_embedding(self, embedding_path):
        """Load embedding weights into the model."""
        if embedding_path.exists():
            self.all_weights.update(self.load_safetensors_file(embedding_path))
            print(f"✅ Loaded Mixtral MoE embedding weights from {embedding_path}")

            # Verify embedding keys
            if "model.embed_tokens.weight" in self.all_weights:
                embed_shape = self.all_weights["model.embed_tokens.weight"].shape
                print(f"   Embedding shape: {embed_shape}")
        return self.all_weights

    def load_lm_head(self, lm_head_path):
        """Load lm_head weights with AWQ support for Mixtral MoE."""
        if lm_head_path.exists():
            self.all_weights.update(self.load_safetensors_file(lm_head_path))
            print(f"✅ Loaded Mixtral MoE lm_head weights from {lm_head_path}")

            # Check AWQ lm_head weights
            if self.is_awq:
                awq_keys = [
                    k
                    for k in self.all_weights.keys()
                    if k.startswith("lm_head.")
                    and any(
                        suffix in k for suffix in [".qweight", ".qzeros", ".scales"]
                    )
                ]
                if awq_keys:
                    print(f"   Found AWQ lm_head weights: {len(awq_keys)} tensors")

            # Check standard lm_head weight
            if "lm_head.weight" in self.all_weights:
                lm_head_shape = self.all_weights["lm_head.weight"].shape
                print(f"   LM head shape: {lm_head_shape}")

        else:
            print("ℹ️ No separate lm_head file found - checking for tied embeddings")
            # Handle tied embeddings (rare for Mixtral but possible)
            if "model.embed_tokens.weight" in self.all_weights:
                self.all_weights["lm_head.weight"] = self.all_weights[
                    "model.embed_tokens.weight"
                ]
                print(
                    "✅ Using tied embeddings - copied embed_tokens weights to lm_head"
                )

        return self.all_weights

    def load_model_norm(self, model_norm_path):
        """Load model norm weights into the model."""
        if model_norm_path.exists():
            self.all_weights.update(self.load_safetensors_file(model_norm_path))
            print(f"✅ Loaded Mixtral MoE model norm weights from {model_norm_path}")

            # Verify model norm
            if "model.norm.weight" in self.all_weights:
                norm_shape = self.all_weights["model.norm.weight"].shape
                print(f"   Model norm shape: {norm_shape}")
        return self.all_weights

    def load_layer_weights(self, layer_idx, layer_path):
        """Load layer weights with vLLM internal format verification."""
        if layer_path.exists():
            weights_before = len(self.all_weights)
            self.all_weights.update(self.load_safetensors_file(layer_path))
            weights_after = len(self.all_weights)

            print(f"✅ Loaded layer {layer_idx} weights from {layer_path}")
            print(f"   Added {weights_after - weights_before} weight tensors")

            # Verify MoE structure for this layer
            self._verify_moe_layer_weights(layer_idx)

        return self.all_weights

    def _verify_moe_layer_weights(self, layer_idx: int):
        """Verify that MoE layer weights match vLLM internal format."""
        layer_prefix = f"model.layers.{layer_idx}"

        # Check attention weights
        attn_keys = [
            k
            for k in self.all_weights.keys()
            if k.startswith(f"{layer_prefix}.self_attn")
        ]
        if attn_keys:
            print(f"   Layer {layer_idx}: {len(attn_keys)} attention weights")

            # Check for fused QKV
            qkv_keys = [k for k in attn_keys if "qkv_proj" in k]
            if qkv_keys:
                print(
                    f"   Layer {layer_idx}: Found fused QKV projections ({len(qkv_keys)} tensors)"
                )

        # Check MoE weights - looking for vLLM internal format
        moe_prefix = f"{layer_prefix}.block_sparse_moe"
        moe_keys = [k for k in self.all_weights.keys() if k.startswith(moe_prefix)]

        if moe_keys:
            print(f"   Layer {layer_idx}: {len(moe_keys)} MoE weights")

            # Check for vLLM internal format
            w13_keys = [k for k in moe_keys if "experts.w13" in k]
            w2_keys = [k for k in moe_keys if "experts.w2" in k]

            if w13_keys:
                print(
                    f"   Layer {layer_idx}: Found w13 (gate+up) weights ({len(w13_keys)} tensors)"
                )
            if w2_keys:
                print(
                    f"   Layer {layer_idx}: Found w2 (down) weights ({len(w2_keys)} tensors)"
                )

            # Check MoE gate
            gate_key = f"{moe_prefix}.gate.weight"
            if gate_key in self.all_weights:
                gate_shape = self.all_weights[gate_key].shape
                print(f"   Layer {layer_idx}: MoE gate shape {gate_shape}")

        # Check layer norms
        norm_keys = [
            k
            for k in self.all_weights.keys()
            if k.startswith(layer_prefix) and "layernorm" in k
        ]
        if norm_keys:
            print(f"   Layer {layer_idx}: {len(norm_keys)} normalization weights")

    def loading_loop(self):
        """Main loading loop for Mixtral MoE models."""
        print("🔄 Starting Mixtral MoE weight loading...")

        # Load core components
        self.load_embedding(self.model_dir / "embedding" / "layer.safetensors")
        self.load_lm_head(self.model_dir / "lm_head" / "layer.safetensors")
        self.load_model_norm(self.model_dir / "norm" / "layer.safetensors")

        # Load assigned layers
        for layer_idx in self.assigned_layers:
            self.load_layer_weights(
                layer_idx, self.model_dir / "layers" / f"layer_{layer_idx}.safetensors"
            )

        print(
            f"✅ Loaded {len(self.all_weights)} total weight tensors from {self.model_dir}"
        )

        # Apply weights to model parameters
        applied_count = 0
        missing_in_model = []
        missing_in_weights = []

        torch.cuda.synchronize()
        gpu_transfer_start_time = time.time()

        # Get model parameter names for comparison
        model_param_names = set(name for name, _ in self.model.named_parameters())
        weight_names = set(self.all_weights.keys())

        # Apply loaded weights to model
        for name, param in self.model.named_parameters():
            if name in self.all_weights:
                with torch.no_grad():
                    # Use pin_memory for efficient GPU transfer
                    pinned_tensor = self.all_weights[name].pin_memory()
                    param.copy_(pinned_tensor.to(param.device, non_blocking=True))
                    applied_count += 1
            else:
                missing_in_weights.append(name)

        # Find weights that weren't applied
        for weight_name in weight_names:
            if weight_name not in model_param_names:
                missing_in_model.append(weight_name)

        print(f"✅ Applied {applied_count} weights to Mixtral MoE model")
        print(f"   Total weights loaded: {len(self.all_weights)}")
        print(f"   Model parameters: {len(model_param_names)}")

        if missing_in_weights:
            print(
                f"⚠️ Model parameters not found in weights ({len(missing_in_weights)}):"
            )
            for name in sorted(missing_in_weights)[:5]:  # Show first 5
                print(f"   - {name}")
            if len(missing_in_weights) > 5:
                print(f"   ... and {len(missing_in_weights) - 5} more")

        if missing_in_model:
            print(f"⚠️ Loaded weights not used by model ({len(missing_in_model)}):")
            for name in sorted(missing_in_model)[:5]:  # Show first 5
                print(f"   - {name}")
            if len(missing_in_model) > 5:
                print(f"   ... and {len(missing_in_model) - 5} more")

        # Calculate GPU transfer performance
        torch.cuda.synchronize()
        gpu_transfer_duration = time.time() - gpu_transfer_start_time

        if gpu_transfer_duration > 0:
            total_bytes = sum(
                tensor.numel() * tensor.element_size()
                for tensor in self.all_weights.values()
            )
            gpu_bandwidth = total_bytes / (1024**3) / gpu_transfer_duration
            print(
                f"⚡ GPU transfer: {gpu_transfer_duration:.2f}s, {gpu_bandwidth:.1f} GB/s"
            )

        return self.all_weights
