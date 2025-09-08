from __future__ import annotations

import time

import torch
from safetensors import safe_open
from transformers import PretrainedConfig

from .base import WeightLoadingAdapter


class QwenWeightLoadingAdapter(WeightLoadingAdapter):
    """Weight loading adapter for Qwen (Qwen2/Qwen3) models with AWQ support."""

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
        if self.is_awq:
            print("🔧 Qwen Weight Loader: AWQ quantization detected")

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
                    # Verify QK normalization weights are loaded for Qwen3
                    if "q_norm.weight" in key or "k_norm.weight" in key:
                        print(
                            f"✅ Loaded QK norm weight: {key} (shape: {self.all_weights[key].shape})"
                        )
        return self.all_weights

    def load_embedding(self, embedding_path):
        """Load embedding weights into the model."""
        if embedding_path.exists():
            self.all_weights.update(self.load_safetensors_file(embedding_path))
            print(f"✅ Loaded Qwen embedding weights from {embedding_path}")

            # Verify embedding keys for Qwen models
            if "model.embed_tokens.weight" in self.all_weights:
                embed_shape = self.all_weights["model.embed_tokens.weight"].shape
                print(f"   Embedding shape: {embed_shape}")
        return self.all_weights

    def load_lm_head(self, lm_head_path):
        """Load lm_head weights with AWQ and tied embeddings support."""
        if lm_head_path.exists():
            self.all_weights.update(self.load_safetensors_file(lm_head_path))
            print(f"✅ Loaded Qwen lm_head weights from {lm_head_path}")

            # Check what type of lm_head we loaded
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

            if "lm_head.weight" in self.all_weights:
                lm_head_shape = self.all_weights["lm_head.weight"].shape
                print(f"   lm_head.weight shape: {lm_head_shape}")

        else:
            print("ℹ️ No separate lm_head file found - handling tied embeddings")

            # Handle tied embeddings for Qwen models
            if "model.embed_tokens.weight" in self.all_weights:
                self.all_weights["lm_head.weight"] = self.all_weights[
                    "model.embed_tokens.weight"
                ]
                print(
                    "✅ Using tied embeddings - copied embed_tokens weights to lm_head"
                )
            elif "embed_tokens.weight" in self.all_weights:
                self.all_weights["lm_head.weight"] = self.all_weights[
                    "embed_tokens.weight"
                ]
                print(
                    "✅ Using tied embeddings - copied embed_tokens weights to lm_head"
                )
            else:
                print("⚠️ Warning: No embedding weights found for tied lm_head")

        return self.all_weights

    def load_model_norm(self, model_norm_path):
        """Load final layer norm weights (model.norm)."""
        if model_norm_path.exists():
            self.all_weights.update(self.load_safetensors_file(model_norm_path))
            print(f"✅ Loaded Qwen model.norm weights from {model_norm_path}")

            if "model.norm.weight" in self.all_weights:
                norm_shape = self.all_weights["model.norm.weight"].shape
                print(f"   model.norm.weight shape: {norm_shape}")
        else:
            print("⚠️ Warning: No model.norm file found")

        return self.all_weights

    def load_layer_weights(self, layer_idx, layer_path):
        """Load transformer layer weights with AWQ and QK normalization support."""
        if layer_path.exists():
            # Count weights before loading this layer
            weights_before = len(self.all_weights)

            self.all_weights.update(self.load_safetensors_file(layer_path))

            weights_after = len(self.all_weights)
            layer_weight_count = weights_after - weights_before

            print(
                f"✅ Loaded Qwen layer {layer_idx} weights from {layer_path} ({layer_weight_count} tensors)"
            )

            # Verify key components are loaded for this layer
            layer_prefix = f"model.layers.{layer_idx}"

            # Check for fused weights (qkv_proj, gate_up_proj from our sharding adapter)
            fused_qkv = f"{layer_prefix}.self_attn.qkv_proj"
            fused_gate_up = f"{layer_prefix}.mlp.gate_up_proj"

            if self.is_awq:
                # Check AWQ quantized weights
                awq_suffixes = [".qweight", ".qzeros", ".scales"]
                for fused_weight in [fused_qkv, fused_gate_up]:
                    for suffix in awq_suffixes:
                        key = f"{fused_weight}{suffix}"
                        if key in self.all_weights:
                            print(
                                f"   ✓ Found AWQ fused weight: {key} (shape: {self.all_weights[key].shape})"
                            )
            else:
                # Check float weights
                for fused_weight in [fused_qkv, fused_gate_up]:
                    key = f"{fused_weight}.weight"
                    if key in self.all_weights:
                        print(
                            f"   ✓ Found fused weight: {key} (shape: {self.all_weights[key].shape})"
                        )

            # Check QK normalization (specific to Qwen3)
            q_norm_key = f"{layer_prefix}.self_attn.q_norm.weight"
            k_norm_key = f"{layer_prefix}.self_attn.k_norm.weight"
            if q_norm_key in self.all_weights and k_norm_key in self.all_weights:
                print(f"   ✓ Found QK normalization weights for layer {layer_idx}")

        else:
            print(f"⚠️ Warning: Layer {layer_idx} not found at {layer_path}")

        return self.all_weights

    def loading_loop(self):
        """Main loading loop for Qwen models with comprehensive logging."""
        print(
            f"\n🔧 QWEN WEIGHT LOADING: Starting for {len(self.assigned_layers)} layers..."
        )
        print(f"   Model dir: {self.model_dir}")
        print(f"   AWQ quantization: {self.is_awq}")
        print(f"   Assigned layers: {self.assigned_layers}")

        # Phase 1: Load essential components
        self.load_embedding(self.model_dir / "embedding" / "layer.safetensors")
        self.load_lm_head(self.model_dir / "lm_head" / "layer.safetensors")
        self.load_model_norm(self.model_dir / "norm" / "layer.safetensors")

        # Phase 2: Load transformer layers
        for layer_idx in self.assigned_layers:
            self.load_layer_weights(
                layer_idx, self.model_dir / "layers" / f"layer_{layer_idx}.safetensors"
            )

        print(f"\n✅ PHASE 1 COMPLETE: Loaded {len(self.all_weights)} weight tensors")

        # Phase 3: Apply weights to model parameters with optimized GPU transfer
        print("\n🔧 PHASE 2: Applying weights to vLLM model...")

        applied_count = 0
        missing_params = []
        torch.cuda.synchronize()
        gpu_transfer_start_time = time.time()

        total_params = len(list(self.model.named_parameters()))
        print(f"   Target model has {total_params} parameters")

        for name, param in self.model.named_parameters():
            if name in self.all_weights:
                with torch.no_grad():
                    # Use pinned memory for faster GPU transfer
                    weight_tensor = self.all_weights[name]
                    pinned_tensor = weight_tensor.pin_memory()
                    param.copy_(pinned_tensor.to(param.device, non_blocking=True))
                    applied_count += 1
            else:
                # Check if this parameter should have been loaded
                is_expected_missing = False

                # Parameters from unassigned layers are expected to be missing
                for i in range(100):  # Assume max 100 layers
                    if f".layers.{i}." in name and i not in self.assigned_layers:
                        is_expected_missing = True
                        break

                if not is_expected_missing:
                    missing_params.append(name)

        # Wait for all GPU transfers to complete
        torch.cuda.synchronize()
        gpu_transfer_duration = time.time() - gpu_transfer_start_time

        # Calculate transfer statistics
        total_size_gb = sum(
            tensor.numel() * tensor.element_size()
            for tensor in self.all_weights.values()
        ) / (1024**3)

        gpu_bandwidth = (
            total_size_gb / gpu_transfer_duration if gpu_transfer_duration > 0 else 0
        )

        print("\n✅ QWEN WEIGHT LOADING COMPLETE!")
        print(f"   Applied: {applied_count}/{total_params} parameters")
        print(f"   Total weight tensors loaded: {len(self.all_weights)}")
        print(f"   Data transferred: {total_size_gb:.2f} GB")
        print(f"   GPU transfer time: {gpu_transfer_duration:.2f}s")
        print(f"   GPU bandwidth: {gpu_bandwidth:.1f} GB/s")

        if missing_params and len(missing_params) < 10:
            print(
                f"   ⚠️ Missing parameters (expected for unassigned layers): {missing_params}"
            )
        elif missing_params:
            print(
                f"   ⚠️ {len(missing_params)} missing parameters (expected for unassigned layers)"
            )

        return self.all_weights
