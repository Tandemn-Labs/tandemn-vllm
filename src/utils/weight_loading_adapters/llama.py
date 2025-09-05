from __future__ import annotations

import time

import torch
from safetensors import safe_open
from transformers import PretrainedConfig

from .base import WeightLoadingAdapter


class LlamaWeightLoadingAdapter(WeightLoadingAdapter):
    """Weight loading adapter for Llama models."""

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

    def load_safetensors_file(self, path):
        """Load safetensors in a dictionary of tensors."""
        if path.exists():
            with safe_open(str(path), framework="pt", device="cpu") as f:
                for key in f.keys():
                    self.all_weights[key] = f.get_tensor(key)
        return self.all_weights

    def load_embedding(self, embedding_path):
        """Load embedding weights into the model."""
        if embedding_path.exists():
            self.all_weights.update(self.load_safetensors_file(embedding_path))
            print(f"✅ Loaded embedding weights from {embedding_path}")
        return self.all_weights

    def load_lm_head(self, lm_head_path):
        """Load lm_head weights into the model."""
        if lm_head_path.exists():
            self.all_weights.update(self.load_safetensors_file(lm_head_path))
            print(f"✅ Loaded lm_head weights from {lm_head_path}")
        else:
            print("ℹ️ No separate lm_head file found - checking for tied embeddings")
            if "model.embed_tokens.weight" in self.all_weights:
                # Llama models with tied embeddings
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
        return self.all_weights

    def load_model_norm(self, model_norm_path):
        """Load model norm weights into the model."""
        if model_norm_path.exists():
            self.all_weights.update(self.load_safetensors_file(model_norm_path))
            print(f"✅ Loaded model norm weights from {model_norm_path}")
        return self.all_weights

    def load_layer_weights(self, layer_idx, layer_path):
        """Load layer weights into the model."""
        if layer_path.exists():
            self.all_weights.update(self.load_safetensors_file(layer_path))
            print(f"✅ Loaded layer {layer_idx} weights from {layer_path}")
        return self.all_weights

    def loading_loop(self):
        self.load_embedding(self.model_dir / "embedding" / "layer.safetensors")
        self.load_lm_head(self.model_dir / "lm_head" / "layer.safetensors")
        self.load_model_norm(self.model_dir / "norm" / "layer.safetensors")
        for layer_idx in self.assigned_layers:
            self.load_layer_weights(
                layer_idx, self.model_dir / "layers" / f"layer_{layer_idx}.safetensors"
            )

        print(f"✅ Loaded {len(self.all_weights)} weights from {self.model_dir}")

        applied_count = 0
        torch.cuda.synchronize()
        gpu_transfer_start_time = time.time()

        for name, param in self.model.named_parameters():
            if name in self.all_weights:
                with torch.no_grad():
                    pinned_tensor = self.all_weights[name].pin_memory()
                    param.copy_(pinned_tensor.to(param.device, non_blocking=True))
                    applied_count += 1

        print(f"✅ Applied {applied_count} weights to the model")
        print(f"   Total weights loaded: {len(self.all_weights)}")
        torch.cuda.synchronize()
        gpu_transfer_duration = time.time() - gpu_transfer_start_time
        gpu_bandwidth = (
            sum(
                tensor.numel() * tensor.element_size()
                for tensor in self.all_weights.values()
            )
            / (1024**3)
            / gpu_transfer_duration
            if gpu_transfer_duration > 0
            else 0
        )
        print(
            f"⚡ GPU transfer: {gpu_transfer_duration:.2f}s, {gpu_bandwidth:.1f} GB/s"
        )

        return self.all_weights
