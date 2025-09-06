"""
Layer sharding utilities for breaking down models into individual layer files.
This module provides functionality to shard models layer by layer with vLLM compatibility.
Refactored to support multiple model families via sharding adapters in
`src/utils/sharding_adapters`.
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from huggingface_hub import hf_hub_download, list_repo_files
from safetensors.torch import save_file
from transformers import AutoConfig, AutoTokenizer

from .sharding_adapters import get_adapter_for_config

# Ensure accelerated downloads from Hugging Face are enabled for this process
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


def get_model_safetensors_files(
    model_name: str, hf_token: Optional[str] = None
) -> List[str]:
    """List safetensors files for a HuggingFace model."""
    try:
        files = list_repo_files(model_name, token=hf_token)
        return [f for f in files if f.endswith(".safetensors")]
    except Exception as e:
        print(f"Error listing files for {model_name}: {e}")
        return []


def _s3_upload_file(local_path: Path, s3_uri: str) -> bool:
    try:
        import boto3  # type: ignore

        # Parse s3://bucket/key format without urlparse
        if s3_uri.startswith("s3://"):
            s3_path = s3_uri[5:]  # Remove "s3://"
            parts = s3_path.split("/", 1)
            if len(parts) == 2:
                bucket, key = parts
            else:
                bucket = parts[0]
                key = ""
        else:
            print(f"❌ Invalid S3 URI format: {s3_uri}")
            return False

        boto3.client("s3").upload_file(str(local_path), bucket, key)
        return True
    except ImportError:
        cmd = ["aws", "s3", "cp", str(local_path), s3_uri]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ aws cli upload failed: {result.stderr}")
            return False
        return True
    except Exception as e:
        print(f"❌ Failed to upload {local_path} to {s3_uri}: {e}")
        return False


def _join_s3_uri(base: str, *parts: str) -> str:
    return (
        base.rstrip("/")
        + "/"
        + "/".join(p.strip("/").replace("\\", "/") for p in parts)
    )


def load_safetensors(local_path: str) -> Dict[str, torch.Tensor]:
    from safetensors.torch import load_file as st_load

    return st_load(local_path)


def download_all_safetensors(
    model_name: str, hf_token: Optional[str], cache_dir: Optional[str]
) -> List[str]:
    paths: List[str] = []
    for file_path in get_model_safetensors_files(model_name, hf_token):
        local_path = hf_hub_download(
            repo_id=model_name, filename=file_path, token=hf_token, cache_dir=cache_dir
        )
        paths.append(local_path)
    return paths


def consolidate_weights_from_files(
    safetensor_paths: List[str],
) -> Dict[str, torch.Tensor]:
    consolidated: Dict[str, torch.Tensor] = {}
    for p in safetensor_paths:
        try:
            weights = load_safetensors(p)
            consolidated.update(weights)
        except Exception as e:
            print(f"Warning: could not read {p}: {e}")
    return consolidated


def shard_model_by_layers_safetensors(
    model_name: str,
    output_dir: str,
    hf_token: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Shard a model by individual layers using direct safetensors processing.
    Avoids loading the full model into memory. Family-specific mapping comes from
    adapters under `sharding_adapters`.
    """
    print(f"🔪 Starting safetensors-based layer sharding for {model_name}")

    # Output setup
    s3_base = os.getenv("S3_SHARDS_BASE")  # e.g., s3://tandemn-model-shards/shards
    use_s3 = bool(s3_base)
    model_dir_name = model_name.replace("/", "_")
    dest_root_uri = _join_s3_uri(s3_base, model_dir_name) if use_s3 else output_dir
    stream_upload = use_s3  # enable per-file upload when S3 is configured
    delete_local_after_upload = True  # free disk space once each file is uploaded

    if use_s3:
        tmpdir = tempfile.mkdtemp(prefix="shards-")
        output_path = Path(tmpdir)
        print(f"🔪 Using temporary directory: {output_path}")
    else:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

    # Config & tokenizer
    config = AutoConfig.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        token=hf_token,
        trust_remote_code=True,
    )
    print(f"🔧 Config: {config}")
    if config.model_type == "mistral" or config.model_name.lower() == "devstral":
        try:
            from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

            tokenizer = MistralTokenizer.from_hf_hub(model_name, token=hf_token)
            print(f"🔧 Mistral Tokenizer: {tokenizer}")
        except Exception as e:
            print(f"🔧 Error loading Mistral Tokenizer: {e}")
            tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
            print(f"🔧 Auto Tokenizer: {tokenizer}")
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            token=hf_token,
            trust_remote_code=True,
        )
    print(f"🔧 Tokenizer: {tokenizer}")

    adapter = get_adapter_for_config(config)
    print(f"🔧 Adapter: {adapter}")
    # Save config/tokenizer
    config_dir = output_path / "config"
    config.save_pretrained(config_dir)
    if config.model_type == "mistral" or config.model_name.lower() == "devstral":
        # mistral tokenizer does not have a save_pretrained method
        print("🔧 Mistral tokenizer does not have a save_pretrained method")
        try:
            tekken_path = hf_hub_download(model_name, "tekken.json", token=hf_token)
            import shutil

            shutil.copy(tekken_path, config_dir / "tekken.json")
        except Exception as e:
            print(f"🔧 Error downloading tekken.json: {e}")
        pass
    else:
        tokenizer.save_pretrained(config_dir)
    print(f"🔧 Config dir: {config_dir}")
    # Stream-upload config/tokenizer artifacts immediately (if S3)
    if stream_upload:
        uploaded_ct = 0
        for file in config_dir.rglob("*"):
            if file.is_file():
                relative_path = file.relative_to(output_path).as_posix()
                s3_uri = _join_s3_uri(dest_root_uri, relative_path)
                if _s3_upload_file(file, s3_uri):
                    uploaded_ct += 1
                    if delete_local_after_upload:
                        try:
                            file.unlink(missing_ok=True)
                        except Exception:
                            pass
                else:
                    print(f"❌ Failed to upload config file to {s3_uri}")
        print(f"✅ Stream-uploaded {uploaded_ct} config/tokenizer files")
    print(f"🔧 Stream-uploaded {uploaded_ct} config/tokenizer files")
    # Read all safetensors once and build an index in memory
    print("📥 Downloading safetensors...")
    paths = download_all_safetensors(model_name, hf_token, cache_dir)
    if not paths:
        raise ValueError(f"No safetensors files found for {model_name}")
    print(f"📦 Found {len(paths)} safetensors shards")

    print("📦 Consolidating weights index...")
    hf_weights = consolidate_weights_from_files(paths)

    num_layers = getattr(config, "num_hidden_layers", getattr(config, "n_layer", None))
    hidden_size = getattr(config, "hidden_size", getattr(config, "n_embd", None))
    if num_layers is None or hidden_size is None:
        raise ValueError("Config missing num_hidden_layers/hidden_size")

    metadata = {
        "model_name": model_name,
        "model_type": getattr(config, "model_type", "unknown"),
        "num_layers": int(num_layers),
        "hidden_size": int(hidden_size),
        "vocab_size": int(getattr(config, "vocab_size", 0)),
        "num_attention_heads": int(getattr(config, "num_attention_heads", 0)),
        "num_key_value_heads": int(getattr(config, "num_key_value_heads", 0)),
        "intermediate_size": int(getattr(config, "intermediate_size", hidden_size * 4)),
        "tie_word_embeddings": bool(getattr(config, "tie_word_embeddings", False)),
        "layer_components": [],
    }

    # Embedding
    try:
        emb = adapter.shard_embedding(hf_weights)
        if emb:
            emb_path = output_path / "embedding" / "layer.safetensors"
            emb_path.parent.mkdir(exist_ok=True)
            save_file(emb, str(emb_path))
            metadata["layer_components"].append(
                {
                    "type": "embedding",
                    "path": "embedding/layer.safetensors",
                    "component_name": "model.embed_tokens",
                }
            )
            print("✅ Saved embedding weights")
            # Stream-upload
            if stream_upload:
                s3_uri = _join_s3_uri(dest_root_uri, "embedding/layer.safetensors")
                if _s3_upload_file(emb_path, s3_uri):
                    if delete_local_after_upload:
                        try:
                            emb_path.unlink(missing_ok=True)
                        except Exception:
                            pass
                else:
                    print(f"❌ Failed to upload {emb_path} to {s3_uri}")
    except Exception as e:
        print(f"⚠️ Embedding export failed: {e}")

    # Layers
    layers_dir = output_path / "layers"
    layers_dir.mkdir(exist_ok=True)
    for i in range(int(num_layers)):
        try:
            print(f"🔪 Processing layer {i}/{num_layers - 1}...")
            layer_shard = adapter.shard_layer(i, hf_weights)
            if layer_shard.weights:
                layer_path = layers_dir / f"layer_{i}.safetensors"
                save_file(layer_shard.weights, str(layer_path))
                metadata["layer_components"].append(
                    {
                        "type": "transformer_layer",
                        "layer_index": i,
                        "path": f"layers/layer_{i}.safetensors",
                        "component_name": f"model.layers.{i}",
                    }
                )
                print(f"✅ Saved layer {i} with {len(layer_shard.weights)} weights")
                # Stream-upload
                if stream_upload:
                    s3_uri = _join_s3_uri(
                        dest_root_uri, f"layers/layer_{i}.safetensors"
                    )
                    if _s3_upload_file(layer_path, s3_uri):
                        if delete_local_after_upload:
                            try:
                                layer_path.unlink(missing_ok=True)
                            except Exception:
                                pass
                    else:
                        print(f"❌ Failed to upload {layer_path} to {s3_uri}")
            else:
                print(f"⚠️ No exportable weights for layer {i}")
        except Exception as e:
            print(f"❌ Error processing layer {i}: {e}")

    # LM head
    try:
        lmh = adapter.shard_lm_head(hf_weights)
        if lmh:
            lm_path = output_path / "lm_head" / "layer.safetensors"
            lm_path.parent.mkdir(exist_ok=True)
            save_file(lmh, str(lm_path))
            metadata["layer_components"].append(
                {
                    "type": "lm_head",
                    "path": "lm_head/layer.safetensors",
                    "component_name": "lm_head",
                }
            )
            print("✅ Saved lm_head weights")
            # Stream-upload
            if stream_upload:
                s3_uri = _join_s3_uri(dest_root_uri, "lm_head/layer.safetensors")
                if _s3_upload_file(lm_path, s3_uri):
                    if delete_local_after_upload:
                        try:
                            lm_path.unlink(missing_ok=True)
                        except Exception:
                            pass
                else:
                    print(f"❌ Failed to upload {lm_path} to {s3_uri}")
        else:
            print("ℹ️ lm_head is tied or absent; skipping explicit save")
    except Exception as e:
        print(f"⚠️ LM head export failed: {e}")

    # Final norm
    try:
        norm = adapter.shard_model_norm(hf_weights)
        if norm:
            norm_path = output_path / "norm" / "layer.safetensors"
            norm_path.parent.mkdir(exist_ok=True)
            save_file(norm, str(norm_path))
            metadata["layer_components"].append(
                {
                    "type": "norm",
                    "path": "norm/layer.safetensors",
                    "component_name": "model.norm",
                }
            )
            print("✅ Saved model.norm weights")
            # Stream-upload
            if stream_upload:
                s3_uri = _join_s3_uri(dest_root_uri, "norm/layer.safetensors")
                if _s3_upload_file(norm_path, s3_uri):
                    if delete_local_after_upload:
                        try:
                            norm_path.unlink(missing_ok=True)
                        except Exception:
                            pass
                else:
                    print(f"❌ Failed to upload {norm_path} to {s3_uri}")
    except Exception as e:
        print(f"⚠️ Model norm export failed: {e}")

    # Save metadata
    metadata_path = output_path / "layer_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Stream-upload metadata last
    if stream_upload:
        s3_uri = _join_s3_uri(dest_root_uri, "layer_metadata.json")
        if _s3_upload_file(metadata_path, s3_uri):
            if delete_local_after_upload:
                try:
                    metadata_path.unlink(missing_ok=True)
                except Exception:
                    pass
        else:
            print(f"❌ Failed to upload metadata to {s3_uri}")

    # If streaming uploads were used, skip the final bulk upload
    if use_s3 and not stream_upload:
        print(f"🔪 Uploading to S3: {dest_root_uri}")
        uploaded = 0
        for file in output_path.rglob("*"):
            if file.is_file():
                relative_path = file.relative_to(output_path).as_posix()
                s3_uri = _join_s3_uri(dest_root_uri, relative_path)
                if not _s3_upload_file(file, s3_uri):
                    print(f"❌ Failed to upload {file} to {s3_uri}")
                else:
                    uploaded += 1
        print(f"✅ Successfully uploaded {uploaded} files to S3")

    # Cleanup temp directory when using S3
    if use_s3:
        try:
            shutil.rmtree(output_path, ignore_errors=True)
            print(f"🔪 Removed temporary directory: {output_path}")
        except Exception as e:
            print(f"⚠️ Failed to remove temporary directory: {e}")
            pass

    print(
        f"✅ Successfully sharded model into {len(metadata['layer_components'])} components"
    )
    print(f"📁 Saved to: {output_path}")

    return {
        "status": "success",
        "output_dir": str(output_path),
        "metadata": metadata,
        "total_components": len(metadata["layer_components"]),
    }
