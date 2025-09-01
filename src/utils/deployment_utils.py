import asyncio
import concurrent.futures
import json
import os
import subprocess
import tempfile
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import aiofiles
import boto3  # type: ignore
import httpx
import torch
from fastapi import HTTPException
from safetensors import safe_open  # type: ignore

from src.config.settings import SERVER_PORT
from src.utils.db_utils import get_active_peers, get_peer_metrics
from src.utils.model_utils import distribute_layers_across_peers


def load_model_metadata(shard_folder: str):
    # Prefer reading layer_metadata.json from S3 if configured
    s3_base = os.getenv("S3_SHARDS_BASE", "")
    model_dir_name = Path(shard_folder).name if shard_folder else ""
    if s3_base:
        try:
            o = urlparse(s3_base)
            bucket = o.netloc
            base_key = o.path.lstrip("/")
            key = (
                f"{base_key.rstrip('/')}/{model_dir_name}/layer_metadata.json"
                if model_dir_name
                else f"{base_key.rstrip('/')}/layer_metadata.json"
            )
            try:
                client = boto3.client("s3")
                obj = client.get_object(Bucket=bucket, Key=key)
                body = obj["Body"].read()
                metadata = json.loads(body)
                print(
                    f"📊 Model metadata (S3): {metadata['num_layers']} layers, type: {metadata.get('model_type', 'unknown')}"
                )
                return metadata
            except ImportError:
                # Fallback to aws cli if boto3 is not installed
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp_path = Path(tmpdir) / "layer_metadata.json"
                    cmd = ["aws", "s3", "cp", f"s3://{bucket}/{key}", str(tmp_path)]
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode != 0:
                        raise HTTPException(
                            status_code=404,
                            detail=f"Failed to read layer_metadata.json from S3 at s3://{bucket}/{key}: {result.stderr.strip()}",
                        )
                    with open(tmp_path, "r") as f:
                        metadata = json.load(f)
                    print(
                        f"📊 Model metadata (S3 via aws cli): {metadata['num_layers']} layers, type: {metadata.get('model_type', 'unknown')}"
                    )
                    return metadata
        except HTTPException:
            raise
        except Exception as e:
            print(f"⚠️ Failed to read layer_metadata.json from S3 base {s3_base}: {e}")
            # Fall back to local if S3 read fails

    # Local fallback
    shard_path = Path(shard_folder)
    metadata_file = shard_path / "layer_metadata.json"
    if not metadata_file.exists() or not shard_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"layer_metadata.json or shard folder not found: {shard_folder}",
        )
    with open(metadata_file, "r") as f:
        metadata = json.load(f)
    print(
        f"📊 Model metadata: {metadata['num_layers']} layers, type: {metadata['model_type']}"
    )
    return metadata


async def get_peers_with_vram():
    active_peers = await get_active_peers()
    if len(active_peers) < 1:
        raise HTTPException(
            status_code=400, detail="No active peers available for deployment"
        )

    peers_vram = {}
    for peer_id in active_peers:
        try:
            metrics_history = await get_peer_metrics(peer_id, time_window=60)
            if metrics_history:
                latest_metrics = metrics_history[0]["metrics"]
                if "total_free_vram_gb" in latest_metrics:
                    peers_vram[peer_id] = latest_metrics["total_free_vram_gb"]
        except Exception as e:
            print(f"❌ Error getting metrics for peer {peer_id}: {e}")

    if not peers_vram:
        raise HTTPException(
            status_code=400, detail="No peers with VRAM information available"
        )

    print(f"👥 Found {len(peers_vram)} peers with VRAM data")
    return peers_vram


def create_distribution_plan(metadata, peers_vram, q_bits: int = 32):
    # Use metadata provided by sharding to avoid hardcoded defaults
    num_hidden_layers = int(metadata.get("num_layers"))
    hidden_size = int(metadata.get("hidden_size"))
    vocab_size = int(metadata.get("vocab_size", 32000))
    num_attention_heads = int(metadata.get("num_attention_heads", 32))
    num_key_value_heads = int(metadata.get("num_key_value_heads", num_attention_heads))
    intermediate_size = int(metadata.get("intermediate_size", hidden_size * 4))

    config = {
        "num_hidden_layers": num_hidden_layers,
        "hidden_size": hidden_size,
        "vocab_size": vocab_size,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "intermediate_size": intermediate_size,
    }

    distribution_plan = distribute_layers_across_peers(
        config=config,
        peers_vram=peers_vram,
        q_bits=q_bits,  # Allow caller to choose precision for VRAM estimate
    )
    print("📋 Distribution plan created:")
    print(f"   • Model can fit: {distribution_plan['can_fit_model']}")
    print(
        f"   • Total VRAM needed: {distribution_plan['model_info']['total_model_vram_gb']:.1f}GB"
    )
    print(f"   • Available VRAM: {distribution_plan['total_available_vram_gb']:.1f}GB")
    print(f"   • Peers involved: {len(distribution_plan['distribution'])}")
    for peer_id, peer_info in distribution_plan["distribution"].items():
        print(
            f"   • {peer_id}: {peer_info['assigned_layers']} layers, {peer_info['estimated_vram_usage']:.1f}GB"
        )

    if not distribution_plan["can_fit_model"]:
        raise HTTPException(
            status_code=400,
            detail=f"Model cannot fit in available VRAM. Need {distribution_plan['model_info']['total_model_vram_gb']:.1f}GB, have {distribution_plan['total_available_vram_gb']:.1f}GB",
        )
    return distribution_plan


# The distribution_plan returned by distribute_layers_across_peers looks like this:
#
# {
#     "distribution": {
#         "peer_id_1": {
#             "assigned_layers": int,            # Number of layers assigned to this peer
#             "handles_embeddings": bool,        # Whether this peer loads the embedding layer
#             "available_vram_gb": float,        # VRAM available to this peer (GB)
#             "estimated_vram_usage": float,     # Estimated VRAM used by this peer (GB)
#             "vram_utilization_percent": float  # % of VRAM utilized by assigned layers
#         },
#         "peer_id_2": { ... },
#         ...
#     },
#     "model_info": {
#         "total_layers": int,                   # Total number of model layers
#         "total_assigned_layers": int,          # Total layers assigned across all peers
#         "vram_per_layer_gb": float,            # VRAM required per layer (GB)
#         "embedding_vram_gb": float,            # VRAM required for embeddings (GB)
#         "total_model_vram_gb": float           # Total VRAM required for the full model (GB)
#     },
#     "can_fit_model": bool,                     # True if the model fits in available VRAM
#     "remaining_layers": int,                   # Layers left unassigned (should be 0 if can_fit_model)
#     "total_peers": int,                        # Number of peers considered
#     "utilized_peers": int,                     # Number of peers actually assigned layers
#     "total_available_vram_gb": float           # Total VRAM available across all peers (GB)
# }


def create_deployment_instructions(request, distribution_plan, peer_table, SERVER_IP):
    """
    Create deployment instructions for each peer based on the distribution plan.
    Includes optional quantization, dtype and qbits to control vLLM init.
    """
    deployment_instructions = {}
    peer_list = list(distribution_plan["distribution"].keys())

    # Determine base URL for model files: prefer S3 if configured
    model_dir_name = Path(
        getattr(request, "shard_folder", "")
    ).name or request.model_name.replace("/", "_")
    s3_base = os.getenv("S3_SHARDS_BASE")  # e.g., s3://tandemn-model-shards/shards
    prefer_s3 = bool(s3_base)

    if prefer_s3:
        base_url = f"{s3_base.rstrip('/')}/{model_dir_name}"
    else:
        base_url = f"http://{SERVER_IP}:{SERVER_PORT}/download_file/{request.model_name.replace('/', '_')}"

    # Check if model has tied embeddings (need to read metadata)
    try:
        # Try to get tie_word_embeddings from model metadata
        metadata = load_model_metadata(getattr(request, "shard_folder", ""))
        tie_word_embeddings = metadata.get("tie_word_embeddings", False)
    except Exception as _:
        # Default to False if we can't read metadata
        tie_word_embeddings = False

    for i, (peer_id, peer_info) in enumerate(distribution_plan["distribution"].items()):
        is_first_peer = i == 0
        is_last_peer = i == len(peer_list) - 1

        assigned_layers = list(
            range(
                sum(
                    p["assigned_layers"]
                    for p in list(distribution_plan["distribution"].values())[:i]
                ),
                sum(
                    p["assigned_layers"]
                    for p in list(distribution_plan["distribution"].values())[: i + 1]
                ),
            )
        )

        required_files = []
        required_files.extend(
            [
                "config/config.json",
                "layer_metadata.json",
            ]
        )

        # Optional tokenizer/config artifacts that improve tokenizer initialization
        optional_files = [
            "config/tokenizer.json",  # HF consolidated tokenizer
            "config/tokenizer.model",  # SentencePiece for LLaMA-style models
            "config/tokenizer_config.json",
            "config/special_tokens_map.json",
            "config/generation_config.json",
            "config/added_tokens.json",
            "config/vocab.json",  # BPE vocab
            "config/merges.txt",  # BPE merges
        ]

        if is_first_peer:
            required_files.append("embedding/layer.safetensors")

        if is_last_peer:
            # Always need norm for last peer
            required_files.extend(["norm/layer.safetensors"])

            # Check if lm_head/layer.safetensors exists in the model folder (S3 or local)
            # If it exists, embeddings are NOT tied, regardless of config
            lm_head_exists = False
            if prefer_s3:
                # S3: check if lm_head/layer.safetensors exists in the S3 folder
                try:
                    o = urlparse(s3_base)
                    bucket = o.netloc
                    base_key = o.path.lstrip("/")
                    lm_head_key = f"{base_key.rstrip('/')}/{model_dir_name}/lm_head/layer.safetensors"
                    client = boto3.client("s3")
                    response = client.list_objects_v2(Bucket=bucket, Prefix=lm_head_key)
                    if "Contents" in response:
                        for obj in response["Contents"]:
                            if obj["Key"] == lm_head_key:
                                lm_head_exists = True
                                print(
                                    f"🔍 S3: Found lm_head/layer.safetensors at {lm_head_key}"
                                )
                                break
                except Exception as e:
                    print(f"⚠️ Could not check S3 for lm_head/layer.safetensors: {e}")
            else:
                # Local: check if lm_head/layer.safetensors exists in the local folder
                local_lm_head_path = (
                    Path(getattr(request, "shard_folder", ""))
                    / "lm_head"
                    / "layer.safetensors"
                )
                if local_lm_head_path.exists():
                    lm_head_exists = True
                    print(
                        f"🔍 Local: Found lm_head/layer.safetensors at {local_lm_head_path}"
                    )
            # # First check tie_word_embeddings from metadata (most reliable)
            # if not tie_word_embeddings:
            #     # Model config says weights are NOT tied, so lm_head should exist
            #     lm_head_exists = True
            #     print(f"ℹ️ Model has tie_word_embeddings=False, expecting separate lm_head")
            # else:
            #     # Weights are tied, no separate lm_head
            #     lm_head_exists = False
            #     print(f"ℹ️ Model has tie_word_embeddings=True, using embeddings for lm_head")

            if lm_head_exists:
                # Model has separate lm_head weights
                required_files.append("lm_head/layer.safetensors")
                print(f"✅ Last peer {peer_id} will download lm_head weights")
            else:
                # Model has tied embeddings - last peer needs embedding file too
                if "embedding/layer.safetensors" not in required_files:
                    required_files.append("embedding/layer.safetensors")
                    print(
                        f"ℹ️ Last peer {peer_id} will download embeddings (tied weights)"
                    )

        for layer_idx in assigned_layers:
            required_files.append(f"layers/layer_{layer_idx}.safetensors")

        next_peer_ticket = peer_list[i + 1] if (i + 1) < len(peer_list) else None

        deployment_instructions[peer_id] = {
            "model_name": request.model_name,
            "assigned_layers": assigned_layers,
            "is_first_peer": is_first_peer,
            "is_last_peer": is_last_peer,
            "required_files": required_files,
            # Attempt to fetch optional files but do not fail if missing
            "optional_files": optional_files,
            "server_download_url": base_url,
            "vram_allocation": peer_info,
            "next_peer_ticket": next_peer_ticket,
            "pipeline": peer_list,
            # NEW: thread quantization/dtype/qbits end-to-end (optional)
            "quantization": getattr(request, "quantization", None),
            "dtype": getattr(request, "dtype", None),
            "qbits": getattr(request, "qbits", None),
            "tie_word_embeddings": tie_word_embeddings,  # Pass this info to peers
        }
    return deployment_instructions


# ============================================================================
# MODEL DOWNLOADING FUNCTIONS
# ============================================================================


async def download_file(
    url: str, local_path: Path, chunk_size: int = 16 * 1024 * 1024
) -> bool:
    """
    Download a file from server with progress tracking and resume capability.

    Args:
        url: Download URL (supports http(s):// and s3://)
        local_path: Local file path to save to
        chunk_size: Download chunk size in bytes (default: 16MB)

    Returns:
        bool: True if download successful, False otherwise
    """
    try:
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Skip download if file already exists and has content
        if local_path.exists() and local_path.stat().st_size > 0:
            # Optional: Verify file size matches server (HEAD request) for HTTP(S) only
            if url.startswith("http://") or url.startswith("https://"):
                try:
                    async with httpx.AsyncClient() as client:
                        head_response = await client.head(url)
                        if head_response.status_code == 200:
                            remote_size = int(
                                head_response.headers.get("content-length", 0)
                            )
                            local_size = local_path.stat().st_size

                            if local_size == remote_size:
                                print(
                                    f"✅ File already exists with correct size, skipping: {local_path.name} ({local_size:,} bytes)"
                                )
                                return True
                            else:
                                print(
                                    f"⚠️ File exists but size mismatch - redownloading: local={local_size:,}, remote={remote_size:,}"
                                )
                        else:
                            print(
                                f"⚠️ Could not verify remote file size (HEAD {head_response.status_code}), downloading anyway"
                            )
                except Exception as e:
                    print(f"⚠️ HEAD request failed: {e}, proceeding with download")
                    # If HEAD fails, just check local file exists and has content
                    print(
                        f"✅ File exists locally, assuming valid: {local_path.name} ({local_path.stat().st_size:,} bytes)"
                    )
                    return True
            else:
                # For s3, if file exists, assume valid
                print(
                    f"✅ File exists locally, assuming valid (s3): {local_path.name} ({local_path.stat().st_size:,} bytes)"
                )
                return True

        # S3 path: use boto3 if available; fallback to aws cli
        if url.startswith("s3://"):
            try:
                from urllib.parse import urlparse

                o = urlparse(url)
                # urlparse yields scheme='s3', netloc='bucket', path='/key'
                bucket = o.netloc
                key = o.path.lstrip("/")
                try:
                    import boto3  # type: ignore

                    # Run S3 download in thread pool to avoid blocking event loop
                    loop = asyncio.get_running_loop()
                    
                    def download_s3_file():
                        client = boto3.client("s3")
                        client.download_file(bucket, key, str(local_path))
                        return True
                    
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                        success = await loop.run_in_executor(executor, download_s3_file)
                    
                    if success:
                        print(f"✅ Downloaded {local_path.name} from s3://{bucket}/{key}")
                        return True
                    else:
                        return False
                        
                except ImportError:
                    print(
                        "ℹ️ boto3 not installed; falling back to aws cli for S3 download"
                    )
                    
                    # Run aws cli in thread pool to avoid blocking event loop
                    loop = asyncio.get_running_loop()
                    
                    def run_aws_cli():
                        cmd = ["aws", "s3", "cp", f"s3://{bucket}/{key}", str(local_path)]
                        result = subprocess.run(cmd, capture_output=True, text=True)
                        return result.returncode == 0, result.stderr
                    
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                        success, stderr = await loop.run_in_executor(executor, run_aws_cli)
                    
                    if success:
                        print(
                            f"✅ Downloaded {local_path.name} via aws cli from s3://{bucket}/{key}"
                        )
                        return True
                    else:
                        print(f"❌ aws cli download failed: {stderr}")
                        return False
            except Exception as e:
                print(f"❌ Failed to download from S3 URL {url}: {e}")
                return False

        # HTTP(S) path: stream with httpx
        async with httpx.AsyncClient() as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()

                total_size = int(response.headers.get("content-length", 0))
                downloaded = 0

                async with aiofiles.open(local_path, "wb") as f:
                    async for chunk in response.aiter_bytes(chunk_size):
                        await f.write(chunk)
                        downloaded += len(chunk)

                        # Progress logging (every 100MB or if total < 100MB for minimal I/O overhead)
                        if total_size > 0 and (
                            downloaded % (100 * 1024 * 1024) == 0
                            or downloaded == total_size
                        ):
                            progress = (downloaded / total_size) * 100
                            print(
                                f"   📥 Downloading {local_path.name}: {progress:.1f}% ({downloaded:,}/{total_size:,} bytes)"
                            )

        print(f"✅ Downloaded {local_path.name} ({downloaded:,} bytes)")
        return True

    except Exception as e:
        print(f"❌ Failed to download {url}: {e}")
        return False


async def download_model_files(instructions: Dict[str, Any]) -> tuple[bool, Path]:
    """
    Download all required model files based on deployment instructions.

    Args:
        instructions: Deployment instructions containing file list and URLs

    Returns:
        tuple[bool, Path]: (success, model_directory_path)
    """
    try:
        # Create local model directory
        model_dir = Path(f"./deployed_models/{instructions['model_name']}")
        model_dir.mkdir(parents=True, exist_ok=True)

        # Download required files
        base_url = instructions["server_download_url"]
        successful_downloads = 0
        total_files = len(instructions["required_files"])

        print(f"📥 Starting download of {total_files} required files...")

        for file_path in instructions["required_files"]:
            file_url = f"{base_url}/{file_path}"
            local_file_path = model_dir / file_path

            print(f"📥 Downloading {file_path}...")
            if await download_file(file_url, local_file_path):
                successful_downloads += 1
            else:
                print(f"❌ Failed to download {file_path}")

        # Attempt optional files without failing the deployment
        optional_files = instructions.get("optional_files", [])
        if optional_files:
            print(
                f"📥 Attempting to download up to {len(optional_files)} optional files..."
            )
        optional_success = 0
        for file_path in optional_files:
            file_url = f"{base_url}/{file_path}"
            local_file_path = model_dir / file_path
            print(f"📥 (optional) Downloading {file_path}...")
            if await download_file(file_url, local_file_path):
                optional_success += 1
            else:
                print(f"ℹ️ Optional file not available: {file_path}")
        if optional_files:
            print(
                f"✅ Optional files downloaded: {optional_success}/{len(optional_files)}"
            )

        # Validate tokenizer presence: at least one of the common tokenizer assets must exist
        tokenizer_candidates = [
            model_dir / "config/tokenizer.json",
            model_dir / "config/tokenizer.model",
            model_dir / "config/vocab.json",
        ]
        if not any(p.exists() and p.stat().st_size > 0 for p in tokenizer_candidates):
            print(
                "⚠️ No tokenizer asset found (tokenizer.json/tokenizer.model/vocab.json). Tokenizer init may fail."
            )

        print(f"✅ All {total_files} required files downloaded successfully")
        return True, model_dir

    except Exception as e:
        print(f"❌ Error downloading model files: {e}")
        return False, Path(".")


# ============================================================================
# MODEL LOADING FUNCTIONS
# ============================================================================


def create_dynamic_vllm_model(
    model_dir: str,
    assigned_layers: List[int],
    quantization: Optional[str] = None,
    dtype: Optional[str] = None,
):
    """
    Create vLLM model with only assigned layers loaded by monkey-patching make_layers.

    Args:
        model_dir: Directory containing model files
        assigned_layers: List of layer indices to load
        quantization: Optional vLLM quantization method (e.g., "bitsandbytes", "awq", "gptq")
        dtype: Optional dtype for activations/weights ("float16", "bfloat16", "float32", "auto")
    """

    # Import vLLM lazily to avoid forcing it on the central server process
    from vllm import LLM
    import time

    # STEP 1: Monkey-patch vLLM's make_layers function (Prime Intellect's key insight)
    def _selective_make_layers(num_hidden_layers: int, layer_fn, prefix: str):
        """Custom make_layers that creates real layers only for assigned indices."""

        from vllm.model_executor.models.utils import (
            PPMissingLayer,
            maybe_offload_to_cpu,
        )

        start_layer = min(assigned_layers) if assigned_layers else 0
        end_layer = max(assigned_layers) + 1 if assigned_layers else 0

        modules = []
        for i in range(num_hidden_layers):
            if i in assigned_layers:
                # Create real layer
                layer = layer_fn(prefix=f"{prefix}.{i}")
                modules.append(maybe_offload_to_cpu(layer))
                print(f"  Created REAL layer {i}")
            else:
                # Create passthrough layer (Prime Intellect's memory optimization)
                modules.append(PPMissingLayer())
                print(f"  Created PPMissingLayer for layer {i}")

        return start_layer, end_layer, torch.nn.ModuleList(modules)

    # Apply the monkey patch
    import vllm.model_executor.models.utils as model_utils

    original_make_layers = model_utils.make_layers
    model_utils.make_layers = _selective_make_layers

    try:
        # STEP 2: Create vLLM model (will use our patched make_layers)
        llm = LLM(
            model=model_dir,
            tensor_parallel_size=1,
            enforce_eager=True,  # Required for custom layer loading
            max_model_len=120,  # Small for demo
            disable_log_stats=True,
            skip_tokenizer_init=False,
            gpu_memory_utilization=0.98,  # Use much less memory
            use_v2_block_manager=False,  # Force legacy engine to avoid v1 memory pre-allocation
            load_format="dummy",  # ← this is the magic flag
            dtype=(dtype or "float16"),
            quantization=quantization,  # vLLM will select kernels/flows accordingly
        )

        # STEP 3: Load weights for assigned layers + essential components (replicating selective_layer_loading_fixed.py)
        try:
            # model_dir may be a config/ subfolder; shards root is its parent
            shards_root = Path(model_dir).resolve().parent

            # Navigate to underlying torch model
            model_runner = llm.llm_engine.model_executor.driver_worker.model_runner
            model = model_runner.model

            # Collect all weights we need to load (matching selective_layer_loading_fixed.py logic)
            cpu_loading_start_time = time.time()
            all_weights = {}
            
            # Helper: load all tensors from a safetensors file
            def load_safetensors_file(path: Path) -> Dict[str, torch.Tensor]:
                """Load all tensors from a safetensors file."""
                tensors = {}
                if path.exists():
                    with safe_open(str(path), framework="pt", device="cpu") as f:
                        for key in f.keys():
                            tensors[key] = f.get_tensor(key)
                return tensors

            # Load embedding (always needed for first peer)
            embed_path = shards_root / "embedding" / "layer.safetensors"
            if embed_path.exists():
                embed_weights = load_safetensors_file(embed_path)
                all_weights.update(embed_weights)
                print(f"✅ Loaded embedding weights from {embed_path}")

            # Load lm_head (always needed for last peer)
            lm_head_path = shards_root / "lm_head" / "layer.safetensors"
            if lm_head_path.exists():
                lm_head_weights = load_safetensors_file(lm_head_path)
                all_weights.update(lm_head_weights)
                print(f"✅ Loaded lm_head weights from {lm_head_path}")
            else:
                # Handle tied embeddings case - copy embed_tokens weights to lm_head
                print("ℹ️ No separate lm_head file found - checking for tied embeddings")
                if "model.embed_tokens.weight" in all_weights:
                    # Llama models with tied embeddings
                    all_weights["lm_head.weight"] = all_weights[
                        "model.embed_tokens.weight"
                    ]
                    print(
                        "✅ Using tied embeddings - copied embed_tokens weights to lm_head"
                    )
                if "embed_tokens.weight" in all_weights:
                    # Alternative naming
                    all_weights["lm_head.weight"] = all_weights["embed_tokens.weight"]
                    print(
                        "✅ Using tied embeddings - copied embed_tokens weights to lm_head"
                    )

            # Load model.norm (always needed for last peer)
            norm_path = shards_root / "norm" / "layer.safetensors"
            if norm_path.exists():
                norm_weights = load_safetensors_file(norm_path)
                all_weights.update(norm_weights)
                print(f"✅ Loaded model.norm weights from {norm_path}")

            # Load only specified transformer layers
            for layer_idx in assigned_layers:
                layer_path = shards_root / "layers" / f"layer_{layer_idx}.safetensors"
                if layer_path.exists():
                    layer_weights = load_safetensors_file(layer_path)
                    all_weights.update(layer_weights)
                    print(f"✅ Loaded layer {layer_idx} weights from {layer_path}")
                else:
                    print(f"⚠️ Warning: Layer {layer_idx} not found at {layer_path}")

            cpu_loading_duration = time.time() - cpu_loading_start_time
            total_cpu_size_gb = sum(tensor.numel() * tensor.element_size() for tensor in all_weights.values()) / (1024**3)
            print(f"⏱️ CPU loading duration: {cpu_loading_duration:.3f}s")

            # STEP 4: Apply the loaded weights to the model (exact replication of selective_layer_loading_fixed.py)
            print("\n🔧 APPLYING LOADED WEIGHTS TO MODEL...")
            print(f"   Total weights loaded: {len(all_weights)}")

            applied_count = 0
            missing_params = []


            # Time the GPU transfer
            torch.cuda.synchronize()
            gpu_transfer_start_time = time.time()

            for name, param in model.named_parameters():
                if name in all_weights:
                    with torch.no_grad():
                        pinned_tensor = all_weights[name].pin_memory()
                        param.copy_(pinned_tensor.to(param.device, non_blocking=True))
                        applied_count += 1
                # else:
                #     # Check if this parameter belongs to a layer we should have loaded
                #     is_expected_missing = False

                #     # Check if it's from an unassigned layer
                #     for i in range(100):  # Assuming max 100 layers
                #         if f".layers.{i}." in name and i not in assigned_layers:
                #             is_expected_missing = True
                #             break

                #     if not is_expected_missing:
                #         missing_params.append(name)

            # Wait for all transfers to complete
            torch.cuda.synchronize()
            gpu_transfer_duration = time.time() - gpu_transfer_start_time
            gpu_bandwidth = total_cpu_size_gb / gpu_transfer_duration if gpu_transfer_duration > 0 else 0
            print(f"🔧 GPU transfer completed: {applied_count} parameters in {gpu_transfer_duration:.2f}s")
            print(f"⚡ GPU transfer: {gpu_transfer_duration:.2f}s, {gpu_bandwidth:.1f} GB/s")

            print(
                f"✅ Applied weights to {applied_count}/{len(list(model.named_parameters()))} parameters"
            )

            if (
                missing_params and len(missing_params) < 20
            ):  # Only show first 20 to avoid spam
                print(
                    f"⚠️ Parameters without loaded weights (first 20): {missing_params[:20]}"
                )
            elif missing_params:
                print(
                    f"⚠️ {len(missing_params)} parameters without loaded weights (expected for unassigned layers)"
                )

        except Exception as e:
            print(f"⚠️ Weight loading/injection failed: {e}")
            import traceback

            traceback.print_exc()

        print("\n✅ Successfully created vLLM model with selective layers!")
        print(f"   Our monkey-patch created real layers for: {assigned_layers}")
        print(f"   Quantization: {quantization} | dtype: {dtype or 'float16'}")
        print("   All other layers are PPMissingLayer (passthrough)")

        return llm

    finally:
        # Restore original function
        model_utils.make_layers = original_make_layers


def create_async_vllm_engine_with_selective_layers(
    model_dir: str,
    assigned_layers: List[int],
    *,
    quantization: Optional[str] = None,
    dtype: Optional[str] = None,
    max_num_seqs: int = 1,
    max_num_batched_tokens: int = 2048,
):
    """
    Create an in-process v0 AsyncLLMEngine with selective layers loaded.
    Preserves your monkey-patch strategy and manually injects weights.
    """

    # Import vLLM lazily to avoid forcing it on the central server process
    from vllm.config import KVTransferConfig
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine  # v0 async engine
    from vllm.usage.usage_lib import UsageContext

    ktc = KVTransferConfig(
        kv_connector="LMCacheConnector",  # v0 connector
        kv_role="kv_both",  # single process does store+retrieve
        # For multi-process prefill/decode, use kv_producer/kv_consumer below
    )

    # STEP 1: Monkey-patch make_layers so only assigned layers are real.
    def _selective_make_layers(num_hidden_layers: int, layer_fn, prefix: str):
        from vllm.model_executor.models.utils import (
            PPMissingLayer,
            maybe_offload_to_cpu,
        )

        modules = []
        for i in range(num_hidden_layers):
            if i in assigned_layers:
                layer = layer_fn(prefix=f"{prefix}.{i}")
                modules.append(maybe_offload_to_cpu(layer))
                print(f"  [Async] Created REAL layer {i}")
            else:
                modules.append(PPMissingLayer())
        return (
            min(assigned_layers) if assigned_layers else 0,
            (max(assigned_layers) + 1) if assigned_layers else 0,
            torch.nn.ModuleList(modules),
        )

    import vllm.model_executor.models.utils as model_utils

    original_make_layers = model_utils.make_layers
    model_utils.make_layers = _selective_make_layers

    try:
        # STEP 2: Build AsyncEngineArgs mirroring your LLM init
        # engine_args = AsyncEngineArgs(
        #     model=model_dir,
        #     tensor_parallel_size=1,
        #     enforce_eager=True,
        #     load_format="dummy",
        #     max_model_len=2048,  # small for demo
        #     disable_log_stats=False,
        #     gpu_memory_utilization=0.8,
        #     skip_tokenizer_init=False,
        #     max_num_seqs=max_num_seqs,
        #     max_num_batched_tokens=max_num_batched_tokens,
        #     quantization=quantization,
        #     dtype=dtype or "float16",  # activations
        #     block_size=16,
        #     kv_transfer_config=ktc,
        #     enable_chunked_prefill=True,  # recommended for v0 + LMCache
        # )
        engine_args = AsyncEngineArgs(
            model=model_dir,
            tensor_parallel_size=1,
            enforce_eager=True,
            load_format="dummy",
            max_model_len=16400,  # Changed from 2048 to enable 128K context
            disable_log_stats=False,
            gpu_memory_utilization=0.9,  # Increased from 0.8 for more memory
            skip_tokenizer_init=False,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=16400,
            quantization=quantization,
            dtype=dtype or "float16",  # activations
            block_size=16,
            kv_transfer_config=ktc,
            enable_chunked_prefill=False,  # recommended for v0 + LMCache
            rope_scaling={  # Add RoPE scaling configuration
                "rope_type": "llama3",
                "factor": 32.0,
                "high_freq_factor": 4.0,
                "low_freq_factor": 1.0,
                "original_max_position_embeddings": 8192,
            },
            rope_theta=500000.0,
        )
        # STEP 3: Create AsyncLLMEngine (v0 path, in-process)
        async_engine = AsyncLLMEngine.from_engine_args(
            engine_args=engine_args,
            usage_context=UsageContext.ENGINE_CONTEXT,
            start_engine_loop=True,
        )

        # STEP 4: Inject weights into the underlying model
        shards_root = Path(model_dir).resolve().parent
        # this is where the model instance is stored
        # in sync vllm we had the model in llm.llm_engine.model_executor.driver_worker.model_runner
        # so a slight change.
        model = async_engine.engine.model_executor.driver_worker.model_runner.model

        # Collect all weights we need to load (matching selective_layer_loading_fixed.py logic)
        all_weights = {}

        # Helper: load all tensors from a safetensors file
        def load_safetensors_file(path: Path) -> Dict[str, torch.Tensor]:
            """Load all tensors from a safetensors file."""
            tensors = {}
            if path.exists():
                with safe_open(str(path), framework="pt", device="cpu") as f:
                    for key in f.keys():
                        tensors[key] = f.get_tensor(key)
            return tensors

        # Load embedding (always needed for first peer)
        embed_path = shards_root / "embedding" / "layer.safetensors"
        if embed_path.exists():
            embed_weights = load_safetensors_file(embed_path)
            all_weights.update(embed_weights)
            print(f"✅ Loaded embedding weights from {embed_path}")

        # Load lm_head (always needed for last peer)
        lm_head_path = shards_root / "lm_head" / "layer.safetensors"
        if lm_head_path.exists():
            lm_head_weights = load_safetensors_file(lm_head_path)
            all_weights.update(lm_head_weights)
            print(f"✅ Loaded lm_head weights from {lm_head_path}")
        else:
            # Handle tied embeddings case - copy embed_tokens weights to lm_head
            print("ℹ️ No separate lm_head file found - checking for tied embeddings")
            if "model.embed_tokens.weight" in all_weights:
                # Llama models with tied embeddings
                all_weights["lm_head.weight"] = all_weights["model.embed_tokens.weight"]
                print(
                    "✅ Using tied embeddings - copied embed_tokens weights to lm_head"
                )
            if "embed_tokens.weight" in all_weights:
                # Alternative naming
                all_weights["lm_head.weight"] = all_weights["embed_tokens.weight"]
                print(
                    "✅ Using tied embeddings - copied embed_tokens weights to lm_head"
                )

        # Load model.norm (always needed for last peer)
        norm_path = shards_root / "norm" / "layer.safetensors"
        if norm_path.exists():
            norm_weights = load_safetensors_file(norm_path)
            all_weights.update(norm_weights)
            print(f"✅ Loaded model.norm weights from {norm_path}")

        # Load only specified transformer layers
        for layer_idx in assigned_layers:
            layer_path = shards_root / "layers" / f"layer_{layer_idx}.safetensors"
            if layer_path.exists():
                layer_weights = load_safetensors_file(layer_path)
                all_weights.update(layer_weights)
                print(f"✅ Loaded layer {layer_idx} weights from {layer_path}")
            else:
                print(f"⚠️ Warning: Layer {layer_idx} not found at {layer_path}")

        print("\n🔧 APPLYING LOADED WEIGHTS TO MODEL...")
        print(f"   Total weights loaded: {len(all_weights)}")

        applied_count = 0
        missing_params = []
        print(model)
        for name, param in model.named_parameters():
            if name in all_weights:
                with torch.no_grad():
                    param.copy_(all_weights[name].to(param.dtype).to(param.device))
                    applied_count += 1
            else:
                # Check if this parameter belongs to a layer we should have loaded
                is_expected_missing = False

                # Check if it's from an unassigned layer
                for i in range(400):  # Assuming max 100 layers
                    if f".layers.{i}." in name and i not in assigned_layers:
                        is_expected_missing = True
                        break

                if not is_expected_missing:
                    missing_params.append(name)

        print(
            f"✅ Applied weights to {applied_count}/{len(list(model.named_parameters()))} parameters"
        )

        if (
            missing_params and len(missing_params) < 20
        ):  # Only show first 20 to avoid spam
            print(
                f"⚠️ Parameters without loaded weights (first 20): {missing_params[:20]}"
            )
        elif missing_params:
            print(
                f"⚠️ {len(missing_params)} parameters without loaded weights (expected for unassigned layers)"
            )
            print(f"Missing params: {missing_params}")

        print("✅ Created AsyncLLMEngine (v0) with selective layers")
        return async_engine

    finally:
        model_utils.make_layers = original_make_layers


async def load_model_with_selective_layers(
    model_dir: Path,
    assigned_layers: List[int],
    quantization: Optional[str] = None,
    dtype: Optional[str] = None,
    *,
    use_async_engine: bool = False,
    max_num_seqs: int = 1,
    max_num_batched_tokens: int = 2048,
):
    """
    Load vLLM model with selective layers in a background thread.

    Args:
        model_dir: Path to model directory
        assigned_layers: List of layer indices to load
        quantization: Optional vLLM quantization method (e.g., "bitsandbytes", "awq", "gptq")
        dtype: Optional dtype for activations/weights ("float16", "bfloat16", "float32", "auto")
        use_async_engine: If True, initialize AsyncLLMEngine (v0) instead of LLM
        max_num_seqs: Scheduler cap to avoid intra-step mixing (ring-buffer behavior)
        max_num_batched_tokens: Upper bound on tokens per batch

    Returns:
        Union[LLM, AsyncLLMEngine]: Loaded model/engine instance

    Raises:
        ValueError: If model loading fails
    """
    try:
        # Use config directory for vLLM initialization
        config_dir = model_dir / "config"
        if not config_dir.exists():
            raise ValueError(f"Config directory not found: {config_dir}")

        print("🔧 Loading model with selective layers...")
        print("Loading only a partial model for vLLM Inference")

        loop = asyncio.get_running_loop()

        if use_async_engine:
            print(
                "⚙️ Using AsyncLLMEngine (v0) path with scheduler limits for ring-buffer behavior"
            )
            create_async_engine_fn = partial(
                create_async_vllm_engine_with_selective_layers,
                str(config_dir),
                assigned_layers,
                quantization=quantization,
                dtype=dtype,
                max_num_seqs=max_num_seqs,
                max_num_batched_tokens=max_num_batched_tokens,
            )
            loaded_model = await loop.run_in_executor(None, create_async_engine_fn)
        else:
            loaded_model = await loop.run_in_executor(
                None,
                create_dynamic_vllm_model,
                str(config_dir),
                assigned_layers,
                quantization,
                dtype,
            )

        if loaded_model is None:
            raise ValueError("Model loading returned None")

        print("✅ Model loaded successfully!")

        print(type(loaded_model), dir(loaded_model))
        executor = loaded_model.llm_engine.model_executor
        print(
            f"Executor:{executor.cache_config.num_gpu_blocks}, {executor.cache_config.block_size}, \
            {executor.model_config.max_model_len}"
        )
        return loaded_model

    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        raise ValueError(f"Model loading failed: {e}")


# ============================================================================
# DEPLOYMENT ORCHESTRATION FUNCTIONS
# ============================================================================


async def report_deployment_completion(
    model_name: str,
    peer_id: str,
    success: bool,
    server_host: str,
    server_port: int,
    max_req_in_batch: int,
):
    """
    Notify the central server that this peer has finished deploying.

    Args:
        model_name: Name of the deployed model
        peer_id: ID of the peer reporting completion
        success: Whether deployment was successful
        server_host: Server hostname
        server_port: Server port
    """
    url = f"http://{server_host}:{server_port}/deployment_complete"
    payload = {
        "model_name": model_name,
        "peer_id": peer_id,
        "success": success,
        "max_req_in_batch": max_req_in_batch,
    }
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
        print(f"📤 Reported deployment completion: {payload}")
    except Exception as e:
        print(f"❌ Failed to report deployment completion: {e}")


class DeploymentAttemptTracker:
    """Track deployment attempts to prevent infinite retries"""

    def __init__(self):
        self.attempts: Dict[str, int] = {}

    def get_attempt_key(self, model_name: str, assigned_layers: List[int]) -> str:
        """Create unique key for deployment attempt tracking"""
        return f"{model_name}_{hash(str(assigned_layers))}"

    def should_attempt_deployment(
        self, model_name: str, assigned_layers: List[int], max_attempts: int = 3
    ) -> bool:
        """Check if deployment should be attempted"""
        attempt_key = self.get_attempt_key(model_name, assigned_layers)
        current_attempts = self.attempts.get(attempt_key, 0)
        return current_attempts < max_attempts

    def record_attempt(self, model_name: str, assigned_layers: List[int]) -> int:
        """Record a deployment attempt and return current attempt number"""
        attempt_key = self.get_attempt_key(model_name, assigned_layers)
        current_attempts = self.attempts.get(attempt_key, 0) + 1
        self.attempts[attempt_key] = current_attempts
        return current_attempts

    def clear_attempts(self, model_name: str, assigned_layers: List[int]):
        """Clear attempts for successful deployment"""
        attempt_key = self.get_attempt_key(model_name, assigned_layers)
        self.attempts[attempt_key] = 0


# Global instance for tracking deployment attempts
deployment_tracker = DeploymentAttemptTracker()


async def deploy_model_orchestrator(instructions: Dict[str, Any]) -> tuple[bool, Any]:
    """
    Orchestrate the complete model deployment process.

    This function coordinates downloading, loading, and setup of a distributed model
    based on deployment instructions from the central server.

    Args:
        instructions: Deployment instructions containing model info, layers, files, etc.

    Returns:
        tuple[bool, Any]: (success, loaded_model_or_none)
    """
    model_name = instructions.get("model_name", "unknown")
    assigned_layers = instructions.get("assigned_layers", [])
    quantization = instructions.get(
        "quantization"
    )  # e.g. "bitsandbytes", "awq", "gptq"
    dtype = instructions.get("dtype")  # e.g. "bfloat16", "float16", "auto"
    use_async_engine = bool(instructions.get("use_async_engine", False))
    max_num_seqs = int(instructions.get("max_num_seqs", 1))
    max_num_batched_tokens = int(instructions.get("max_num_batched_tokens", 2048))

    print("🚀 Starting model deployment orchestration...")
    print(f"   Model: {model_name}")
    print(f"   Assigned layers: {assigned_layers}")
    print(f"   Is first peer: {instructions.get('is_first_peer', False)}")
    print(f"   Is last peer: {instructions.get('is_last_peer', False)}")
    print(f"   Required files: {len(instructions.get('required_files', []))}")
    print(f"   Quantization: {quantization} | dtype: {dtype}")
    print(
        f"   Engine: {'AsyncLLMEngine(v0)' if use_async_engine else 'LLM'} | max_num_seqs={max_num_seqs} | max_num_batched_tokens={max_num_batched_tokens}"
    )

    # Check deployment attempts
    if not deployment_tracker.should_attempt_deployment(model_name, assigned_layers):
        print(f"❌ Maximum deployment attempts reached for {model_name}, giving up")
        return False, None

    attempt_num = deployment_tracker.record_attempt(model_name, assigned_layers)
    print(f"🔄 Deployment attempt {attempt_num}/3")

    try:
        # Phase 1: Download model files
        print("📥 Phase 1: Downloading model files...")
        download_success, model_dir = await download_model_files(instructions)
        if not download_success:
            print("❌ File download phase failed")
            return False, None

        # Phase 2: Load model with selective layers
        print("🔧 Phase 2: Loading model with selective layers...")
        try:
            loaded_model = await load_model_with_selective_layers(
                model_dir,
                assigned_layers,
                quantization=quantization,
                dtype=dtype,
                use_async_engine=use_async_engine,
                max_num_seqs=max_num_seqs,
                max_num_batched_tokens=max_num_batched_tokens,
            )
        except ValueError as e:
            print(f"❌ Model loading phase failed: {e}")
            return False, None

        # Phase 3: Success - clear attempts and return
        deployment_tracker.clear_attempts(model_name, assigned_layers)

        print("✅ Model deployment orchestration completed successfully!")
        print(
            f"   Peer role: {'First' if instructions.get('is_first_peer') else 'Last' if instructions.get('is_last_peer') else 'Middle'}"
        )
        print(f"   Loaded layers: {assigned_layers}")
        # memory saving is rough / unchanged here

        return True, loaded_model

    except Exception as e:
        print(f"❌ Model deployment orchestration failed: {e}")
        return False, None
