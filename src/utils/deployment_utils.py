from pathlib import Path
import json
import asyncio
import httpx
import aiofiles
import torch
from typing import Dict, Any, List, Optional
from fastapi import HTTPException
from src.utils.db_utils import get_active_peers, get_peer_metrics
from src.utils.model_utils import distribute_layers_across_peers

def load_model_metadata(shard_folder: str):
    shard_path = Path(shard_folder)
    metadata_file = shard_path / "layer_metadata.json"
    if not metadata_file.exists() or not shard_path.exists():
        raise HTTPException(status_code=404, detail=f"layer_metadata.json or shard folder not found: {shard_folder}")
    with open(metadata_file, "r") as f:
        metadata = json.load(f)
    print(f"📊 Model metadata: {metadata['num_layers']} layers, type: {metadata['model_type']}")
    return metadata

async def get_peers_with_vram():
    active_peers = await get_active_peers()
    if len(active_peers) < 1:
        raise HTTPException(status_code=400, detail="No active peers available for deployment")
    
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
        raise HTTPException(status_code=400, detail="No peers with VRAM information available")
    
    print(f"👥 Found {len(peers_vram)} peers with VRAM data")
    return peers_vram

def create_distribution_plan(metadata, peers_vram, q_bits: int = 32):
    config = {
        "num_hidden_layers": metadata['num_layers'],
        "hidden_size": metadata['hidden_size'],
        "vocab_size": 32000,  # Default for demo
        "num_attention_heads": 32,  # Default for demo  
        "num_key_value_heads": 32,  # Default for demo
        "intermediate_size": metadata['hidden_size'] * 4,  # Standard ratio
    }
    
    distribution_plan = distribute_layers_across_peers(
        config=config,
        peers_vram=peers_vram,
        q_bits=q_bits  # Allow caller to choose precision for VRAM estimate
    )
    print(f"📋 Distribution plan created:")
    print(f"   • Model can fit: {distribution_plan['can_fit_model']}")
    print(f"   • Total VRAM needed: {distribution_plan['model_info']['total_model_vram_gb']:.1f}GB")
    print(f"   • Available VRAM: {distribution_plan['total_available_vram_gb']:.1f}GB")
    print(f"   • Peers involved: {len(distribution_plan['distribution'])}")
    for peer_id, peer_info in distribution_plan['distribution'].items():
        print(f"   • {peer_id}: {peer_info['assigned_layers']} layers, {peer_info['estimated_vram_usage']:.1f}GB")

    if not distribution_plan["can_fit_model"]:
        raise HTTPException(
            status_code=400,
            detail=f"Model cannot fit in available VRAM. Need {distribution_plan['model_info']['total_model_vram_gb']:.1f}GB, have {distribution_plan['total_available_vram_gb']:.1f}GB"
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
    
    for i, (peer_id, peer_info) in enumerate(distribution_plan["distribution"].items()):
        is_first_peer = (i == 0)
        is_last_peer = (i == len(peer_list) - 1)
        
        assigned_layers = list(range(
            sum(p["assigned_layers"] for p in list(distribution_plan["distribution"].values())[:i]),
            sum(p["assigned_layers"] for p in list(distribution_plan["distribution"].values())[:i+1])
        ))

        required_files = []
        required_files.extend([
            "config/config.json",
            "config/tokenizer.json", 
            "config/tokenizer_config.json",
        ])
        
        if is_first_peer:
            required_files.append("embedding/layer.safetensors")
            
        if is_last_peer:
            required_files.extend([
                "lm_head/layer.safetensors",
                "norm/layer.safetensors"
            ])
        
        for layer_idx in assigned_layers:
            required_files.append(f"layers/layer_{layer_idx}.safetensors")
        
        next_peer_ticket = peer_list[i + 1] if (i + 1) < len(peer_list) else None

        deployment_instructions[peer_id] = {
            "model_name": request.model_name,
            "assigned_layers": assigned_layers,
            "is_first_peer": is_first_peer,
            "is_last_peer": is_last_peer,
            "required_files": required_files,
            "server_download_url": f"http://{SERVER_IP}:8000/download_file/{request.model_name.replace('/', '_')}",
            "vram_allocation": peer_info,
            "next_peer_ticket": next_peer_ticket,
            "pipeline": peer_list,
            # NEW: thread quantization/dtype/qbits end-to-end (optional)
            "quantization": getattr(request, "quantization", None),
            "dtype": getattr(request, "dtype", None),
            "qbits": getattr(request, "qbits", None),
        }
    return deployment_instructions


# ============================================================================
# MODEL DOWNLOADING FUNCTIONS
# ============================================================================

async def download_file(url: str, local_path: Path, chunk_size: int = 16*1024*1024) -> bool:
    """
    Download a file from server with progress tracking and resume capability.
    
    Args:
        url: Download URL
        local_path: Local file path to save to
        chunk_size: Download chunk size in bytes (default: 16MB)
        
    Returns:
        bool: True if download successful, False otherwise
    """
    try:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Skip download if file already exists and has content
        if local_path.exists() and local_path.stat().st_size > 0:
            # Optional: Verify file size matches server (HEAD request)
            try:
                async with httpx.AsyncClient() as client:
                    head_response = await client.head(url)
                    if head_response.status_code == 200:
                        remote_size = int(head_response.headers.get("content-length", 0))
                        local_size = local_path.stat().st_size
                        
                        if local_size == remote_size:
                            print(f"✅ File already exists with correct size, skipping: {local_path.name} ({local_size:,} bytes)")
                            return True
                        else:
                            print(f"⚠️ File exists but size mismatch - redownloading: local={local_size:,}, remote={remote_size:,}")
                    else:
                        print(f"⚠️ Could not verify remote file size (HEAD {head_response.status_code}), downloading anyway")
            except Exception as e:
                print(f"⚠️ HEAD request failed: {e}, proceeding with download")
                # If HEAD fails, just check local file exists and has content
                print(f"✅ File exists locally, assuming valid: {local_path.name} ({local_path.stat().st_size:,} bytes)")
                return True
        
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
                        if total_size > 0 and (downloaded % (100 * 1024 * 1024) == 0 or downloaded == total_size):
                            progress = (downloaded / total_size) * 100
                            print(f"   📥 Downloading {local_path.name}: {progress:.1f}% ({downloaded:,}/{total_size:,} bytes)")
        
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
        
        print(f"📥 Starting download of {total_files} files...")
        
        for file_path in instructions["required_files"]:
            file_url = f"{base_url}/{file_path}"
            local_file_path = model_dir / file_path
            
            print(f"📥 Downloading {file_path}...")
            if await download_file(file_url, local_file_path):
                successful_downloads += 1
            else:
                print(f"❌ Failed to download {file_path}")
        
        if successful_downloads != total_files:
            print(f"❌ Only {successful_downloads}/{total_files} files downloaded successfully")
            return False, model_dir
        
        print(f"✅ All {total_files} files downloaded successfully")
        return True, model_dir
        
    except Exception as e:
        print(f"❌ Error downloading model files: {e}")
        return False, Path(".")


# ============================================================================
# MODEL LOADING FUNCTIONS
# ============================================================================

def create_dynamic_vllm_model(model_dir: str, assigned_layers: List[int],
                              quantization: Optional[str] = None,
                              dtype: Optional[str] = None):
    """
    Create vLLM model with only assigned layers loaded by monkey-patching make_layers.

    Args:
        model_dir: Directory containing model files
        assigned_layers: List of layer indices to load
        quantization: Optional vLLM quantization method (e.g., "bitsandbytes", "awq", "gptq")
        dtype: Optional dtype for activations/weights ("float16", "bfloat16", "float32", "auto")
    """
    
    # STEP 1: Monkey-patch vLLM's make_layers function (Prime Intellect's key insight)
    def _selective_make_layers(num_hidden_layers: int, layer_fn, prefix: str):
        """Custom make_layers that creates real layers only for assigned indices."""
        from vllm.model_executor.models.utils import PPMissingLayer, maybe_offload_to_cpu
        
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
        from vllm import LLM, SamplingParams
        
        llm = LLM(
            model=model_dir,
            tensor_parallel_size=1,
            enforce_eager=True,  # Required for custom layer loading
            max_model_len=128,   # Small for demo
            disable_log_stats=True,
            skip_tokenizer_init=False,
            gpu_memory_utilization=0.8,  # Use much less memory
            use_v2_block_manager=False,  # Force legacy engine to avoid v1 memory pre-allocation
            load_format="dummy",     # ← this is the magic flag
            dtype=(dtype or "float16"),
            quantization=quantization  # vLLM will select kernels/flows accordingly
        )
        
        print(f"✅ Successfully created vLLM model with selective layers!")
        print(f"   Our monkey-patch created real layers for: {assigned_layers}")
        print(f"   Quantization: {quantization} | dtype: {dtype or 'float16'}")
        print(f"   All other layers are PPMissingLayer (passthrough)")
        
        return llm
        
    finally:
        # Restore original function
        model_utils.make_layers = original_make_layers


async def load_model_with_selective_layers(model_dir: Path,
                                           assigned_layers: List[int],
                                           quantization: Optional[str] = None,
                                           dtype: Optional[str] = None):
    """
    Load vLLM model with selective layers in a background thread.
    
    Args:
        model_dir: Path to model directory
        assigned_layers: List of layer indices to load
        quantization: Optional vLLM quantization method (e.g., "bitsandbytes", "awq", "gptq")
        dtype: Optional dtype for activations/weights ("float16", "bfloat16", "float32", "auto")
        
    Returns:
        LLM: Loaded vLLM model instance
        
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
        return loaded_model
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        raise ValueError(f"Model loading failed: {e}")


# ============================================================================
# DEPLOYMENT ORCHESTRATION FUNCTIONS
# ============================================================================

async def report_deployment_completion(model_name: str, peer_id: str, success: bool, server_host: str, server_port: int):
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
        "success": success
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
    
    def should_attempt_deployment(self, model_name: str, assigned_layers: List[int], max_attempts: int = 3) -> bool:
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
    model_name = instructions.get('model_name', 'unknown')
    assigned_layers = instructions.get('assigned_layers', [])
    quantization = instructions.get('quantization')  # e.g. "bitsandbytes", "awq", "gptq"
    dtype = instructions.get('dtype')  # e.g. "bfloat16", "float16", "auto"
    
    print(f"🚀 Starting model deployment orchestration...")
    print(f"   Model: {model_name}")
    print(f"   Assigned layers: {assigned_layers}")
    print(f"   Is first peer: {instructions.get('is_first_peer', False)}")
    print(f"   Is last peer: {instructions.get('is_last_peer', False)}")
    print(f"   Required files: {len(instructions.get('required_files', []))}")
    print(f"   Quantization: {quantization} | dtype: {dtype}")

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
            )
        except ValueError as e:
            print(f"❌ Model loading phase failed: {e}")
            return False, None
        
        # Phase 3: Success - clear attempts and return
        deployment_tracker.clear_attempts(model_name, assigned_layers)
        
        print(f"✅ Model deployment orchestration completed successfully!")
        print(f"   Peer role: {'First' if instructions.get('is_first_peer') else 'Last' if instructions.get('is_last_peer') else 'Middle'}")
        print(f"   Loaded layers: {assigned_layers}")
        # memory saving is rough / unchanged here
        
        return True, loaded_model
        
    except Exception as e:
        print(f"❌ Model deployment orchestration failed: {e}")
        return False, None