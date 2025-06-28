import asyncio
import socket
import tempfile
import os
import json
import torch
import iroh
import httpx
from iroh.iroh_ffi import uniffi_set_event_loop
from pathlib import Path
from typing import List, Dict, Any, Optional
import time
import aiofiles

# FORCE vLLM v0 mode (required for selective layer loading)
os.environ["VLLM_USE_V1"] = "0"
print("🔧 FORCED vLLM v0 mode (VLLM_USE_V1=0) for selective layer loading compatibility")

from src.config.settings import (
    SERVER_HOST,
    SERVER_PORT,
    GPU_METRICS_INTERVAL
)
from src.utils.db_utils import (
    register_peer,
    deregister_peer,
    get_active_peers,
    update_peer_metrics
)
from src.utils.gpu_utils import (
    get_system_metrics,
    format_metrics_for_db
)

# Global variables for deployed model
deployed_model = None
deployment_status = "idle"  # idle, downloading, loading, ready, failed

# Global Iroh objects for completion reporting
current_doc = None
current_node = None 
current_peer_id = None

# Constants for document keys (preserved for future pipeline parallel inference)
TRIGGER_KEY = "job_trigger"  # Key for triggering pipeline operations
FINAL_RESULT_KEY = "final_result"  # Key for final pipeline results

# Matrix computation removed - keeping blob communication for future pipeline parallel inference

# ============================================================================
# SELECTIVE LAYER LOADING IMPLEMENTATION  
# ============================================================================

def create_dynamic_vllm_model(model_dir: str, assigned_layers: List[int]):
    """Create vLLM model with only assigned layers loaded by monkey-patching make_layers."""
    
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
            gpu_memory_utilization=0.3,  # Use much less memory
            use_v2_block_manager=False,  # Force legacy engine to avoid v1 memory pre-allocation
        )
        
        print(f"✅ Successfully created vLLM model with selective layers!")
        print(f"   Our monkey-patch created real layers for: {assigned_layers}")
        print(f"   All other layers are PPMissingLayer (passthrough)")
        
        return llm
        
    finally:
        # Restore original function
        model_utils.make_layers = original_make_layers

# ============================================================================
# FILE DOWNLOAD AND DEPLOYMENT LOGIC
# ============================================================================

async def download_file(url: str, local_path: Path, chunk_size: int = 16*1024*1024) -> bool:  # 16MB chunks for maximum speed
    """Download a file from server with progress tracking."""
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

async def deploy_model_from_instructions(instructions: Dict[str, Any]) -> bool:
    """Deploy model based on deployment instructions received from server."""
    global deployed_model, deployment_status
    
    # Check if model is already successfully deployed
    if deployment_status == "ready" and deployed_model is not None:
        print(f"✅ Model already successfully deployed, skipping new deployment")
        return True
    
    # Prevent multiple concurrent deployments
    if deployment_status in ["downloading", "loading"]:
        print(f"⚠️ Deployment already in progress ({deployment_status}), skipping...")
        return False
    
    try:
        deployment_status = "downloading"
        print(f"🚀 Starting model deployment...")
        print(f"   Model: {instructions['model_name']}")
        print(f"   Assigned layers: {instructions['assigned_layers']}")
        print(f"   Is first peer: {instructions['is_first_peer']}")
        print(f"   Is last peer: {instructions['is_last_peer']}")
        print(f"   Required files: {len(instructions['required_files'])}")
        
        # Create local model directory
        model_dir = Path(f"./deployed_models/{instructions['model_name']}")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Download required files
        base_url = instructions["server_download_url"]
        successful_downloads = 0
        total_files = len(instructions["required_files"])
        
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
            deployment_status = "failed"
            return False
        
        print(f"✅ All {total_files} files downloaded successfully")
        
        # Load the model with selective layers
        deployment_status = "loading"
        print(f"🔧 Loading model with selective layers...")
        
        # Use config directory for vLLM initialization
        config_dir = model_dir / "config"
        if not config_dir.exists():
            print(f"❌ Config directory not found: {config_dir}")
            deployment_status = "failed"
            return False
        
        deployed_model = create_dynamic_vllm_model(
            model_dir=str(config_dir),
            assigned_layers=instructions["assigned_layers"]
        )
        
        deployment_status = "ready"
        print(f"✅ Model deployment completed successfully!")
        print(f"   Peer role: {'First' if instructions['is_first_peer'] else 'Last' if instructions['is_last_peer'] else 'Middle'}")
        print(f"   Loaded layers: {instructions['assigned_layers']}")
        print(f"   Memory optimization: ~{100 * (22 - len(instructions['assigned_layers'])) / 22:.1f}% VRAM savings")
        
        # Report completion to server
        await report_deployment_completion(instructions['model_name'], success=True)
        
        return True
        
    except Exception as e:
        print(f"❌ Model deployment failed: {e}")
        deployment_status = "failed"
        
        # Report failure to server  
        try:
            await report_deployment_completion(instructions['model_name'], success=False)
        except:
            pass  # Don't fail on reporting failure
        
        return False

async def report_deployment_completion(model_name: str, success: bool):
    """Report deployment completion status to server via Iroh."""
    try:
        # Access global Iroh objects (these will be set in main())
        global current_doc, current_node, current_peer_id
        
        if not all([current_doc, current_node, current_peer_id]):
            print("⚠️ Cannot report completion - Iroh not initialized")
            return
            
        author = await current_node.authors().create()
        completion_key = f"deployment_complete_{current_peer_id}_{int(time.time())}"
        completion_data = json.dumps({
            "peer_id": current_peer_id,
            "model_name": model_name,
            "success": success,
            "timestamp": int(time.time())
        }).encode()
        
        await current_doc.set_bytes(author, completion_key.encode(), completion_data)
        print(f"📤 Reported deployment {'success' if success else 'failure'} to server")
        
    except Exception as e:
        print(f"❌ Failed to report deployment completion: {e}")

async def handle_deployment_instruction(doc, node, instruction_data: bytes):
    """Handle deployment instruction received via Iroh."""
    try:
        instruction = json.loads(instruction_data.decode())
        
        if instruction.get("action") == "deploy_model":
            instructions = instruction.get("instructions", {})
            print(f"📨 Received deployment instruction for {instructions.get('model_name', 'unknown')}")
            
            # Deploy model in background (non-blocking)
            asyncio.create_task(deploy_model_from_instructions(instructions))
            
        else:
            print(f"⚠️  Unknown instruction action: {instruction.get('action')}")
            
    except Exception as e:
        print(f"❌ Error handling deployment instruction: {e}")

async def monitor_deployment_instructions(doc, node, peer_id: str):
    """Monitor for deployment instructions sent to this peer."""
    seen_hashes = set()
    
    while True:
        try:
            # Look for deployment instructions addressed to this peer
            entries = await doc.get_many(iroh.Query.all(None))
            
            for entry in entries:
                key = entry.key().decode()
                hash_value = entry.content_hash()
                
                # Skip if we've already processed this entry
                if hash_value in seen_hashes:
                    continue
                
                # Check if this is a deployment instruction for us
                if key.startswith(f"deploy_instruction_{peer_id}_"):
                    seen_hashes.add(hash_value)
                    content = await node.blobs().read_to_bytes(hash_value)
                    
                    # Add timestamp check to ignore old instructions
                    try:
                        # Extract timestamp from key: deploy_instruction_{peer_id}_{timestamp}
                        timestamp_str = key.split("_")[-1]
                        instruction_time = int(timestamp_str)
                        current_time = int(time.time())
                        
                        # Ignore instructions older than 30 seconds
                        if current_time - instruction_time > 30:
                            print(f"⏰ Ignoring old deployment instruction from {instruction_time}")
                            continue
                    except (ValueError, IndexError):
                        # If timestamp parsing fails, process the instruction anyway
                        pass
                    
                    await handle_deployment_instruction(doc, node, content)
            
        except Exception as e:
            print(f"❌ Error monitoring deployment instructions: {e}")
        
        await asyncio.sleep(2)  # Check every 2 seconds

def get_model_status() -> Dict[str, Any]:
    """Get current model deployment status."""
    global deployed_model, deployment_status
    
    return {
        "status": deployment_status,
        "model_loaded": deployed_model is not None,
        "ready_for_inference": deployment_status == "ready" and deployed_model is not None
    }

# ============================================================================
# EXISTING FUNCTIONALITY (Matrix computation, heartbeat, etc.)
# ============================================================================

async def get_shared_ticket():
    """
    Fetch the shared ticket from the server.
    
    Returns:
        str: The shared ticket for joining the Iroh document
        
    Raises:
        Exception: If unable to fetch the ticket from the server
    """
    try:
        server_url = f"http://{SERVER_HOST}:{SERVER_PORT}"
        async with httpx.AsyncClient() as client:
            response = await client.get(f'{server_url}/ticket')
            response.raise_for_status()
            ticket_data = response.json()
            ticket = ticket_data["ticket"]
            print(f"✅ Fetched shared ticket from server at {server_url}")
            return ticket
    except httpx.RequestError as e:
        raise Exception(f"Failed to connect to server: {e}")
    except httpx.HTTPStatusError as e:
        raise Exception(f"Server returned error {e.response.status_code}: {e.response.text}")
    except json.JSONDecodeError as e:
        raise Exception(f"Invalid ticket format received: {e}")
    except KeyError as e:
        raise Exception(f"Missing 'ticket' field in server response")
    except Exception as e:
        raise Exception(f"Unexpected error fetching ticket: {e}")

async def send_blob(doc, author, peer_id: str, data: torch.Tensor):
    """
    Send a tensor to another peer in the network.
    
    Args:
        doc: Iroh document
        author: Iroh author for writing
        peer_id: ID of the recipient peer
        data: Tensor data to send
    """
    try:
        encoded = json.dumps(data.tolist()).encode()
        await doc.set_bytes(author, peer_id.encode(), encoded)
        print(f"📤 Sent to {peer_id}: {data}")
    except Exception as e:
        print(f"❌ Failed to send to {peer_id}: {e}")

async def receive_blob(doc, peer_id: str, node):
    """
    Wait for and receive a tensor addressed to this peer.
    
    Args:
        doc: Iroh document
        peer_id: This peer's ID
        node: Iroh node
        
    Returns:
        The received tensor
    """
    seen = set()  # Track already processed content hashes
    while True:
        try:
            entries = await doc.get_many(iroh.Query.all(None))
            for entry in entries:
                key = entry.key().decode()
                if key != peer_id:
                    continue
                hash = entry.content_hash()
                if hash in seen:
                    continue
                seen.add(hash)
                content = await node.blobs().read_to_bytes(hash)
                tensor = torch.tensor(json.loads(content.decode()))
                return tensor
        except Exception as e:
            print(f"❌ Polling error for {peer_id}: {e}")
        await asyncio.sleep(2)  # Poll every 2 seconds

# ============================================================================
# PIPELINE PARALLEL COMMUNICATION (for future hidden state passing)
# ============================================================================

async def send_hidden_states(doc, author, target_peer: str, hidden_states: torch.Tensor):
    """Send hidden states to next peer in pipeline (for future pipeline parallel inference)."""
    return await send_blob(doc, author, target_peer, hidden_states)

async def receive_hidden_states(doc, peer_id: str, node):
    """Receive hidden states from previous peer in pipeline (for future pipeline parallel inference)."""
    return await receive_blob(doc, peer_id, node)

# Matrix computation removed - blob communication functions preserved for future pipeline parallel inference

async def upload_metrics(doc, author, peer_id: str):
    """Upload system metrics to the document and database."""
    try:
        metrics = get_system_metrics()
        formatted_metrics = format_metrics_for_db(metrics)
        
        # Create a heartbeat key with timestamp for freshness
        heartbeat_key = f"heartbeat_{peer_id}_{int(asyncio.get_event_loop().time() * 1000)}"
        gpu_info = formatted_metrics.get("gpu_info", [])
        total_free_vram = formatted_metrics.get("total_free_vram_gb", 0.0)
        
        # Compact metrics for Iroh document
        compact_metrics = {
            "cpu": metrics.cpu_percent,
            "ram": metrics.ram_percent,
            "free_vram": total_free_vram,
            "gpu_count": len(gpu_info),
            "timestamp": formatted_metrics["timestamp"].isoformat() if hasattr(formatted_metrics["timestamp"], "isoformat") else str(formatted_metrics["timestamp"])
        }
        value = json.dumps(compact_metrics).encode()
        
        # Store in Iroh document with timestamped key
        await doc.set_bytes(author, heartbeat_key.encode(), value)
        
        # Store full metrics in database
        await update_peer_metrics(peer_id, formatted_metrics)
        
        # Brief status for every heartbeat  
        print(f"💓 Heartbeat {peer_id}: CPU {metrics.cpu_percent:.1f}%, VRAM {total_free_vram:.1f}GB")
    except Exception as e:
        print(f"❌ Failed to upload metrics: {e}")

async def continuous_heartbeat(doc, author, peer_id: str, node, interval_ms: int = 1000):
    """
    Continuously send heartbeat with metrics to the server and monitor for acknowledgments.
    Stops sending if server doesn't acknowledge within timeout.
    
    Args:
        doc: Iroh document
        author: Iroh author for writing  
        peer_id: This peer's ID
        node: Iroh node for reading acknowledgments
        interval_ms: Heartbeat interval in milliseconds (default: 1000ms = 1 second)
    """
    print(f"💓 Starting bidirectional heartbeat every {interval_ms}ms")
    
    last_ack_time = time.time()
    server_timeout = 30  # Stop heartbeats if no server ack for 30 seconds
    grace_period = 10   # Give server 10 seconds to start acknowledging
    seen_acks = set()
    
    while True:
        try:
            # Send heartbeat
            await upload_metrics(doc, author, peer_id)
            current_time = time.time()
            
            # Check for server acknowledgments
            try:
                entries = await doc.get_many(iroh.Query.all(None))
                for entry in entries:
                    key = entry.key().decode()
                    hash_value = entry.content_hash()
                    
                    # Look for acknowledgments from server
                    if key.startswith(f"heartbeat_ack_{peer_id}_") and hash_value not in seen_acks:
                        seen_acks.add(hash_value)
                        last_ack_time = current_time
                        print(f"💓 Server acknowledged heartbeat")
                        break
            except Exception as e:
                print(f"⚠️ Error checking server acknowledgments: {e}")
            
            # Check if server is responsive (after grace period)
            if current_time - last_ack_time > server_timeout and current_time - last_ack_time > grace_period:
                print(f"💔 Server unresponsive for {server_timeout}s, stopping heartbeats")
                break
            
            await asyncio.sleep(interval_ms / 1000.0)  # Convert ms to seconds
            
        except asyncio.CancelledError:
            print(f"💓 Heartbeat cancelled for {peer_id}")
            break
        except Exception as e:
            print(f"❌ Heartbeat error for {peer_id}: {e}")
            # Continue heartbeat even if one upload fails
            await asyncio.sleep(1)  # Wait 1 second before retry

async def main():
    """Main function to run the distributed computation node"""
    global current_doc, current_node, current_peer_id
    
    # Set up the asyncio event loop for Iroh
    uniffi_set_event_loop(asyncio.get_running_loop())

    # Set up a unique data directory for this node
    hostname = socket.gethostname()
    data_dir = os.path.join(tempfile.gettempdir(), f"iroh_node_{hostname}")
    
    # Configure and initialize the Iroh node
    options = iroh.NodeOptions()
    options.enable_docs = True
    node = await iroh.Iroh.memory_with_options(options)
    peer_id = await node.net().node_id()
    print(f"🤖 Running as peer: {peer_id}")
    
    # Set globals for completion reporting
    current_node = node
    current_peer_id = peer_id

    # Register this peer in MongoDB
    await register_peer(peer_id, hostname)
    print(f"✅ Registered in MongoDB as {peer_id}")

    # Fetch the shared ticket from the server
    try:
        shared_ticket = await get_shared_ticket()
    except Exception as e:
        print(f"❌ Failed to get shared ticket: {e}")
        return

    # Join the shared document and create an author for writing
    doc = await node.docs().join(iroh.DocTicket(shared_ticket))
    author = await node.authors().create()
    
    # Set global doc for completion reporting
    current_doc = doc
    
    # Upload initial system metrics
    await upload_metrics(doc, author, peer_id)

    # Start continuous heartbeat as background task
    heartbeat_interval_ms = int(os.getenv("HEARTBEAT_INTERVAL_MS", "1000"))  # Default 1 second
    heartbeat_task = asyncio.create_task(
        continuous_heartbeat(doc, author, peer_id, node, heartbeat_interval_ms)
    )
    
    # Start deployment instruction monitoring as background task
    deployment_task = asyncio.create_task(
        monitor_deployment_instructions(doc, node, peer_id)
    )

    try:
        # Peer is now ready for deployment instructions
        # Matrix computation system removed - peers now wait for model deployment
        print("✅ Peer ready for model deployment instructions")
        
        # Keep peer alive and responsive to deployment instructions
        while True:
            await asyncio.sleep(5)  # Heartbeat and deployment monitoring run in background

    except Exception as e:
        print(f"❌ Error in main loop: {e}")
    finally:
        # Cancel background tasks
        heartbeat_task.cancel()
        deployment_task.cancel()
        
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass
            
        try:
            await deployment_task
        except asyncio.CancelledError:
            pass
        
        # Deregister peer when shutting down
        await deregister_peer(peer_id)
        print(f"👋 Deregistered {peer_id} from pipeline")

if __name__ == "__main__":
    asyncio.run(main()) 