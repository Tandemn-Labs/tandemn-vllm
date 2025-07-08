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
from colorama import Fore, Style, init as colorama_init
from iroh import PublicKey, NodeAddr, MessageType
from src.utils.inference_utils import register_inference_hooks, INFERENCE_CONTEXT
import pickle
from vllm import SamplingParams

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
    format_metrics_for_db,
    get_total_free_vram
)

# Global variables for deployed model
deployed_model = None
deployment_status = "idle"  # idle, downloading, loading, ready, failed
start_inference_run = None # this is the global variable for the inference run
assigned_layers_global=[] # this is the global variable for the assigned layers

# Global Iroh objects for completion reporting
current_node = None 
current_peer_id = None
hidden_state_gossip_sink = None # this is the gossip sink for the hidden states

# Constants for document keys
# TRIGGER_KEY = "job_trigger"  # Key used to trigger a new computation job
# FINAL_RESULT_KEY = "final_result"  # Key used to store the final computation result

# # Predefined matrices for each position in the pipeline
# MATRIX_MAP = {
#     0: torch.tensor([[2., 0.], [1., 2.]]),  # Matrix A (first machine)
#     1: torch.tensor([[0., 1.], [1., 0.]]),  # Matrix B (second machine)
#     2: torch.tensor([[1., 1.], [0., 1.]]),  # Matrix C (third machine)
# }

# IROH STARTS HERE
class DeploymentGossipCallback(iroh.GossipMessageCallback):
    "Handle deployment instructions received via gossip"
    def __init__(self, node, peer_id):
        super().__init__()
        self.node = node
        self.peer_id = peer_id

    async def on_message(self, msg):
        print("="*100)
        print(f"�� [DEBUG] DeploymentGossipCallback received message")
        t = msg.type()  # ✅ Call the method
        print(f"🔍 [DEBUG] Message type: {t}")
        print("="*100)
        
        if t == MessageType.JOINED:
            print("🔎 Deployment mesh membership:", msg.as_joined())
            return
            
        if t == MessageType.RECEIVED:
            print(f"�� [DEBUG] Processing RECEIVED deployment message")
            rc = msg.as_received()
            print(f"🔍 [DEBUG] Message content length: {len(rc.content)} bytes")
            
            try:
                payload = json.loads(rc.content.decode())  # ✅ Use content, not payload
                print(f"🔍 [DEBUG] Parsed JSON payload: {list(payload.keys())}")
                
                # Check if this is a deployment instruction
                if payload.get("action") != "deploy_model":
                    print(f"🔍 [DEBUG] Not a deployment instruction, skipping")
                    return
                
                # check if the target peer is this peer
                target_peer_id = payload.get("target_peer_id")
                if target_peer_id != self.peer_id:
                    print(f"🔍 [DEBUG] Not a deployment instruction for this peer, skipping, {target_peer_id}")
                    return
                    
                instructions = payload.get("instructions", {})
                print(f"📨 Received deployment instructions for {self.peer_id}: {instructions}")
                
                # Deploy model in background
                asyncio.create_task(deploy_model_from_instructions(instructions))
                
            except Exception as e:
                print(f"❌ Error handling deployment instruction: {e}")
                print(f"❌ Exception type: {type(e)}")
                import traceback
                traceback.print_exc()
# IROH ENDS HERE

# IROH STARTS HERE
class HiddenStateCallback(iroh.GossipMessageCallback):
    "handles the hidden state gossip messages"
    async def on_message(self, msg):
        if msg.type()!=MessageType.RECEIVED:
            print(f"🔍 [DEBUG] HiddenStateCallback received non-RECEIVED message")
            return
        try:
            ref = json.loads(msg.as_received().content.decode())
            request_id=ref.get("request_id")

            # check if this message is for this peer or some else
            if not request_id or request_id not in INFERENCE_CONTEXT:
                print(f"🔍 [DEBUG] HiddenStateCallback received message for unknown request_id: {request_id}")
                return
            print(f"🔍 [DEBUG] HiddenStateCallback received message for request_id: {request_id}")
            
            # Properly download the blob first before reading
            blob_ticket_str = ref.get("blob_ticket")
            if not blob_ticket_str:
                print(f"❌ No blob_ticket in message for {request_id}")
                return
                
            print(f"🔍 [DEBUG] Starting blob download for request {request_id}")
            blob_ticket = iroh.BlobTicket(blob_ticket_str)
            blob_hash = blob_ticket.hash()
            opts = blob_ticket.as_download_options()
            
            # Define a simple download callback
            class DownloadProgressCallback(iroh.DownloadCallback):
                async def progress(self, progress):
                    print(f"🔍 [DEBUG] Download progress: {progress}")
                    pass  # Silent progress tracking
            
            # Download the blob using ticket-derived options
            print(f"🔍 [DEBUG] Downloading blob {blob_hash}...")
            await current_node.blobs().download(blob_hash, opts, DownloadProgressCallback())
            print(f"🔍 [DEBUG] Download completed, reading bytes...")
            
            # Now read the downloaded blob
            hidden_bytes = await current_node.blobs().read_to_bytes(blob_hash)
            print(f"🔍 [DEBUG] Read {len(hidden_bytes)} bytes from blob")
            data = pickle.loads(hidden_bytes)

            # check if hidden state or residual
            # if information is provided, set the future to the result so the pre_hooks can be unblocked
            if ref.get("is_residual"):
                future=INFERENCE_CONTEXT[request_id].get("residual_future")
                if future:
                    future.set_result(data)
                    print(f"✅ Set residual future for {request_id}")
            else:
                future=INFERENCE_CONTEXT[request_id].get("hidden_state_future")
                if future:
                    future.set_result(data)
                    print(f"✅ Set hidden_state future for {request_id}")
        except Exception as e:
            print(f"❌ Error in HiddenStateCallback: {e}")
            import traceback
            traceback.print_exc()
# IROH ENDS HERE

# IROH STARTS HERE
# this is the callback function that needs to be called for the inference to start

class TriggerCallback(iroh.GossipMessageCallback):
    "handles the initial inference trigger from the server"
    async def on_message(self, msg):
        if msg.type()!=MessageType.RECEIVED:
            print(f"🔍 [DEBUG] TriggerCallback received non-RECEIVED message")
            return
        
        try:
            payload=json.loads(msg.as_received().content.decode())
            if payload.get("action")!="start_inference":
                print(f"🔍 [DEBUG] TriggerCallback received non-start_inference message")
                return
            
            pipeline = payload.get("pipeline")
            request_id = payload.get("request_id")
            
            # ALL peers need to know about this request for hidden state handling
            if request_id and current_peer_id in pipeline:
                print(f"🔍 [DEBUG] TriggerCallback initializing INFERENCE_CONTEXT for {request_id}")
                INFERENCE_CONTEXT[request_id] = {}
            
            # we need to make sure that ONLY THE FIRST PEER starts the inference process
            if pipeline and pipeline[0]==current_peer_id:
                print(f"🔍 [DEBUG] TriggerCallback received start_inference message from the first peer")
                if not start_inference_run:
                    print(f"🔍 [DEBUG] TriggerCallback received start_inference message from the first peer, but start_inference_run is not set")
                    return
                
                ##get the inference context
                INFERENCE_CONTEXT[payload.get("request_id")]={}
                sampling_params= SamplingParams(**payload.get("sampling_params"))

                # start the synchronous and blocking inference run in a separate thread
                # so that the main thread can continue gossip messages and the main loop isnt blocked
                loop=asyncio.get_running_loop()
                # loop.run_in_executor(None, # use default thread pool
                #                     start_inference_run,
                #                     payload.get("request_id"),
                #                     pipeline,
                #                     payload.get("token_ids"),
                #                     sampling_params,
                #                     payload.get("assigned_layers"))
                loop.run_in_executor(
                    None,
                    start_inference_run,
                    payload["request_id"],
                    payload["pipeline"],
                    payload["input_text"],
                    SamplingParams(**payload["sampling_params"]),
                    payload["assigned_layers"],            # will now be your dict
                )
        except Exception as e:
            print(f"❌ Error in TriggerCallback: {e}")
            import traceback
            traceback.print_exc()

        print(f"🔍 [DEBUG] TriggerCallback completed for {payload.get('request_id')}")
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
            gpu_memory_utilization=0.8,  # Use much less memory
            use_v2_block_manager=False,  # Force legacy engine to avoid v1 memory pre-allocation
            load_format="dummy",     # ← this is the magic flag
            dtype="bfloat16"
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
    global deployed_model, deployment_status, start_inference_run, hidden_state_gossip_sink, current_node, current_peer_id
    
    model_name = instructions.get('model_name', 'unknown')
    
    # Check if model is already successfully deployed
    if deployment_status == "ready" and deployed_model is not None:
        print(f"✅ Model {model_name} already successfully deployed, skipping new deployment")
        return True
    
    # Prevent multiple concurrent deployments
    if deployment_status in ["downloading", "loading"]:
        print(f"⚠️ Deployment already in progress ({deployment_status}), skipping...")
        return False
    
    # Track deployment attempts to prevent infinite retries
    global deployment_attempts
    if not hasattr(deploy_model_from_instructions, 'deployment_attempts'):
        deploy_model_from_instructions.deployment_attempts = {}
    
    attempt_key = f"{model_name}_{hash(str(instructions['assigned_layers']))}"
    current_attempts = deploy_model_from_instructions.deployment_attempts.get(attempt_key, 0)
    
    if current_attempts >= 3:
        print(f"❌ Maximum deployment attempts (3) reached for {model_name}, giving up")
        deployment_status = "failed"
        return False
    
    deploy_model_from_instructions.deployment_attempts[attempt_key] = current_attempts + 1
    print(f"🚀 Starting model deployment (attempt {current_attempts + 1}/3)...")

    try:
        deployment_status = "downloading"
        print(f"   Model: {model_name}")
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
        print("Loading only a partial model for vLLM Inference")

        # running the blocking operation within the machine runner,
        # in a different thread, so the heartbeat loop can continue
        # and the peer does not STALL during a deployment.
        loop_for_deployment = asyncio.get_running_loop()
        
        # AWAIT the result from the background thread. This pauses this function
        # but allows the event loop to run other tasks (like heartbeats).
        # Any exception from the background thread (like the out-of-memory error)
        # will be raised here and caught by the outer try/except block.
        loaded_model = await loop_for_deployment.run_in_executor(
            None,
            create_dynamic_vllm_model,
            str(config_dir),
            instructions["assigned_layers"],
        )

        # Explicitly update the global state ONLY after successful loading.
        deployed_model = loaded_model

        # Now we can safely check if the model was loaded before proceeding.
        if deployed_model is None:
            print("❌ Model loading returned None. Deployment cannot continue.")
            # Raise an exception to be caught by the main handler.
            raise ValueError("Model loading failed in the background thread.")

        # If loading is successful, prepare the inference function.
        print("✅ Model loaded successfully, registering inference hooks...")
        server_url = f"http://{SERVER_HOST}:{SERVER_PORT}"
        start_inference_run = register_inference_hooks(
            llm=deployed_model,
            node=current_node,
            hidden_state_gossip_sink=hidden_state_gossip_sink,
            peer_id=current_peer_id,
            server_url=server_url
        )
        assigned_layers_global = instructions['assigned_layers']

        ##################################################
        deployment_status = "ready"
        print(f"✅ Model deployment completed successfully!")
        print(f"   Peer role: {'First' if instructions['is_first_peer'] else 'Last' if instructions['is_last_peer'] else 'Middle'}")
        print(f"   Loaded layers: {instructions['assigned_layers']}")
        print(f"   Memory optimization: ~{100 * (28 - len(instructions['assigned_layers'])) / 28:.1f}% VRAM savings")
        
        # Clear attempts counter on success
        deploy_model_from_instructions.deployment_attempts[attempt_key] = 0
        deployment_status = "ready"
        print(f"🔍 [DEBUG] Deployment status set to ready")
        
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

# async def report_deployment_completion(model_name: str, success: bool):
#     """Report deployment completion status to server via Iroh."""
#     try:
#         # Access global Iroh objects (these will be set in main())
#         global current_doc, current_node, current_peer_id
        
#         if not all([current_doc, current_node, current_peer_id]):
#             print("⚠️ Cannot report completion - Iroh not initialized")
#             return
            
#         author = await current_node.authors().create()
#         completion_key = f"deployment_complete_{current_peer_id}_{int(time.time())}"
#         completion_data = json.dumps({
#             "peer_id": current_peer_id,
#             "model_name": model_name,
#             "success": success,
#             "timestamp": int(time.time())
#         }).encode()
        
#         await current_doc.set_bytes(author, completion_key.encode(), completion_data)
#         print(f"📤 Reported deployment {'success' if success else 'failure'} to server")
        
#     except Exception as e:
#         print(f"❌ Failed to report deployment completion: {e}")


async def report_deployment_completion(model_name: str, success: bool):
    """
    Notify the central server that this peer has finished deploying.
    """
    url = f"http://{SERVER_HOST}:{SERVER_PORT}/deployment_complete"
    # url = f"http://localhost:8000/deployment_complete"
    payload = {
        "model_name": model_name,
        "peer_id": current_peer_id,
        "success": success
    }
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
        print(f"📤 Reported deployment completion: {payload}")
    except Exception as e:
        print(f"❌ Failed to report deployment completion: {e}")

# async def handle_deployment_instruction(doc, node, instruction_data: bytes):
#     """Handle deployment instruction received via Iroh."""
#     try:
#         instruction = json.loads(instruction_data.decode())
        
#         if instruction.get("action") == "deploy_model":
#             instructions = instruction.get("instructions", {})
#             print(f"📨 Received deployment instruction for {instructions.get('model_name', 'unknown')}")
            
#             # Deploy model in background (non-blocking)
#             asyncio.create_task(deploy_model_from_instructions(instructions))
            
#         else:
#             print(f"⚠️  Unknown instruction action: {instruction.get('action')}")
            
#     except Exception as e:
#         print(f"❌ Error handling deployment instruction: {e}")

# async def monitor_deployment_instructions(doc, node, peer_id: str):
#     """Monitor for deployment instructions sent to this peer."""
#     seen_hashes = set()
    
#     while True:
#         try:
#             # Look for deployment instructions addressed to this peer
#             entries = await doc.get_many(iroh.Query.all(None))
            
#             for entry in entries:
#                 key = entry.key().decode()
#                 hash_value = entry.content_hash()
                
#                 # Skip if we've already processed this entry
#                 if hash_value in seen_hashes:
#                     continue
                
#                 # Check if this is a deployment instruction for us
#                 if key.startswith(f"deploy_instruction_{peer_id}_"):
#                     seen_hashes.add(hash_value)
#                     content = await node.blobs().read_to_bytes(hash_value)
                    
#                     # Add timestamp check to ignore old instructions
#                     try:
#                         # Extract timestamp from key: deploy_instruction_{peer_id}_{timestamp}
#                         timestamp_str = key.split("_")[-1]
#                         instruction_time = int(timestamp_str)
#                         current_time = int(time.time())
                        
#                         # Ignore instructions older than 30 seconds
#                         if current_time - instruction_time > 30:
#                             print(f"⏰ Ignoring old deployment instruction from {instruction_time}")
#                             continue
#                     except (ValueError, IndexError):
#                         # If timestamp parsing fails, process the instruction anyway
#                         pass
                    
#                     await handle_deployment_instruction(doc, node, content)
            
#         except Exception as e:
#             print(f"❌ Error monitoring deployment instructions: {e}")
        
#         await asyncio.sleep(2)  # Check every 2 seconds

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

# async def get_shared_ticket():
#     """
#     Fetch the shared ticket from the server.
    
#     Returns:
#         str: The shared ticket for joining the Iroh document
        
#     Raises:
#         Exception: If unable to fetch the ticket from the server
#     """
#     try:
#         server_url = f"http://{SERVER_HOST}:{SERVER_PORT}"
#         async with httpx.AsyncClient() as client:
#             response = await client.get(f'{server_url}/ticket')
#             response.raise_for_status()
#             ticket_data = response.json()
#             ticket = ticket_data["ticket"]
#             print(f"✅ Fetched shared ticket from server at {server_url}")
#             return ticket
#     except httpx.RequestError as e:
#         raise Exception(f"Failed to connect to server: {e}")
#     except httpx.HTTPStatusError as e:
#         raise Exception(f"Server returned error {e.response.status_code}: {e.response.text}")
#     except json.JSONDecodeError as e:
#         raise Exception(f"Invalid ticket format received: {e}")
#     except KeyError as e:
#         raise Exception(f"Missing 'ticket' field in server response")
#     except Exception as e:
#         raise Exception(f"Unexpected error fetching ticket: {e}")

# async def send_blob(doc, author, peer_id: str, data: torch.Tensor):
#     """
#     Send a tensor to another peer in the network.
    
#     Args:
#         doc: Iroh document
#         author: Iroh author for writing
#         peer_id: ID of the recipient peer
#         data: Tensor data to send
#     """
#     try:
#         encoded = json.dumps(data.tolist()).encode()
#         await doc.set_bytes(author, peer_id.encode(), encoded)
#         print(f"📤 Sent to {peer_id}: {data}")
#     except Exception as e:
#         print(f"❌ Failed to send to {peer_id}: {e}")

# async def receive_blob(doc, peer_id: str, node):
#     """
#     Wait for and receive a tensor addressed to this peer.
    
#     Args:
#         doc: Iroh document
#         peer_id: This peer's ID
#         node: Iroh node
        
#     Returns:
#         The received tensor
#     """
#     seen = set()  # Track already processed content hashes
#     while True:
#         try:
#             entries = await doc.get_many(iroh.Query.all(None))
#             for entry in entries:
#                 key = entry.key().decode()
#                 if key != peer_id:
#                     continue
#                 hash = entry.content_hash()
#                 if hash in seen:
#                     continue
#                 seen.add(hash)
#                 content = await node.blobs().read_to_bytes(hash)
#                 tensor = torch.tensor(json.loads(content.decode()))
#                 return tensor
#         except Exception as e:
#             print(f"❌ Polling error for {peer_id}: {e}")
#         await asyncio.sleep(2)  # Poll every 2 seconds

# async def process_once(doc, author, peer_id: str, next_peer: Optional[str], is_first: bool, is_last: bool, local_matrix: torch.Tensor, node):
#     """
#     Process one computation job in the pipeline.
    
#     Args:
#         doc: Iroh document
#         author: Iroh author for writing
#         peer_id: This peer's ID
#         next_peer: ID of the next peer in the pipeline
#         is_first: Whether this is the first machine in the pipeline
#         is_last: Whether this is the last machine in the pipeline
#         local_matrix: The matrix assigned to this peer
#         node: Iroh node
#     """
#     try:
#         if is_first:
#             # First machine waits for job trigger
#             trigger = await receive_blob(doc, TRIGGER_KEY, node)
#             print(f"📥 Received trigger: {trigger}")
#             input_matrix = trigger
#         else:
#             # Other machines wait for input from previous machine
#             input_matrix = await receive_blob(doc, peer_id, node)
#             print(f"📥 Received input: {input_matrix}")
        
#         # Perform matrix multiplication
#         result = torch.matmul(input_matrix, local_matrix)
#         print(f"🔢 Computed result: {result}")
        
#         if is_last:
#             # Last machine stores final result
#             await send_blob(doc, author, FINAL_RESULT_KEY, result)
#             print("✅ Stored final result")
#         elif next_peer:
#             # Pass result to next machine
#             await send_blob(doc, author, next_peer, result)
#             print(f"📤 Sent result to {next_peer}")
#         else:
#             print("⚠️ No next peer specified, cannot send result")
            
#     except Exception as e:
#         print(f"❌ Error in computation: {e}")

colorama_init(autoreset=True)
COLORS = [Fore.CYAN, Fore.MAGENTA, Fore.YELLOW, Fore.GREEN, Fore.BLUE]
PEER_COLOR = COLORS[int(socket.gethostname().__hash__()) % len(COLORS)]  # deterministic per host

async def http_heartbeat_loop(peer_id: str, interval_s: float = 1.0):
    """Send heartbeat to central server over HTTP and exit if server stops responding."""
    consecutive_failures = 0
    max_failures = 5  # ~5 seconds tolerance
    server_url = f"http://{SERVER_HOST}:{SERVER_PORT}/heartbeat"
    deployment_subscribed = False  # Track if we've subscribed to deployment
    hidden_state_subscribed = False  # Track if we've subscribed to hidden-state transfers
    trigger_subscribed = False  # Track if we've subscribed to inference triggers

    # Topology tracking - only update network when peers change
    known_peers = set()  # Track known peer node_ids
    server_added = False  # Track if server was added to network

    async with httpx.AsyncClient(timeout=2.0) as client:
        while True:
            try:
                metrics = get_system_metrics()
                metrics_dict = format_metrics_for_db(metrics)
                total_free_vram = metrics_dict["total_free_vram_gb"]
                payload = {
                    "peer_id": peer_id,
                    "node_id": peer_id,
                    "addresses": node_addr_obj.direct_addresses(),
                    "relay_url": node_addr_obj.relay_url(),
                    **{k: v for k, v in metrics_dict.items() if k != "timestamp"},
                    "gpu_count": len(metrics.gpu_info),
                    "timestamp": int(time.time())
                }
                r = await client.post(server_url, json=payload)
                if r.status_code == 200:
                    #IROH STARTS HERE
                    data = r.json()
                    # print(f"🗒️ Received data: {data}")
                    
                    # Only add server to network once
                    if not server_added:
                        srv_id = data["server_id"]
                        srv_addrs = data["server_addresses"]
                        srv_relay = data["relay_url"]
                        pk = PublicKey.from_string(srv_id)
                        srv_node_addr = NodeAddr(pk, srv_relay, srv_addrs)
                        await current_node.net().add_node_addr(srv_node_addr)
                        server_added = True
                        print(f"🔗 Added server to network: {srv_id[:8]}")
                    
                    # Check for new peers and only add them if they're actually new
                    current_peer_ids = {p["node_id"] for p in data.get("peers", [])}
                    new_peers = current_peer_ids - known_peers
                    
                    for p in data.get("peers", []):
                        remote_peer_id = p["node_id"]
                        if remote_peer_id in new_peers:
                            peer_addrs = p["addresses"]
                            peer_relay = p["relay_url"]
                            pk = PublicKey.from_string(remote_peer_id)
                            peer_node_addr = NodeAddr(pk, peer_relay, peer_addrs)
                            await current_node.net().add_node_addr(peer_node_addr)
                            print(f"🔗 Added new peer to network: {remote_peer_id[:8]}")
                    #IROH ENDS HERE
                    # Update known peers
                    known_peers = current_peer_ids
                    bootstrap_peers = [srv_id] + [p["node_id"] for p in data.get("peers", [])]
                    print(f"🔗 Bootstrap peers: {bootstrap_peers}")
                    
                    if not deployment_subscribed:
                        deployment_topic = bytes("deployment_topic".ljust(32), 'utf-8')[:32]
                        await current_node.gossip().subscribe(deployment_topic, bootstrap_peers, DeploymentGossipCallback(current_node, peer_id))
                        deployment_subscribed = True
                        print(f"🎯 Subscribed to deployment instructions on topic {deployment_topic}")
                    if not hidden_state_subscribed:
                        hidden_state_topic = bytes("hidden_state_topic".ljust(32), 'utf-8')[:32]
                        global hidden_state_gossip_sink
                        hidden_state_gossip_sink = await current_node.gossip().subscribe(
                            hidden_state_topic,
                            bootstrap_peers,
                            HiddenStateCallback(),
                        )
                        hidden_state_subscribed = True
                        print(f"🎯 Subscribed to hidden-state transfers on topic {hidden_state_topic}")
                    if not trigger_subscribed:
                        trigger_topic = bytes("trigger_topic".ljust(32), 'utf-8')[:32]
                        await current_node.gossip().subscribe(trigger_topic, bootstrap_peers, TriggerCallback())
                        trigger_subscribed = True
                        print(f"🎯 Subscribed to inference triggers on topic {trigger_topic}")
                        consecutive_failures = 0  # Reset on success
                    print(f"{PEER_COLOR}💓 Sent heartbeat | CPU {metrics.cpu_percent:.1f}% VRAM {total_free_vram:.1f} GB → ACK {Style.RESET_ALL}")
                else:
                    consecutive_failures += 1
                    print(f"{PEER_COLOR}⚠️ Heartbeat HTTP {r.status_code}{Style.RESET_ALL}")
            except Exception as e:
                consecutive_failures += 1
                print(f"{PEER_COLOR}⚠️ Heartbeat error: {e}{Style.RESET_ALL}")

            if consecutive_failures >= max_failures:
                print(f"{PEER_COLOR}💔 Lost contact with server, shutting down peer{Style.RESET_ALL}")
                os._exit(1)

            await asyncio.sleep(interval_s)

async def main():

    """Main function to run the distributed computation node"""
    global current_node, current_peer_id, node_id_obj, node_addr_obj
    
    bootstrap_peers=[]
    # Set up the asyncio event loop for Iroh
    uniffi_set_event_loop(asyncio.get_running_loop())

    # Set up a unique data directory for this node
    hostname = socket.gethostname()
    data_dir = os.path.join(tempfile.gettempdir(), f"iroh_node_{hostname}")
    
    # Configure and initialize the Iroh node
    options = iroh.NodeOptions()
    options.enable_gossip = True    
    node = await iroh.Iroh.memory_with_options(options)

    peer_id = await node.net().node_id()
    node_id_obj   = await node.net().node_id()
    node_addr_obj = await node.net().node_addr()
    print(f"🤖 Running as peer: {peer_id}")
    
    # Set globals for completion reporting
    current_node = node
    current_peer_id = peer_id # questionable if this is needed

    # start heartbeat loop
    heartbeat_task = asyncio.create_task(http_heartbeat_loop(peer_id))
    # Register this peer in MongoDB
    await register_peer(peer_id, hostname)
    print(f"✅ Registered in MongoDB as {peer_id}")

    await asyncio.sleep(1)
    # # Subscribe to inference triggers
    # trigger_topic = bytes("trigger_topic".ljust(32), 'utf-8')[:32]
    # await node.gossip().subscribe(trigger_topic, bootstrap_peers, TriggerCallback(processor))



    # --- NEW: subscribe to deployment instructions ---
    # deployment_topic = bytes("deployment_topic".ljust(32), 'utf-8')[:32]
    # await node.gossip().subscribe(deployment_topic, bootstrap_peers, DeploymentGossipCallback(node, peer_id))
    # print(f"🎯 Subscribed to deployment instructions on topic {deployment_topic}")




    # Fetch the shared ticket from the server
    # try:  
        # shared_ticket = await get_shared_ticket()
    # except Exception as e:
    #     print(f"❌ Failed to get shared ticket: {e}")
    #     return

    # Join the shared document and create an author for writing
    # doc = await node.docs().join(iroh.DocTicket(shared_ticket))
    # author = await node.authors().create()
    
    # Set global doc for completion reporting   
    # current_doc = doc
    
    # Start HTTP heartbeat loop (1 Hz by default)
    
    # Start deployment instruction monitoring as background task
    # deployment_task = asyncio.create_task(
    #     monitor_deployment_instructions(doc, node, peer_id)
    # )

    try:
        # Wait until this peer is included in the pipeline configuration
        print("⏳ Waiting to be included in the pipeline...")
        while True:
            pipeline = await get_active_peers()
            print(pipeline)
            if peer_id in pipeline:
                break
            await asyncio.sleep(2)

        # Determine this peer's position and role in the pipeline
        index = pipeline.index(peer_id)
        is_first = index == 0
        is_last = index == len(pipeline) - 1
        next_peer = pipeline[index + 1] if not is_last else None
        # local_matrix = MATRIX_MAP.get(index)
        # if local_matrix is None:
        #     print(f"❌ No matrix found for pipeline index {index}")
        #     return

        # print(f"✅ Position: {index} | First: {is_first} | Last: {is_last}")

        # Main processing loop - continuously process computation jobs
        while True:
            print("🔄 Processing...")
            # await process_once(doc, author, peer_id, next_peer, is_first, is_last, local_matrix, node)
            await asyncio.sleep(2)  # Small delay between processing cycles

    except Exception as e:
        print(f"❌ Error in main loop: {e}")
    finally:
        # Cancel background tasks
        heartbeat_task.cancel()
        # deployment_task.cancel()
        
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass
            
        # try:
        #     await deployment_task
        # except asyncio.CancelledError:
        #     pass
        
        # Deregister peer when shutting down
        await deregister_peer(peer_id)
        print(f"👋 Deregistered {peer_id} from pipeline")

if __name__ == "__main__":
    asyncio.run(main()) 