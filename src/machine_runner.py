import asyncio
import socket
import tempfile
import os
import json
import torch
import iroh
import httpx
from pathlib import Path
from typing import List, Dict, Any, Optional
import time
import aiofiles
from colorama import Fore, Style, init as colorama_init
from src.utils.inference_utils import register_inference_hooks, INFERENCE_CONTEXT
import pickle
import numpy as np

## Tensor_Iroh Starts here ###########################
from src.utils.tensor_protocol_adapter import TensorTransport
######################################################
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


# Global TensorTransport instance (lazy-started in main) #######################
tensor_transport: TensorTransport | None = None
peer_ticket_map: dict[str, str] = {}  # peer_id  → ticket string (filled in heartbeat)
current_peer_ticket = None # this is the global variable for the current peer ticket
deployed_model = None # this is the global variable for the deployed model
deployment_status = "idle"  # idle, downloading, loading, ready, failed
start_inference_run = None # this is the global variable for the inference run
assigned_layers_global=[] # this is the global variable for the assigned layers
central_server_ticket = None # this is the global variable for the central server ticket
################################################################################

colorama_init(autoreset=True)
COLORS = [Fore.CYAN, Fore.MAGENTA, Fore.YELLOW, Fore.GREEN, Fore.BLUE]
PEER_COLOR = COLORS[int(socket.gethostname().__hash__()) % len(COLORS)]  # deterministic per host



# async def monitor_inference_tensors():
#     """Monitor for incoming hidden states, residuals, and sampler outputs during inference"""
#     global tensor_transport
#     print("🎯 Starting inference tensor monitor...")
    
#     while True:
#         try:
#             msg = await tensor_transport.recv()
#             if msg is None:
#                 await asyncio.sleep(0.01)
#                 continue
                
#             name = msg.get("name", "")
#             tensor = msg.get("tensor")
            
#             # Parse the tensor name to determine type
#             if "_hidden_state" in name or "_residual" in name:
#                 # Extract request_id and step from name
#                 parts = name.split("_")
#                 request_id = parts[0]
#                 step_idx = int(parts[1].replace("step", ""))
                
#                 # Store in INFERENCE_CONTEXT
#                 if request_id not in INFERENCE_CONTEXT:
#                     INFERENCE_CONTEXT[request_id] = {}
#                 if step_idx not in INFERENCE_CONTEXT[request_id]:
#                     INFERENCE_CONTEXT[request_id][step_idx] = {}
                
#                 if "_residual" in name:
#                     INFERENCE_CONTEXT[request_id][step_idx]["residual"] = tensor
#                 else:
#                     INFERENCE_CONTEXT[request_id][step_idx]["hidden_state"] = tensor
                    
#             elif "_sampler_output" in name:
#                 # Handle sampler output from last peer
#                 parts = name.split("_")
#                 request_id = parts[0]
#                 step_idx = int(parts[1].replace("step", ""))
                
#                 # Unpickle the sampler output
#                 sampler_output = pickle.loads(tensor.tobytes())
                
#                 if request_id not in INFERENCE_CONTEXT:
#                     INFERENCE_CONTEXT[request_id] = {}
#                 if step_idx not in INFERENCE_CONTEXT[request_id]:
#                     INFERENCE_CONTEXT[request_id][step_idx] = {}
                    
#                 INFERENCE_CONTEXT[request_id][step_idx]["sampler_output"] = sampler_output
                
#         except Exception as e:
#             print(f"❌ Error in inference tensor monitor: {e}")
#             await asyncio.sleep(0.1)
#         except asyncio.CancelledError:
#             print("monitor_inference_tensors cancelled, exiting cleanly.")
#             return


# async def monitor_tensor_transport_for_deployment():
#     """
#     Continuously monitor tensor_transport for incoming deployment instructions.
#     When a deployment instruction is received, handle it accordingly.
#     """
#     global tensor_transport
#     print("🚦 Starting tensor_transport deployment instruction monitor loop...")
#     while True:
#         try:
#             # Wait for any tensor to arrive
#             msg = await tensor_transport.recv()
#             if msg is None:
#                 await asyncio.sleep(0.1)
#                 continue

#             # Expecting: {"tensor": torch.Tensor}
#             tensor = msg.get("tensor")
#             if tensor is None:
#                 print("⚠️ Received message without tensor payload")
#                 continue

#             # Convert tensor (np.uint8) back to bytes
#             if hasattr(tensor, "numpy"):
#                 arr = tensor.numpy()
#             else:
#                 arr = tensor
#             if arr.dtype != np.uint8:
#                 print(f"⚠️ Received tensor with unexpected dtype: {arr.dtype}")
#                 continue
#             instruction_data = arr.tobytes()

#             # Parse and handle deployment instruction
#             try:
#                 instruction = json.loads(instruction_data.decode())
#                 if instruction.get("action") == "deploy_model":
#                     instructions = instruction.get("instructions", {})
#                     print("\n" + "="*80)
#                     print("🌟🌟🌟 [DEPLOYMENT INSTRUCTION TENSOR RECEIVED] 🌟🌟🌟")
#                     print(f"🟣 Raw tensor payload (full): {tensor}")
#                     print(f"🟣 Numpy array (full): {arr}")
#                     print(f"🟣 Numpy array (as list, full): {arr.tolist() if hasattr(arr, 'tolist') else arr}")
#                     print(f"📨 Received deployment instruction for {instructions.get('model_name', 'unknown')}")
#                     print("="*80 + "\n")
#                     # Deploy model in background (non-blocking)
#                     asyncio.create_task(deploy_model_from_instructions(instructions))
#                 else:
#                     print(f"⚠️  Unknown instruction action: {instruction.get('action')}")
#             except Exception as e:
#                 print(f"❌ Error handling deployment instruction: {e}")

#         except Exception as e:
#             print(f"❌ Error in tensor_transport monitor loop: {e}")
#             await asyncio.sleep(1)
#         except asyncio.CancelledError:
#             print("monitor_tensor_transport_for_deployment cancelled, exiting cleanly.")
#             return


# async def monitor_inference_context():
#     """Debug monitor for INFERENCE_CONTEXT - shows what data is being stored"""
#     print("📊 Starting INFERENCE_CONTEXT monitor (debug)...")
    
#     while True:
#         try:
#             if INFERENCE_CONTEXT:
#                 print(f"\n{'='*60}")
#                 print(f"📊 INFERENCE_CONTEXT Status @ {time.strftime('%H:%M:%S')}")
#                 print(f"{'='*60}")
                
#                 for req_id, req_data in INFERENCE_CONTEXT.items():
#                     print(f"\n📋 Request: {req_id}")
#                     for step_idx, step_data in req_data.items():
#                         if isinstance(step_data, dict):
#                             print(f"   Step {step_idx}:")
#                             for key, value in step_data.items():
#                                 if hasattr(value, 'shape'):
#                                     print(f"     - {key}: tensor with shape {value.shape}")
#                                 else:
#                                     print(f"     - {key}: {type(value).__name__}")
#                         else:
#                             print(f"   {step_idx}: {type(step_data).__name__}")
                
#                 print(f"{'='*60}\n")
            
#             await asyncio.sleep(5)  # Check every 5 seconds
            
#         except asyncio.CancelledError:
#             print("monitor_inference_context cancelled, exiting cleanly.")
#             return
#         except Exception as e:
#             print(f"❌ Error in INFERENCE_CONTEXT monitor: {e}")
#             await asyncio.sleep(5)


# ============================================================================
# UNIFIED MESSAGE GATEWAY - SINGLE POINT FOR ALL TENSOR TRANSPORT MESSAGES
# ============================================================================

async def unified_message_gateway():
    """
    Single gateway that receives ALL messages from tensor_transport and routes them
    to the appropriate handlers. This prevents multiple monitors from competing
    for the same message queue.
    """
    global tensor_transport, start_inference_run, current_peer_ticket
    print("🌐 Starting UNIFIED Message Gateway...")
    
    while True:
        try:
            # Single point of message reception
            msg = await tensor_transport.recv()
            if msg is None:
                await asyncio.sleep(0.01)
                continue
                
            # Get message metadata
            name = msg.get("name", "")
            tensor = msg.get("tensor")
            
            if tensor is None:
                print("⚠️ Received message without tensor payload")
                continue
            
            print(f"📨 Gateway received message: '{name}' (tensor shape: {tensor.shape if hasattr(tensor, 'shape') else 'unknown'})")
            
            # Route message based on name/content
            try:
                # 1. DEPLOYMENT INSTRUCTIONS (highest priority)
                if name.lower() in ["deploy", "deployment"] or "deploy" in name.lower():
                    await handle_deployment_message(tensor)
                    
                # 2. INFERENCE TRIGGERS
                elif name.lower() in ["inference", "inference_trigger"] or "inference" in name.lower():
                    await handle_inference_trigger_message(tensor)
                    
                # 3. INFERENCE DATA (hidden states, residuals, sampler outputs)
                elif any(keyword in name for keyword in ["_hidden_state", "_residual", "_sampler_output", "_combined"]):
                    await handle_inference_data_message(name, tensor)
                    
                # 4. UNKNOWN MESSAGE TYPE
                else:
                    print(f"⚠️ Unknown message type: '{name}' - attempting to parse as JSON")
                    await handle_unknown_message(name, tensor)
                    
            except Exception as e:
                print(f"❌ Error routing message '{name}': {e}")
                import traceback
                traceback.print_exc()
                
        except asyncio.CancelledError:
            print("unified_message_gateway cancelled, exiting cleanly.")
            return
        except Exception as e:
            print(f"❌ Error in unified message gateway: {e}")
            await asyncio.sleep(0.1)


async def handle_deployment_message(tensor):
    """Handle deployment instruction messages"""
    try:
        # Convert tensor to bytes
        if hasattr(tensor, "numpy"):
            arr = tensor.numpy()
        else:
            arr = tensor
            
        if arr.dtype != np.uint8:
            print(f"⚠️ Deployment tensor has unexpected dtype: {arr.dtype}")
            return
            
        instruction_data = arr.tobytes()
        instruction = json.loads(instruction_data.decode())
        
        if instruction.get("action") == "deploy_model":
            instructions = instruction.get("instructions", {})
            print("\n" + "="*80)
            print("🌟🌟🌟 [DEPLOYMENT INSTRUCTION RECEIVED] 🌟🌟🌟")
            print(f"📨 Model: {instructions.get('model_name', 'unknown')}")
            print(f"📋 Layers: {instructions.get('assigned_layers', [])}")
            print(f"🔗 Is first: {instructions.get('is_first_peer', False)}")
            print(f"🔗 Is last: {instructions.get('is_last_peer', False)}")
            print("="*80 + "\n")
            
            # Deploy model in background
            asyncio.create_task(deploy_model_from_instructions(instructions))
        else:
            print(f"⚠️ Unknown deployment action: {instruction.get('action')}")
            
    except Exception as e:
        print(f"❌ Error handling deployment message: {e}")
        import traceback
        traceback.print_exc()


async def handle_inference_trigger_message(tensor):
    """Handle inference trigger messages"""
    global start_inference_run, current_peer_ticket
    
    try:
        # Convert tensor to bytes
        if hasattr(tensor, "numpy"):
            arr = tensor.numpy()
        else:
            arr = tensor
            
        if arr.dtype != np.uint8:
            print(f"⚠️ Inference trigger tensor has unexpected dtype: {arr.dtype}")
            return
            
        trigger_data = arr.tobytes()
        trigger = json.loads(trigger_data.decode())
        
        if trigger.get("action") != "start_inference":
            print(f"⚠️ Not a start_inference action: {trigger.get('action')}")
            return
            
        request_id = trigger.get("request_id")
        pipeline = trigger.get("pipeline")
        
        if not request_id or not pipeline:
            print(f"❌ Invalid inference trigger: missing request_id or pipeline")
            return
            
        print(f"\n{'='*80}")
        print(f"🚀 INFERENCE TRIGGER RECEIVED 🚀")
        print(f"📋 Request ID: {request_id}")
        print(f"🔗 Pipeline: {pipeline}")
        print(f"📝 Input: {trigger.get('input_text', '')[:50]}...")
        print(f"{'='*80}\n")
        
        # Initialize INFERENCE_CONTEXT for this request
        if request_id not in INFERENCE_CONTEXT:
            INFERENCE_CONTEXT[request_id] = {}
        
        # Check if inference runner is initialized
        if not start_inference_run:
            print("❌ start_inference_run not initialized! Cannot start inference.")
            return
        
        # ALL PEERS should start inference!
        peer_position = pipeline.index(current_peer_ticket) if current_peer_ticket in pipeline else -1
        
        if peer_position == -1:
            print(f"❌ This peer ({current_peer_ticket}) is not in the pipeline!")
            return
            
        is_first_peer = (peer_position == 0)
        
        print(f"📍 This peer is {'FIRST' if is_first_peer else f'#{peer_position+1}'} in the pipeline")
        print(f"{'✅ Starting inference as FIRST peer' if is_first_peer else '⏳ Starting inference and waiting for tensors from previous peer'}")
        
        # Start inference in background thread for ALL peers
        try:
            from vllm import SamplingParams
            loop = asyncio.get_running_loop()
            
            # Extract parameters
            input_text = trigger.get("input_text", "")
            sampling_params = SamplingParams(**trigger.get("sampling_params", {}))
            assigned_layers = trigger.get("assigned_layers", {})
            
            print(f"🏃 Starting inference run in background thread...")
            future = loop.run_in_executor(
                None,
                start_inference_run,
                request_id,
                pipeline,
                input_text,
                sampling_params,
                assigned_layers,
            )
            
            print(f"✅ Inference started for request {request_id}")
            
        except Exception as e:
            print(f"❌ Failed to start inference: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"❌ Error handling inference trigger: {e}")
        import traceback
        traceback.print_exc()


async def handle_inference_data_message(name: str, tensor):
    """Handle inference data messages (combined tensors or sampler outputs)"""
    try:
        print(f"📊 Processing inference data: {name}")
        
        # Parse the tensor name to determine type
        if "_combined" in name:
            # Name format: {request_id}_step{step_idx}_combined
            # Example: req_1753809548666_0_step0_combined
            
            # Find the last occurrence of "_step" to handle request_ids with underscores
            step_marker_idx = name.rfind("_step")
            if step_marker_idx == -1:
                print(f"❌ Invalid tensor name format (no _step found): {name}")
                return
            
            # Extract request_id (everything before the last _step)
            request_id = name[:step_marker_idx]
            
            # Extract the step number
            rest = name[step_marker_idx + 5:]  # Skip "_step"
            step_idx = int(rest.split("_")[0])
            
            print(f"📋 Parsed: request_id='{request_id}', step={step_idx}, type='combined'")
            
            # Unstack the combined tensor
            if tensor.shape[0] != 2:
                print(f"❌ Invalid combined tensor shape: {tensor.shape}")
                return
                
            hidden_state = tensor[0]
            residual = tensor[1]
            
            # Store in INFERENCE_CONTEXT
            if request_id not in INFERENCE_CONTEXT:
                INFERENCE_CONTEXT[request_id] = {}
            if str(step_idx) not in INFERENCE_CONTEXT[request_id]:
                INFERENCE_CONTEXT[request_id][str(step_idx)] = {}
            
            INFERENCE_CONTEXT[request_id][str(step_idx)]["hidden_state"] = hidden_state
            INFERENCE_CONTEXT[request_id][str(step_idx)]["residual"] = residual
            print(f"✅ Stored both hidden_state and residual for {request_id} step {step_idx}")
                
        elif "_sampler_output" in name:
            # Handle sampler output from last peer
            step_marker_idx = name.rfind("_step")  # Use rfind here too
            if step_marker_idx == -1:
                print(f"❌ Invalid sampler output name format: {name}")
                return
                
            request_id = name[:step_marker_idx]
            rest = name[step_marker_idx + 5:]
            step_idx = int(rest.split("_")[0])
            
            # Convert tensor to numpy array first, then unpickle
            if hasattr(tensor, "numpy"):
                # PyTorch tensor - convert to numpy
                arr = tensor.numpy()
            else:
                # Already numpy
                arr = tensor
            
            # Unpickle the sampler output
            sampler_output = pickle.loads(arr.tobytes())
            
            if request_id not in INFERENCE_CONTEXT:
                INFERENCE_CONTEXT[request_id] = {}
            if str(step_idx) not in INFERENCE_CONTEXT[request_id]:
                INFERENCE_CONTEXT[request_id][str(step_idx)] = {}
                
            INFERENCE_CONTEXT[request_id][str(step_idx)]["sampler_output"] = sampler_output
            print(f"✅ Stored sampler_output for {request_id} step {step_idx}")
            
    except Exception as e:
        print(f"❌ Error handling inference data '{name}': {e}")
        import traceback
        traceback.print_exc()

async def handle_unknown_message(name: str, tensor):
    """Handle unknown message types by attempting to parse Was JSON"""
    try:
        # Convert tensor to bytes and try to parse as JSON
        if hasattr(tensor, "numpy"):
            arr = tensor.numpy()
        else:
            arr = tensor
            
        if arr.dtype != np.uint8:
            print(f"⚠️ Unknown message tensor has unexpected dtype: {arr.dtype}")
            return
            
        data = arr.tobytes()
        parsed = json.loads(data.decode())
        
        print(f"📋 Unknown message '{name}' parsed as JSON:")
        print(f"   Action: {parsed.get('action', 'unknown')}")
        print(f"   Keys: {list(parsed.keys())}")
        
        # Try to route based on action
        action = parsed.get("action")
        if action == "deploy_model":
            print("🔄 Routing unknown message to deployment handler")
            await handle_deployment_message(tensor)
        elif action == "start_inference":
            print("🔄 Routing unknown message to inference trigger handler")
            await handle_inference_trigger_message(tensor)
        else:
            print(f"⚠️ Cannot route unknown action: {action}")
            
    except Exception as e:
        print(f"❌ Error handling unknown message '{name}': {e}")


async def debug_inference_context_monitor():
    """Separate debug monitor that doesn't compete with message reception"""
    print("📊 Starting INFERENCE_CONTEXT debug monitor...")
    
    while True:
        try:
            if INFERENCE_CONTEXT:
                print(f"\n{'='*60}")
                print(f"📊 INFERENCE_CONTEXT @ {time.strftime('%H:%M:%S')}")
                print(f"{'='*60}")
                
                for req_id, req_data in INFERENCE_CONTEXT.items():
                    print(f"\n📋 Request: {req_id}")
                    for step_idx, step_data in req_data.items():
                        if isinstance(step_data, dict):
                            print(f"   Step {step_idx}:")
                            for key, value in step_data.items():
                                if hasattr(value, 'shape'):
                                    print(f"     - {key}: tensor {value.shape}")
                                else:
                                    print(f"     - {key}: {type(value).__name__}")
                        else:
                            print(f"   {step_idx}: {type(step_data).__name__}")
                
                print(f"{'='*60}\n")
            
            await asyncio.sleep(5)  # Check every 5 seconds
            
        except asyncio.CancelledError:
            print("debug_inference_context_monitor cancelled, exiting cleanly.")
            return
        except Exception as e:
            print(f"❌ Error in debug monitor: {e}")
            await asyncio.sleep(5)

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
            dtype="float16"
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
    global deployed_model, deployment_status, start_inference_run, tensor_transport
    # get the model name
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
        print(f"   Next peer ticket: {instructions['next_peer_ticket']}")
        
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
        # Pass the full pipeline of peer tickets to register_inference_hooks
        start_inference_run = register_inference_hooks(
            llm=deployed_model,
            node=tensor_transport,
            peer_id=current_peer_ticket,
            server_url=server_url,
            next_peer_ticket=instructions.get("next_peer_ticket"),
            pipeline=instructions.get("pipeline")  # Send all peer tickets
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


async def report_deployment_completion(model_name: str, success: bool):
    """
    Notify the central server that this peer has finished deploying.
    """
    url = f"http://{SERVER_HOST}:{SERVER_PORT}/deployment_complete"
    payload = {
        "model_name": model_name,
        "peer_id": current_peer_ticket,
        "success": success
    }
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
        print(f"📤 Reported deployment completion: {payload}")
    except Exception as e:
        print(f"❌ Failed to report deployment completion: {e}")



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

async def http_heartbeat_loop(current_peer_ticket: str, interval_s: float = 1.0):
    """Send heartbeat to central server over HTTP and exit if server stops responding."""
    global central_server_ticket
    consecutive_failures = 0
    max_failures = 10 # 10 seconds tolerance
    server_url = f"http://{SERVER_HOST}:{SERVER_PORT}/heartbeat"
    server_added = False
        
    async with httpx.AsyncClient(timeout=2.0) as client:
        while True:
            try:
                metrics = get_system_metrics()
                metrics_dict = format_metrics_for_db(metrics)
                total_free_vram = metrics_dict["total_free_vram_gb"]
                payload = {
                    "peer_id": current_peer_ticket,
                    **{k: v for k, v in metrics_dict.items() if k != "timestamp"},
                    "gpu_count": len(metrics.gpu_info),
                    "timestamp": int(time.time())
                }
                response = await client.post(server_url, json=payload)
                if response.status_code == 200:
                    data = response.json()
                    # Only add server to network once
                    if not server_added:
                        central_server_ticket = data["central_server_ticket"]
                        print(f"🔗 Central server ticket: {central_server_ticket}")
                        server_added = True
                    print(f"{PEER_COLOR}💓 Sent heartbeat | CPU {metrics.cpu_percent:.1f}% VRAM {total_free_vram:.1f} GB → ACK {Style.RESET_ALL}")
                else:
                    consecutive_failures += 1
                    print(f"{PEER_COLOR}⚠️ Heartbeat HTTP {response.status_code}{Style.RESET_ALL}")
            except Exception as e:
                consecutive_failures += 1
                print(f"{PEER_COLOR}⚠️ Heartbeat error: {e}{Style.RESET_ALL}")
            if consecutive_failures >= max_failures:
                print(f"{PEER_COLOR}💔 Lost contact with server, shutting down peer{Style.RESET_ALL}")
                os._exit(1)
            await asyncio.sleep(interval_s)




async def main():
    """Main function to run the distributed computation node"""
    global current_peer_ticket, peer_ticket_map, tensor_transport
    # Set up Tensor_Iroh and get the ticket #################################
    tensor_transport = TensorTransport()
    await tensor_transport.start()
    current_peer_ticket = tensor_transport.ticket
    print(f"🪪 TensorTransport started – ticket:\n{current_peer_ticket}\n")
    # Set up a unique data directory for this node
    hostname = socket.gethostname()
    ticket_and_hostname = f"Ticket: {current_peer_ticket}, Hostname: {hostname}"
    peer_ticket_map[ticket_and_hostname] = current_peer_ticket # check if we even need ticket_and_hostname
    print(f"🤖 Running as peer: {current_peer_ticket}")
    heartbeat_task = asyncio.create_task(http_heartbeat_loop(current_peer_ticket))
    await register_peer(current_peer_ticket, hostname)
    print(f"✅ Registered in MongoDB as {current_peer_ticket}")
    print("Starting unified message gateway...")
    gateway_task = asyncio.create_task(unified_message_gateway())
    debug_monitor_task = asyncio.create_task(debug_inference_context_monitor())
    await asyncio.sleep(1)


    try:
        # Wait until this peer is included in the pipeline configuration
        print("⏳ Waiting to be included in the pipeline...")
        while True:
            pipeline = await get_active_peers()
            print(f"🔗 Pipeline: {pipeline}")
            if current_peer_ticket in pipeline:
                print(f"✅ Included in pipeline as {current_peer_ticket}")
                break
            await asyncio.sleep(2)

        # Determine this peer's position and role in the pipeline
        index = pipeline.index(current_peer_ticket)
        is_first = index == 0
        is_last = index == len(pipeline) - 1
        next_peer = pipeline[index + 1] if not is_last else None

        print(f"✅ Position: {index} | First: {is_first} | Last: {is_last}")

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
        gateway_task.cancel()
        debug_monitor_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass
        try:
            await gateway_task
        except asyncio.CancelledError:
            pass
        try:
            await debug_monitor_task
        except asyncio.CancelledError:
            pass
        # Deregister peer when shutting down
        # await deregister_peer(current_peer_ticket)
        print(f"👋 Deregistered {current_peer_ticket} from pipeline")

    #########################################################################

if __name__ == "__main__":
    asyncio.run(main()) 





# # IROH STARTS HERE
# class DeploymentGossipCallback(iroh.GossipMessageCallback):
#     "Handle deployment instructions received via gossip"
#     def __init__(self, node, peer_id):
#         super().__init__()
#         self.node = node
#         self.peer_id = peer_id

#     async def on_message(self, msg):
#         print("="*100)
#         print(f"�� [DEBUG] DeploymentGossipCallback received message")
#         t = msg.type()  # ✅ Call the method
#         print(f"🔍 [DEBUG] Message type: {t}")
#         print("="*100)
        
#         if t == MessageType.JOINED:
#             print("🔎 Deployment mesh membership:", msg.as_joined())
#             return
            
#         if t == MessageType.RECEIVED:
#             print(f"�� [DEBUG] Processing RECEIVED deployment message")
#             rc = msg.as_received()
#             print(f"🔍 [DEBUG] Message content length: {len(rc.content)} bytes")
            
#             try:
#                 payload = json.loads(rc.content.decode())  # ✅ Use content, not payload
#                 print(f"🔍 [DEBUG] Parsed JSON payload: {list(payload.keys())}")
                
#                 # Check if this is a deployment instruction
#                 if payload.get("action") != "deploy_model":
#                     print(f"🔍 [DEBUG] Not a deployment instruction, skipping")
#                     return
                
#                 # check if the target peer is this peer
#                 target_peer_id = payload.get("target_peer_id")
#                 if target_peer_id != self.peer_id:
#                     print(f"🔍 [DEBUG] Not a deployment instruction for this peer, skipping, {target_peer_id}")
#                     return
                    
#                 instructions = payload.get("instructions", {})
#                 print(f"📨 Received deployment instructions for {self.peer_id}: {instructions}")
                
#                 # Deploy model in background
#                 asyncio.create_task(deploy_model_from_instructions(instructions))
                
#             except Exception as e:
#                 print(f"❌ Error handling deployment instruction: {e}")
#                 print(f"❌ Exception type: {type(e)}")
#                 import traceback
#                 traceback.print_exc()


# IROH STARTS HERE
# class TriggerCallback(iroh.GossipMessageCallback):
#     """Handles the initial inference trigger from the server"""
#     async def on_message(self, msg):
#         if msg.type() != MessageType.RECEIVED:
#             print(f"🔍 [DEBUG] TriggerCallback received non-RECEIVED message")
#             return
        
#         try:
#             payload = json.loads(msg.as_received().content.decode())
#             if payload.get("action") != "start_inference":
#                 print(f"🔍 [DEBUG] TriggerCallback received non-start_inference message")
#                 return
            
#             pipeline = payload.get("pipeline")
#             request_id = payload.get("request_id")
            
#             # This peer is not in the pipeline, ignore
#             if not (request_id and current_peer_id in pipeline):
#                 return

#             # only the first peer gets to start the inference
#             is_first_peer = pipeline[0] == current_peer_id
#             print(f"🔍 [DEBUG] TriggerCallback initializing INFERENCE_CONTEXT for {request_id} (is_first_peer: {is_first_peer})")

#             # Initialize context for this request (no futures needed anymore)
#             if request_id not in INFERENCE_CONTEXT:
#                 INFERENCE_CONTEXT[request_id] = {}
            
#             if not start_inference_run:
#                 print(f"🔍 [DEBUG] start_inference_run is not set, cannot proceed.")
#                 return
            
#             # Start the inference run in a background thread
#             input_text = payload.get("input_text")

#             # Import SamplingParams locally to avoid top-level import issues
#             from vllm import SamplingParams

#             loop = asyncio.get_running_loop()
#             loop.run_in_executor(
#                 None,
#                 start_inference_run,
#                 payload["request_id"],
#                 payload["pipeline"],
#                 input_text,
#                 SamplingParams(**payload["sampling_params"]),
#                 payload["assigned_layers"],
#             )

#         except Exception as e:
#             print(f"❌ Error in TriggerCallback: {e}")
#             import traceback
#             traceback.print_exc()

#         print(f"🔍 [DEBUG] TriggerCallback completed for {payload.get('request_id')}")
# IROH ENDS HERE



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




# async def monitor_tensor_transport_for_inference_trigger():
#     """
#     Dedicated monitor for inference trigger messages from the server.
#     Only the first peer in the pipeline will actually start the inference.
#     """
#     global tensor_transport, start_inference_run, current_peer_ticket
#     print("🎯 Starting inference trigger monitor...")
    
#     while True:
#         try:
#             # Wait for any message to arrive
#             msg = await tensor_transport.recv()
#             if msg is None:
#                 await asyncio.sleep(0.01)
#                 continue
                
#             # Check if this is an inference trigger by name
#             name = msg.get("name", "")
#             if "inference" not in name.lower():
#                 continue  # Not an inference message, skip
                
#             # Get the tensor payload
#             tensor = msg.get("tensor")
#             if tensor is None:
#                 print("⚠️ Received inference message without tensor payload")
#                 continue
                
#             # Convert tensor (np.uint8) back to bytes
#             if hasattr(tensor, "numpy"):
#                 arr = tensor.numpy()
#             else:
#                 arr = tensor
#             if arr.dtype != np.uint8:
#                 print(f"⚠️ Received inference tensor with unexpected dtype: {arr.dtype}")
#                 continue
                
#             trigger_data = arr.tobytes()
            
#             # Parse the inference trigger
#             try:
#                 trigger = json.loads(trigger_data.decode())
#                 if trigger.get("action") != "start_inference":
#                     continue  # Not a start_inference action
                    
#                 request_id = trigger.get("request_id")
#                 pipeline = trigger.get("pipeline")
                
#                 if not request_id or not pipeline:
#                     print(f"❌ Invalid inference trigger: missing request_id or pipeline")
#                     continue
                
#                 print(f"\n{'='*80}")
#                 print(f"🚀 INFERENCE TRIGGER RECEIVED 🚀")
#                 print(f"📋 Request ID: {request_id}")
#                 print(f"🔗 Pipeline: {pipeline}")
#                 print(f"📝 Input: {trigger.get('input_text', '')[:50]}...")
#                 print(f"{'='*80}\n")
                
#                 # Check if this peer is the first in the pipeline
#                 if current_peer_ticket == pipeline[0]:
#                     print(f"✅ This peer ({current_peer_ticket}) is FIRST in pipeline - starting inference!")
                    
#                     # Initialize INFERENCE_CONTEXT for this request
#                     if request_id not in INFERENCE_CONTEXT:
#                         INFERENCE_CONTEXT[request_id] = {}
                    
#                     # Check if inference runner is initialized
#                     if not start_inference_run:
#                         print("❌ start_inference_run not initialized! Cannot start inference.")
#                         continue
                    
#                     # Start inference in background thread
#                     try:
#                         from vllm import SamplingParams
#                         loop = asyncio.get_running_loop()
                        
#                         # Extract parameters
#                         input_text = trigger.get("input_text", "")
#                         sampling_params = SamplingParams(**trigger.get("sampling_params", {}))
#                         assigned_layers = trigger.get("assigned_layers", {})
                        
#                         print(f"🏃 Starting inference run in background thread...")
#                         future = loop.run_in_executor(
#                             None,
#                             start_inference_run,
#                             request_id,
#                             pipeline,
#                             input_text,
#                             sampling_params,
#                             assigned_layers,
#                         )
                        
#                         # Optionally, you can track the future
#                         print(f"✅ Inference started for request {request_id}")
                        
#                     except Exception as e:
#                         print(f"❌ Failed to start inference: {e}")
#                         import traceback
#                         traceback.print_exc()
#                 else:
#                     print(f"ℹ️ This peer ({current_peer_ticket}) is NOT first in pipeline - ignoring trigger")
#                     print(f"   First peer should be: {pipeline[0]}")
                    
#             except json.JSONDecodeError as e:
#                 print(f"❌ Failed to parse inference trigger JSON: {e}")
#             except Exception as e:
#                 print(f"❌ Error handling inference trigger: {e}")
#                 import traceback
#                 traceback.print_exc()