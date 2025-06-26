import asyncio
import iroh
import torch
import json
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import FileResponse, Response
from iroh.iroh_ffi import uniffi_set_event_loop
from pydantic import BaseModel
from typing import Optional, List
import uuid
import time
from datetime import datetime
from pathlib import Path

from src.config.settings import (
    SERVER_HOST,
    SERVER_PORT,
    DEFAULT_QBITS,
    DEFAULT_CONFIG_FILENAME
)
from src.utils.db_utils import (
    setup_collections,
    get_active_peers,
    update_peer_metrics,
    cleanup_inactive_peers,
    get_peer_metrics,
    db as _db,
    PEERS_COLLECTION
)
from src.utils.gpu_utils import get_system_metrics, format_metrics_for_db
from src.utils.model_utils import (
    download_config,
    estimate_parameters,
    estimate_vram,
    calculate_max_layers_for_peer,
    distribute_layers_across_peers
)
from src.utils.sharding_utils import shard_model_by_layers

# Initialize FastAPI application
app = FastAPI(title="Iroh Tandemn Server")

# Global variables for Iroh node and document management
node = None  # Iroh node instance
doc = None   # Iroh document instance
ticket = None  # Document sharing ticket
server_peer_id = None  # Unique identifier for this server instance

# Task tracking for background operations
background_tasks = {}  # Dict to track running background tasks

# Deployment tracking to prevent infinite loops
active_deployments = {}  # Track ongoing deployments by model_name

# Constants for document keys
TRIGGER_KEY = "job_trigger"  # Key for job initiation
FINAL_RESULT_KEY = "final_result"  # Key for final computation result

async def monitor_and_acknowledge_heartbeats():
    """Monitor peer heartbeats and send acknowledgments back to keep the connection alive."""
    global doc, node
    
    print("💓 Starting heartbeat acknowledgment monitor...")
    seen_heartbeats = set()
    
    while True:
        try:
            author = await node.authors().create()
            entries = await doc.get_many(iroh.Query.all(None))
            
            for entry in entries:
                key = entry.key().decode()
                hash_value = entry.content_hash()
                
                # Skip if we've already processed this heartbeat
                if hash_value in seen_heartbeats:
                    continue
                
                # Look for heartbeat messages from peers
                if key.startswith("heartbeat_") and not key.startswith("heartbeat_ack_"):
                    # Extract peer_id from heartbeat key: heartbeat_{peer_id}_{timestamp}
                    try:
                        parts = key.split("_")
                        if len(parts) >= 3:
                            peer_id = parts[1]
                            timestamp = parts[2]
                            
                            # Send acknowledgment back to peer
                            ack_key = f"heartbeat_ack_{peer_id}_{timestamp}"
                            ack_value = json.dumps({
                                "server_id": server_peer_id,
                                "ack_timestamp": int(time.time() * 1000),
                                "status": "alive"
                            }).encode()
                            
                            await doc.set_bytes(author, ack_key.encode(), ack_value)
                            seen_heartbeats.add(hash_value)
                            
                    except Exception as e:
                        print(f"❌ Error processing heartbeat {key}: {e}")
            
        except Exception as e:
            print(f"❌ Error in heartbeat monitor: {e}")
        
        await asyncio.sleep(2)  # Check every 2 seconds

class ModelEstimationRequest(BaseModel):
    model_id: str
    hf_token: str
    qbits: int = DEFAULT_QBITS
    filename: str = DEFAULT_CONFIG_FILENAME

class ModelShardingRequest(BaseModel):
    model_id: str
    hf_token: str
    model_layers_key: str = "model.layers"
    cache_dir: Optional[str] = None

class ModelDeploymentRequest(BaseModel):
    model_name: str
    shard_folder: str



@app.on_event("startup")
async def startup():
    """Initialize the Iroh node and document on server startup"""
    global node, doc, ticket, server_peer_id

    print("🚀 Starting Iroh Tandemn server...")
    # Configure async event loop for Iroh
    uniffi_set_event_loop(asyncio.get_running_loop())

    # Set up MongoDB collections
    await setup_collections()
    print("✅ MongoDB collections configured")

    # Reset peer activity status on fresh server start
    reset_result = await _db[PEERS_COLLECTION].update_many({}, {"$set": {"is_active": False}})
    if reset_result.modified_count > 0:
        print(f"🔄 Reset {reset_result.modified_count} peer(s) to inactive state")
    else:
        print("🔄 No previously active peers found to reset")

    # Initialize Iroh node with document support
    options = iroh.NodeOptions()
    options.enable_docs = True
    node = await iroh.Iroh.memory_with_options(options)

    # Create and share document
    doc = await node.docs().create()
    ticket = await doc.share(iroh.ShareMode.WRITE, iroh.AddrInfoOptions.RELAY_AND_ADDRESSES)
    server_peer_id = await node.net().node_id()

    # Record initial system metrics
    author = await node.authors().create()
    key = server_peer_id.encode()
    metrics = get_system_metrics()
    formatted_metrics = format_metrics_for_db(metrics)
    value = f"CPU: {metrics.cpu_percent}%\nRAM: {metrics.ram_percent}%".encode()

    try:
        # Store system metrics in the document and MongoDB
        await doc.set_bytes(author, key, value)
        await update_peer_metrics(server_peer_id, formatted_metrics)
        print(f"✅ Server metrics stored with key {server_peer_id}")
    except Exception as e:
        print(f"❌ Failed to send server metrics: {e}")

    # Join the document to enable updates
    doc = await node.docs().join(ticket)
    print("✅ Iroh node started")
    print("📎 SHARE THIS TICKET WITH ALL INTERNAL MACHINES:\n")
    print(str(ticket) + "\n")
    
    # Start heartbeat acknowledgment monitor
    asyncio.create_task(monitor_and_acknowledge_heartbeats())

@app.get("/health")
async def health():
    """Endpoint to check health and status of all connected machines"""
    try:
        # Get active peers from MongoDB
        active_peers = await get_active_peers()
        results = []

        # Get latest metrics for each active peer
        for peer_id in active_peers:
            try:
                # Get metrics from MongoDB time-series collection
                metrics_history = await get_peer_metrics(peer_id, time_window=60)  # Last minute
                if metrics_history:
                    latest_metrics = metrics_history[0]["metrics"]  # Most recent metrics
                    results.append({
                        "machine_id": peer_id,
                        "metrics": latest_metrics,
                        "timestamp": metrics_history[0]["timestamp"]
                    })
            except Exception as inner_e:
                print(f"❌ Failed reading metrics for {peer_id}: {inner_e}")
                results.append({
                    "machine_id": peer_id,
                    "error": str(inner_e)
                })

        # Clean up inactive peers
        await cleanup_inactive_peers()

        return {"status": "success", "machines": results}

    except Exception as e:
        print(f"/health failed: {repr(e)}")
        return {"status": "error", "detail": str(e)}

@app.get("/metrics/{peer_id}")
async def get_peer_metrics_endpoint(peer_id: str, time_window: int = 300):
    """
    Get metrics history for a specific peer.
    
    Args:
        peer_id: The peer ID to get metrics for
        time_window: Time window in seconds (default: 5 minutes)
    """
    try:
        metrics = await get_peer_metrics(peer_id, time_window)
        return {
            "status": "success",
            "peer_id": peer_id,
            "metrics_history": metrics
        }
    except Exception as e:
        return {
            "status": "error",
            "detail": str(e)
        }

@app.get("/ticket")
async def get_ticket():
    """Endpoint to retrieve the document sharing ticket"""
    return {"ticket": str(ticket)}

@app.post("/estimate_model")
async def estimate_model(request: ModelEstimationRequest):
    """
    Estimate model parameters and VRAM requirements.
    
    Args:
        request: ModelEstimationRequest containing model_id, hf_token, qbits, and filename
        
    Returns:
        Dictionary containing total parameters and estimated VRAM
    """
    try:
        config = await download_config(
            request.model_id,
            request.hf_token,
            request.filename
        )
        total_params = estimate_parameters(config)
        vram_gb = estimate_vram(total_params, request.qbits)
        
        return {
            "status": "success",
            "total_params": total_params,
            "estimated_vram_gb": vram_gb
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.post("/identify_peers")
async def identify_peers(request: ModelEstimationRequest):
    """
    Identify the minimum number of peers needed to run a specific model based on available vRAM.
    
    Args:
        request: ModelEstimationRequest containing model_id, hf_token, and qbits
        
    Returns:
        Information about the minimum set of peers needed to run the model
    """
    try:
        # First, estimate the model's VRAM requirements
        estimation_result = await estimate_model(request)
        required_vram = estimation_result["estimated_vram_gb"]
        
        # Get active peers and their metrics
        active_peers = await get_active_peers()
        peers_with_vram = []
        total_available_vram = 0
        
        for peer_id in active_peers:
            try:
                # Get latest metrics for this peer
                metrics_history = await get_peer_metrics(peer_id, time_window=60)
                if not metrics_history:
                    continue
                    
                latest_metrics = metrics_history[0]["metrics"]
                
                # Check if GPU metrics are available
                if "total_free_vram_gb" in latest_metrics:
                    free_vram = latest_metrics["total_free_vram_gb"]
                    total_available_vram += free_vram
                    
                    peers_with_vram.append({
                        "peer_id": peer_id,
                        "free_vram_gb": free_vram,
                        "metrics": latest_metrics
                    })
            except Exception as e:
                print(f"❌ Error processing peer {peer_id}: {e}")
        
        # Sort peers by available VRAM in descending order
        peers_with_vram.sort(key=lambda x: x["free_vram_gb"], reverse=True)
        
        # Select minimum number of peers needed
        selected_peers = []
        remaining_vram_needed = required_vram
        
        for peer in peers_with_vram:
            if remaining_vram_needed <= 0:
                break
            selected_peers.append(peer)
            remaining_vram_needed -= peer["free_vram_gb"]
        
        # Calculate total VRAM from selected peers
        selected_vram = sum(peer["free_vram_gb"] for peer in selected_peers)
        can_host_model = selected_vram >= required_vram
        
        return {
            "status": "success",
            "model_info": {
                "name": request.model_id,
                "required_vram_gb": required_vram,
                "total_params": estimation_result["total_params"],
                "quantization": f"{request.qbits}-bit"
            },
            "network_info": {
                "total_peers": len(active_peers),
                "peers_with_vram": len(peers_with_vram),
                "selected_peers": len(selected_peers),
                "total_available_vram_gb": selected_vram,
                "can_host_model": can_host_model,
                "vram_utilization_percentage": (required_vram / selected_vram * 100) if selected_vram > 0 else 0
            },
            "selected_peers": selected_peers
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to identify suitable peers: {str(e)}"
        )

@app.post("/start_job")
async def start_job():
    """Endpoint to initiate a distributed computation job"""
    global doc, node
    author = await node.authors().create()

    # Define input matrix for computation
    U = torch.tensor([[1., 2.], [3., 4.]])

    try:
        # Send job payload to the first machine in the pipeline
        payload = json.dumps(U.tolist()).encode()
        await doc.set_bytes(author, TRIGGER_KEY.encode(), payload)
        print("🚀 Sent job payload to first machine")
        return {"status": "triggered"}
    except Exception as e:
        print(f"❌ Failed to send job trigger: {e}")
        return {"status": "error", "detail": str(e)}

@app.get("/result")
async def get_final_result():
    """Endpoint to retrieve the final computation result"""
    try:
        # Query all entries and look for the final result
        query = iroh.Query.all(None)
        entries = await doc.get_many(query)

        for entry in reversed(entries):
            if entry.key().decode() == FINAL_RESULT_KEY:
                content = await node.blobs().read_to_bytes(entry.content_hash())
                tensor = torch.tensor(json.loads(content.decode()))
                return {
                    "status": "success",
                    "result": tensor.tolist()
                }

        return {"status": "waiting", "detail": "No final result yet."}
    except Exception as e:
        return {"status": "error", "detail": str(e)}

@app.post("/calculate_peer_layers")
async def calculate_peer_layers(request: ModelEstimationRequest, peer_id: str, available_vram_gb: float):
    """
    Calculate how many layers can fit in a specific peer's available VRAM.
    
    Args:
        request: ModelEstimationRequest containing model_id, hf_token, and qbits
        peer_id: ID of the peer to calculate for
        available_vram_gb: Available VRAM in GB for this peer
        
    Returns:
        Detailed calculation of how many layers can fit in the peer
    """
    try:
        # Download model configuration
        config = await download_config(
            request.model_id,
            request.hf_token,
            request.filename
        )
        
        # Calculate layer capacity for this peer
        layer_calculation = calculate_max_layers_for_peer(
            config, 
            available_vram_gb, 
            request.qbits
        )
        
        return {
            "status": "success",
            "peer_id": peer_id,
            "model_id": request.model_id,
            "quantization": f"{request.qbits}-bit",
            "calculation": layer_calculation
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to calculate layers for peer {peer_id}: {str(e)}"
        )

@app.post("/distribute_model_layers")
async def distribute_model_layers(request: ModelEstimationRequest):
    """
    Create an optimal distribution plan for model layers across all available GPU peers.
    
    Args:
        request: ModelEstimationRequest containing model_id, hf_token, and qbits
        
    Returns:
        Optimal distribution plan showing how to split the model across peers
    """
    try:
        # Download model configuration
        config = await download_config(
            request.model_id,
            request.hf_token,
            request.filename
        )
        
        # Get active peers and their VRAM availability
        active_peers = await get_active_peers()
        peers_vram = {}
        
        for peer_id in active_peers:
            try:
                # Get latest metrics for this peer
                metrics_history = await get_peer_metrics(peer_id, time_window=60)
                if not metrics_history:
                    continue
                    
                latest_metrics = metrics_history[0]["metrics"]
                
                # Check if GPU metrics are available
                if "total_free_vram_gb" in latest_metrics:
                    peers_vram[peer_id] = latest_metrics["total_free_vram_gb"]
                    
            except Exception as e:
                print(f"❌ Error processing peer {peer_id}: {e}")
        
        if not peers_vram:
            raise HTTPException(
                status_code=400,
                detail="No peers with available VRAM found"
            )
        
        # Create distribution plan
        distribution_plan = distribute_layers_across_peers(
            config,
            peers_vram,
            request.qbits
        )
        
        return {
            "status": "success",
            "model_id": request.model_id,
            "quantization": f"{request.qbits}-bit",
            "distribution_plan": distribution_plan
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create distribution plan: {str(e)}"
        )

@app.get("/peer_layer_capacity/{peer_id}")
async def get_peer_layer_capacity(peer_id: str, model_id: str, qbits: int = DEFAULT_QBITS):
    """
    Get the current layer capacity for a specific peer based on its latest metrics.
    
    Args:
        peer_id: ID of the peer to check
        model_id: Model to calculate capacity for
        qbits: Quantization bits (default from settings)
        
    Returns:
        Layer capacity information for the peer
    """
    try:
        # Get peer's latest metrics
        metrics_history = await get_peer_metrics(peer_id, time_window=60)
        if not metrics_history:
            raise HTTPException(
                status_code=404,
                detail=f"No recent metrics found for peer {peer_id}"
            )
        
        latest_metrics = metrics_history[0]["metrics"]
        
        if "total_free_vram_gb" not in latest_metrics:
            raise HTTPException(
                status_code=400,
                detail=f"No VRAM information available for peer {peer_id}"
            )
        
        available_vram = latest_metrics["total_free_vram_gb"]
        
        # Download model configuration (using default HF token)
        config = await download_config(model_id)
        
        # Calculate layer capacity
        layer_calculation = calculate_max_layers_for_peer(
            config, 
            available_vram, 
            qbits
        )
        
        return {
            "status": "success",
            "peer_id": peer_id,
            "model_id": model_id,
            "quantization": f"{qbits}-bit",
            "current_metrics": latest_metrics,
            "layer_capacity": layer_calculation
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get layer capacity for peer {peer_id}: {str(e)}"
        )

def run_sharding_task(task_id: str, request: ModelShardingRequest):
    """Background task function to run model sharding."""
    try:
        # Update task status
        background_tasks[task_id]["status"] = "running"
        background_tasks[task_id]["started_at"] = datetime.now().isoformat()
        
        # Extract model name for directory naming
        model_name_safe = request.model_id.replace("/", "_").replace("\\", "_")
        output_dir = f"./shards/{model_name_safe}"
        
        print(f"🔪 Starting background layer sharding for {request.model_id}")
        print(f"📁 Output directory: {output_dir}")
        
        # Call the sharding function
        result = shard_model_by_layers(
            model_name=request.model_id,
            output_dir=output_dir,
            hf_token=request.hf_token,
            cache_dir=request.cache_dir,
            model_layers_key=request.model_layers_key
        )
        
        # Update task with successful result
        background_tasks[task_id].update({
            "status": "completed",
            "completed_at": datetime.now().isoformat(),
            "result": {
                "model_id": request.model_id,
                "output_directory": result["output_dir"],
                "total_components": result["total_components"],
                "metadata": result["metadata"],
                "message": f"Successfully created {result['total_components']} layer components"
            }
        })
        
        print(f"✅ Background sharding completed for {request.model_id}")
        
    except Exception as e:
        # Update task with error
        background_tasks[task_id].update({
            "status": "failed",
            "completed_at": datetime.now().isoformat(),
            "error": str(e)
        })
        print(f"❌ Background sharding failed for {request.model_id}: {e}")

@app.post("/create_layer_shards")
async def create_layer_shards(request: ModelShardingRequest, background_tasks_runner: BackgroundTasks):
    """
    Start layer sharding for a model in the background.
    
    Args:
        request: ModelShardingRequest containing model_id, hf_token, and optional parameters
        background_tasks_runner: FastAPI background tasks runner
        
    Returns:
        Task ID for tracking the sharding progress
    """
    # Generate unique task ID
    task_id = str(uuid.uuid4())
    
    # Initialize task tracking
    background_tasks[task_id] = {
        "task_id": task_id,
        "model_id": request.model_id,
        "status": "queued",
        "created_at": datetime.now().isoformat(),
        "started_at": None,
        "completed_at": None,
        "result": None,
        "error": None
    }
    
    # Start background task
    background_tasks_runner.add_task(run_sharding_task, task_id, request)
    
    print(f"🚀 Queued layer sharding task {task_id} for {request.model_id}")
    
    return {
        "status": "queued",
        "task_id": task_id,
        "model_id": request.model_id,
        "message": "Layer sharding task started in background. Use /task_status/{task_id} to check progress."
    }

@app.get("/task_status/{task_id}")
async def get_task_status(task_id: str):
    """
    Get the status of a background sharding task.
    
    Args:
        task_id: The task ID returned from create_layer_shards
        
    Returns:
        Current status and details of the task
    """
    if task_id not in background_tasks:
            raise HTTPException(
            status_code=404,
            detail=f"Task {task_id} not found"
            )

    task = background_tasks[task_id]
    
    # Calculate duration if task is running or completed
    duration = None
    if task["started_at"]:
        start_time = datetime.fromisoformat(task["started_at"])
        end_time = datetime.fromisoformat(task["completed_at"]) if task["completed_at"] else datetime.now()
        duration = str(end_time - start_time)
    
    response = {
        "task_id": task_id,
        "model_id": task["model_id"],
        "status": task["status"],
        "created_at": task["created_at"],
        "started_at": task["started_at"],
        "completed_at": task["completed_at"],
        "duration": duration
    }
    
    if task["status"] == "completed" and task["result"]:
        response["result"] = task["result"]
    elif task["status"] == "failed" and task["error"]:
        response["error"] = task["error"]
    
    return response

@app.get("/tasks")
async def list_all_tasks():
    """
    List all background tasks and their current status.
    
    Returns:
        List of all tasks with their status
    """
    tasks_summary = []
    for task_id, task in background_tasks.items():
        tasks_summary.append({
            "task_id": task_id,
            "model_id": task["model_id"],
            "status": task["status"],
            "created_at": task["created_at"],
            "completed_at": task["completed_at"]
        })

        return {
        "total_tasks": len(tasks_summary),
        "tasks": sorted(tasks_summary, key=lambda x: x["created_at"], reverse=True)
    }

@app.api_route("/download_file/{model_name}/{file_path:path}", methods=["GET", "HEAD"])
async def download_model_file(model_name: str, file_path: str, request: Request):
    """
    Serve model files for download by peers.
    
    Args:
        model_name: Name of the model (e.g., TinyLlama_TinyLlama-1.1B-Chat-v1.0)
        file_path: Relative path to file within model directory
        
    Returns:
        File response for download
    """
    try:
        # Construct full file path
        full_path = Path(f"./shards/{model_name}/{file_path}")
        
        # Security check - ensure path is within shards directory
        shards_dir = Path("./shards").resolve()
        if not full_path.resolve().is_relative_to(shards_dir):
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Check if file exists
        if not full_path.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
        
        # Handle HEAD requests (return headers only, no file content)
        if request.method == "HEAD":
            return Response(
                headers={
                    "Accept-Ranges": "bytes",
                    "Cache-Control": "public, max-age=3600",
                    "Content-Length": str(full_path.stat().st_size),
                    "Content-Type": "application/octet-stream"
                }
            )
        
        # Handle GET requests (return actual file)
        return FileResponse(
            path=str(full_path),
            filename=full_path.name,
            media_type="application/octet-stream",
            headers={
                "Accept-Ranges": "bytes",  # Enable range requests for resumable downloads
                "Cache-Control": "public, max-age=3600",  # Cache for 1 hour
                "Content-Length": str(full_path.stat().st_size)  # Explicit content length
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error serving file: {str(e)}")

@app.post("/deploy")
async def deploy_model(request: ModelDeploymentRequest):
    """
    Deploy a pre-sharded model across available peers with optimized distribution.
    
    Args:
        request: ModelDeploymentRequest containing model_name and shard_folder
        
    Returns:
        Deployment status and peer assignments
    """
    global active_deployments
    
    try:
        # Check if deployment is already in progress
        if request.model_name in active_deployments:
            # Check if deployment is stale (older than 5 minutes)
            deployment_age = time.time() - active_deployments[request.model_name]["started_at"]
            if deployment_age > 300:  # 5 minutes
                print(f"🧹 Cleaning up stale deployment (age: {deployment_age:.1f}s)")
                del active_deployments[request.model_name]
            else:
                print(f"⚠️ Deployment already in progress for {request.model_name} (age: {deployment_age:.1f}s)")
                return {
                    "status": "deployment_in_progress",
                    "model_name": request.model_name,
                    "message": f"Deployment for {request.model_name} is already in progress. Please wait."
                }
        
        # Mark deployment as active
        active_deployments[request.model_name] = {
            "started_at": time.time(),
            "status": "in_progress"
        }
        
        print(f"🚀 Starting deployment for {request.model_name}")
        
        # 1. Validate shard folder exists and read metadata
        shard_path = Path(request.shard_folder)
        if not shard_path.exists():
            raise HTTPException(status_code=404, detail=f"Shard folder not found: {request.shard_folder}")
        
        metadata_file = shard_path / "layer_metadata.json"
        if not metadata_file.exists():
            raise HTTPException(status_code=404, detail="layer_metadata.json not found in shard folder")
        
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        
        print(f"📊 Model metadata: {metadata['num_layers']} layers, type: {metadata['model_type']}")
        
        # 2. Get active peers and their VRAM availability
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

        # 3. Create distribution plan using existing logic
        # Create a dummy config for distribution planning
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
            q_bits=16  # Default quantization
        )

        if not distribution_plan["can_fit_model"]:
            raise HTTPException(
                status_code=400,
                detail=f"Model cannot fit in available VRAM. Need {distribution_plan['model_info']['total_model_vram_gb']:.1f}GB, have {distribution_plan['total_available_vram_gb']:.1f}GB"
            )

        # 4. Create optimized deployment instructions for each peer
        deployment_instructions = {}
        peer_list = list(distribution_plan["distribution"].keys())
        
        for i, (peer_id, peer_info) in enumerate(distribution_plan["distribution"].items()):
            is_first_peer = (i == 0)
            is_last_peer = (i == len(peer_list) - 1)
            
            assigned_layers = list(range(
                sum(p["assigned_layers"] for p in list(distribution_plan["distribution"].values())[:i]),
                sum(p["assigned_layers"] for p in list(distribution_plan["distribution"].values())[:i+1])
            ))

            # Determine required files based on peer position
            required_files = []
            
            # Config files (all peers need these)
            required_files.extend([
                "config/config.json",
                "config/tokenizer.json", 
                "config/tokenizer_config.json",
                "config/model.safetensors"  # Dummy model file
            ])
            
            # Essential components based on position
            if is_first_peer:
                required_files.append("embedding/layer.safetensors")
                
            if is_last_peer:
                required_files.extend([
                    "lm_head/layer.safetensors",
                    "norm/layer.safetensors"
                ])
            
            # Assigned layer files
            for layer_idx in assigned_layers:
                required_files.append(f"layers/layer_{layer_idx}.safetensors")
            
            deployment_instructions[peer_id] = {
                "model_name": request.model_name,
                "assigned_layers": assigned_layers,
                "is_first_peer": is_first_peer,
                "is_last_peer": is_last_peer,
                "required_files": required_files,
                "server_download_url": f"http://localhost:8000/download_file/{request.model_name.replace('/', '_')}",
                "vram_allocation": peer_info
            }
        
        # 5. Send deployment instructions to each peer via Iroh
        global doc, node
        author = await node.authors().create()
        
        for peer_id, instructions in deployment_instructions.items():
            try:
                instruction_key = f"deploy_instruction_{peer_id}_{int(time.time())}"
                instruction_data = json.dumps({
                    "action": "deploy_model",
                    "instructions": instructions
                }).encode()
                
                await doc.set_bytes(author, instruction_key.encode(), instruction_data)
                print(f"📤 Sent deployment instructions to {peer_id}")
                
            except Exception as e:
                print(f"❌ Failed to send instructions to {peer_id}: {e}")
        
        print(f"✅ Deployment initiated for {request.model_name}")
        print(f"📊 Distribution: {len(deployment_instructions)} peers")
        
        # Keep deployment as in_progress - peers will update status when done
        # DON'T mark as completed here - instructions were just sent, not completed!
        active_deployments[request.model_name]["instructions_sent_at"] = time.time()

        return {
            "status": "deployment_initiated",
            "model_name": request.model_name,
            "total_peers": len(deployment_instructions),
            "distribution_plan": distribution_plan,
            "deployment_instructions": deployment_instructions,
            "message": f"Deployment instructions sent to {len(deployment_instructions)} peers. Peers will download required files and load model."
        }

    except HTTPException:
        # Clean up deployment tracking on error
        if request.model_name in active_deployments:
            del active_deployments[request.model_name]
        raise
    except Exception as e:
        # Clean up deployment tracking on error
        if request.model_name in active_deployments:
            del active_deployments[request.model_name]
        print(f"❌ Deployment failed: {e}")
        raise HTTPException(status_code=500, detail=f"Deployment failed: {str(e)}")

