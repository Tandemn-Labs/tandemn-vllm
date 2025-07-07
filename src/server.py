import asyncio
import iroh
import torch
import json
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import FileResponse, Response
from iroh.iroh_ffi import uniffi_set_event_loop
from iroh import PublicKey, NodeAddr
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uuid
import time
from datetime import datetime
from pathlib import Path
from colorama import Fore, Style, init as colorama_init  # ADD
from src.utils.sharding_utils import shard_model_by_layers_safetensors
from transformers import AutoTokenizer
import os

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
# from src.utils.gpu_utils import get_system_metrics, format_metrics_for_db
from src.utils.model_utils import (
    download_config,
    estimate_parameters,
    estimate_vram,
    calculate_max_layers_for_peer,
    distribute_layers_across_peers
)


# Initialize FastAPI application
app = FastAPI(title="Iroh Tandemn Server")

SERVER_IP = "172.16.1.249"
# Global Variables for Iroh Node and Gossip Management
# IROH STARTS HERE
node = None  # Iroh node instance
trigger_gossip_sink = None
deployment_gossip_sink = None
active_inferences = {}
TRIGGER_TOPIC = bytes("trigger_topic".ljust(32), "utf-8")[:32]
DEPLOYMENT_TOPIC = bytes("deployment_topic".ljust(32), "utf-8")[:32]
# IROH ENDS HERE

peer_table: Dict[str,NodeAddr] = {}
server_id = None  # Unique identifier for this server instance
# Peer tracking for topology changes
active_peer_ids = set()  # Currently active peer node_ids
peer_last_seen = {}  # Track when each peer was last seen
PEER_TIMEOUT = 30  # Seconds before considering a peer dead


# doc = None   # Iroh document instance
# ticket = None  # Document sharing ticket

# Task tracking for background operations
background_tasks = {}  # Dict to track running background tasks

# Deployment tracking to prevent infinite loops
active_deployments = {}  # Track ongoing deployments by model_name

# Constants for document keys
TRIGGER_KEY = "job_trigger"  # Key for job initiation
FINAL_RESULT_KEY = "final_result"  # Key for final computation result

colorama_init(autoreset=True)
COLORS = [Fore.CYAN, Fore.MAGENTA, Fore.YELLOW, Fore.GREEN, Fore.BLUE]
peer_color_map = {}

def _get_peer_color(peer_id: str):
    if peer_id not in peer_color_map:
        peer_color_map[peer_id] = COLORS[len(peer_color_map) % len(COLORS)]
    return peer_color_map[peer_id]

class HeartbeatRequest(BaseModel):
    """Schema for heartbeat POSTs from peers."""
    peer_id: str
    node_id: str #iroh_node_id
    addresses: List[str] #addresses
    relay_url: Optional[str] = None #relay_url
    cpu: float
    ram: float
    gpu_count: int
    gpu_info: Optional[List[Dict[str, Any]]] = None
    total_free_vram_gb: Optional[float] = None
    timestamp: int

class NoopCallback(iroh.GossipMessageCallback):
    async def on_message(self, msg):
        return

# still has to change
class InferenceResponse(BaseModel):
    request_id: str
    status: str  # "processing", "completed", "failed"
    result: Optional[str] = None
    processing_time: Optional[float] = None

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

class ModelShardingSafetensorsRequest(BaseModel):
    model_id: str
    hf_token: str
    cache_dir: Optional[str] = None

class ModelDeploymentRequest(BaseModel):
    model_name: str
    shard_folder: str

# IROH STARTS HERE
async def refresh_trigger_sink():
    """(Re)create the trigger gossip sink with current peer list."""
    global trigger_gossip_sink
    peer_ids = list(peer_table.keys())  # list of node_id strings
    # Recreate sink each time to update peer set
    trigger_gossip_sink = await node.gossip().subscribe(TRIGGER_TOPIC, peer_ids, NoopCallback())
    print(f"📡 Trigger sink refreshed with peers: {peer_ids}")

async def refresh_deployment_sink():
    """(Re)create the deployment gossip sink with current peer list."""
    global deployment_gossip_sink
    peer_ids = list(peer_table.keys())  # list of node_id strings
    # Recreate sink each time to update peer set
    deployment_gossip_sink = await node.gossip().subscribe(
        DEPLOYMENT_TOPIC,
        peer_ids,
        NoopCallback())
    print(f"📡 Deployment sink refreshed with peers: {peer_ids}")
# IROH ENDS HERE

@app.on_event("startup")
async def startup():
    """Initialize the Iroh node and document on server startup"""
    # global node, doc, ticket, server_peer_id
    global node, server_id, trigger_gossip_sink, active_inferences, TRIGGER_TOPIC, peer_table

    print("🚀 Starting Iroh Tandemn server...")
    # Configure async event loop for Iroh
    uniffi_set_event_loop(asyncio.get_running_loop())

    # Set up MongoDB collections
    await setup_collections()
    print("✅ MongoDB collections configured")

    # Clear all peer records on fresh server start to prevent stale data; comment all this for prod
    delete_result = await _db[PEERS_COLLECTION].delete_many({})
    if delete_result.deleted_count > 0:
        print(f"🧹 Cleared {delete_result.deleted_count} stale peer record(s) from previous sessions")
    else:
        print("🧹 No previous peer records found to clear")
    # only need to do this when we are testing ^^^

    # IROH STARTS HERE
    options = iroh.NodeOptions()
    options.enable_gossip = True
    node = await iroh.Iroh.memory_with_options(options)
    server_id = await node.net().node_id()
    print(f"✅ Iroh node started with server_id: {server_id}")

    await refresh_trigger_sink()    
    await refresh_deployment_sink()
    # IROH ENDS HERE
    
    # print("📡 Using HTTP for completions instead of gossip")    # options.enable_docs = True

    # # Create and share document
    # doc = await node.docs().create()
    # ticket = await doc.share(iroh.ShareMode.WRITE, iroh.AddrInfoOptions.RELAY_AND_ADDRESSES)

    # Record initial system metrics
    # author = await node.authors().create()
    # key = server_peer_id.encode()
    # metrics = get_system_metrics()
    # formatted_metrics = format_metrics_for_db(metrics)
    # value = f"CPU: {metrics.cpu_percent}%\nRAM: {metrics.ram_percent}%".encode()

    # try:
    #     # Store system metrics in the document and MongoDB
    #     await doc.set_bytes(author, key, value)
    #     await update_peer_metrics(server_peer_id, formatted_metrics)
    #     print(f"✅ Server metrics stored with key {server_peer_id}")
    # except Exception as e:
    #     print(f"❌ Failed to send server metrics: {e}")

    # Join the document to enable updates
    # doc = await node.docs().join(ticket)
    # print("✅ Iroh node started")
    # print("📎 SHARE THIS TICKET WITH ALL INTERNAL MACHINES:\n")
    # print(str(ticket) + "\n")
    

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

@app.post("/heartbeat")
async def heartbeat_endpoint(hb: HeartbeatRequest, request: Request):
    """Receive heartbeat from a peer and store metrics in MongoDB."""
    try:
        peer_ip = request.client.host if request.client else "unknown"
        
        # Track when this peer was last seen
        current_time = time.time()
        peer_last_seen[hb.node_id] = current_time
        
        # Check if this is a new peer or if peer table needs updating
        topology_changed = False
        
        if hb.node_id not in peer_table:
            # New peer - add to Iroh network
            pk = PublicKey.from_string(hb.node_id)
            addr = NodeAddr(pk, hb.relay_url, hb.addresses)
            await node.net().add_node_addr(addr)
            peer_table[hb.node_id] = addr
            active_peer_ids.add(hb.node_id)
            topology_changed = True
            print(f"🗒️ New peer detected: {hb.peer_id}")
        
        # Check for dead peers (haven't sent heartbeat in PEER_TIMEOUT seconds)
        dead_peers = []
        for node_id, last_seen in list(peer_last_seen.items()):
            if current_time - last_seen > PEER_TIMEOUT:
                dead_peers.append(node_id)
        
        # Remove dead peers from tracking
        for dead_node_id in dead_peers:
            if dead_node_id in peer_table:
                del peer_table[dead_node_id]
                active_peer_ids.discard(dead_node_id)
                del peer_last_seen[dead_node_id]
                topology_changed = True
                print(f"💀 Peer {dead_node_id[:8]} timed out and removed")
        
        # Only refresh sinks if topology actually changed
        if topology_changed:
            await refresh_trigger_sink()
            await refresh_deployment_sink()
            print(f"🔄 Network topology changed - sinks refreshed")
        
        print(f"🗒️ Active peers: {len(active_peer_ids)} ({list(p[:8] for p in active_peer_ids)})")
        
        # Get server address once
        srv_addr = await node.net().node_addr()
        
        # Build peer list (excluding requester)
        peers_payload = []
        for pid, naddr in peer_table.items():
            if pid == hb.node_id:
                continue
            peers_payload.append({
                "node_id": pid,
                "addresses": naddr.direct_addresses(),
                "relay_url": naddr.relay_url()
            })
        
        # Compact metrics object similar to previous format
        formatted_metrics = {
            "cpu_percent": hb.cpu,
            "ram_percent": hb.ram,
            "total_free_vram_gb": hb.total_free_vram_gb,
            "gpu_count": hb.gpu_count,
            "gpu_info": hb.gpu_info or [],
            "timestamp": datetime.fromtimestamp(hb.timestamp)
        }

        # Update MongoDB (time-series) using existing helper
        await update_peer_metrics(hb.peer_id, formatted_metrics)
        # Upsert peer record
        await _db[PEERS_COLLECTION].update_one(
            {"peer_id": hb.peer_id},
            {"$set": {"peer_id": hb.peer_id,
                      "node_id": hb.node_id, 
                      "ip": peer_ip,
                       "is_active": True,
                        "last_seen": datetime.utcnow()}},
            upsert=True
        )
        # Colored log
        color = _get_peer_color(hb.peer_id)
        print(f"{color}💓 HB from {hb.peer_id[:6]} @ {peer_ip} | CPU {hb.cpu:.1f}% RAM {hb.ram:.1f}% VRAM {hb.total_free_vram_gb:.1f} GB {Style.RESET_ALL}")
        
        return {
            "status": "ok",
            "server_id": server_id,
            "server_addresses": srv_addr.direct_addresses(),
            "relay_url": srv_addr.relay_url(),
            "peers": peers_payload
        }
    except Exception as e:
        print(f"❌ Heartbeat processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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

# @app.get("/ticket")
# async def get_ticket():
#     """Endpoint to retrieve the document sharing ticket"""
#     return {"ticket": str(ticket)}

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

async def run_sharding_safetensors_task(task_id: str, request: ModelShardingSafetensorsRequest):
    """Background task function to run safetensors-based model sharding."""
    # Update task status
    background_tasks[task_id]["status"] = "running"
    background_tasks[task_id]["started_at"] = datetime.now().isoformat()

    # Prepare output directory
    model_name_safe = request.model_id.replace("/", "_").replace("\\", "_")
    output_dir = f"./shards/{model_name_safe}"

    print(f"🔪 Starting background safetensors-based layer sharding for {request.model_id}")
    print(f"📁 Output directory: {output_dir}")

    try:
        # Offload the blocking sharding call to a thread pool
        result = await asyncio.to_thread(
            shard_model_by_layers_safetensors,
            request.model_id,
            output_dir,
            request.hf_token,
            request.cache_dir
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
                "message": f"Successfully created {result['total_components']} layer components using safetensors processing"
            }
        })

        print(f"✅ Background safetensors-based sharding completed for {request.model_id}")
    except Exception as e:
        # Update task with error
        background_tasks[task_id].update({
            "status": "failed",
            "completed_at": datetime.now().isoformat(),
            "error": str(e)
        })
        print(f"❌ Background safetensors-based sharding failed for {request.model_id}: {e}")



@app.post("/create_layer_shards_safetensors")
async def create_layer_shards_safetensors(request: ModelShardingSafetensorsRequest, background_tasks_runner: BackgroundTasks):
    """
    Start safetensors-based layer sharding for a model in the background.
    This approach processes safetensors files directly without loading the entire model into memory.
    
    Args:
        request: ModelShardingSafetensorsRequest containing model_id, hf_token, and optional parameters
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
        "error": None,
        "method": "safetensors"  # Mark this as safetensors-based
    }
    
    # Start background task
    background_tasks_runner.add_task(run_sharding_safetensors_task, task_id, request)
    
    print(f"🚀 Queued safetensors-based layer sharding task {task_id} for {request.model_id}")
    
    return {
        "status": "queued",
        "task_id": task_id,
        "model_id": request.model_id,
        "method": "safetensors",
        "message": "Safetensors-based layer sharding task started in background. Use /task_status/{task_id} to check progress."
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
    global active_deployments, deployment_gossip_sink
    
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
            q_bits=16  # Default quantization, for testing for now
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
                # "config/model.safetensors"  # Dummy model file
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
                "server_download_url": f"http://{SERVER_IP}:8000/download_file/{request.model_name.replace('/', '_')}", #TODO: change to server ip
                "vram_allocation": peer_info
            }
        
        # 5. Send deployment instructions to each peer via Iroh
        # IROH STARTS HERE
        for peer_id, instructions in deployment_instructions.items():
            try:
                payload = json.dumps({
                    "action": "deploy_model",
                    "target_peer_id": peer_id, # fix this if needed
                    "instructions": instructions
                }).encode()
                await deployment_gossip_sink.broadcast(payload)
                print(f"📤 Sent deployment instructions to {peer_id}")
            except Exception as e:
                print(f"❌ Failed to send instructions to {peer_id}: {e}")
        # IROH ENDS HERE
        print(f"✅ Deployment initiated for {request.model_name}")
        print(f"📊 Distribution: {len(deployment_instructions)} peers")
        
        # Keep deployment as in_progress - peers will update status when done
        # DON'T mark as completed here - instructions were just sent, not completed!
        active_deployments[request.model_name].update({
            "instructions_sent_at": time.time(),
            "deployment_map": {peer_id: info["assigned_layers"] for peer_id, info in deployment_instructions.items()},
            "completion_status": {peer_id: "pending" for peer_id in deployment_instructions.keys()}
        })

        return {
            "status": "deployment_initiated",
            "model_name": request.model_name,
            "total_peers": len(deployment_instructions),
            "distribution_plan": distribution_plan,
            "deployment_instructions": deployment_instructions,
            "message": f"Deployment instructions sent to {len(deployment_instructions)} peers. Peers will download required files and load model."
        }
        # global doc, node
        # author = await node.authors().create()
        
        # for peer_id, instructions in deployment_instructions.items():
        #     try:
        #         instruction_key = f"deploy_instruction_{peer_id}_{int(time.time())}"
        #         instruction_data = json.dumps({
        #             "action": "deploy_model",
        #             "instructions": instructions
        #         }).encode()
                
        #         await doc.set_bytes(author, instruction_key.encode(), instruction_data)
        #         print(f"📤 Sent deployment instructions to {peer_id}")
                
        #     except Exception as e:
        #         print(f"❌ Failed to send instructions to {peer_id}: {e}")
        
        # print(f"✅ Deployment initiated for {request.model_name}")
        # print(f"📊 Distribution: {len(deployment_instructions)} peers")
        
        # # Keep deployment as in_progress - peers will update status when done
        # # DON'T mark as completed here - instructions were just sent, not completed!
        # active_deployments[request.model_name].update({
        #     "instructions_sent_at": time.time(),
        #     "deployment_map": {peer_id: info["assigned_layers"] for peer_id, info in deployment_instructions.items()},
        #     "completion_status": {peer_id: "pending" for peer_id in deployment_instructions.keys()}
        # })

        # return {
        #     "status": "deployment_initiated",
        #     "model_name": request.model_name,
        #     "total_peers": len(deployment_instructions),
        #     "distribution_plan": distribution_plan,
        #     "deployment_instructions": deployment_instructions,
        #     "message": f"Deployment instructions sent to {len(deployment_instructions)} peers. Peers will download required files and load model."
        # }

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

@app.get("/deployment_status/{model_name}")
async def get_deployment_status(model_name: str):
    """
    Get the current deployment status showing which layers are deployed where.
    
    Args:
        model_name: Name of the deployed model
        
    Returns:
        Current deployment status and layer distribution
    """
    if model_name not in active_deployments:
        raise HTTPException(status_code=404, detail=f"No deployment found for model {model_name}")
    
    deployment = active_deployments[model_name]
    return {
        "status": "success",
        "model_name": model_name,
        "deployment_status": deployment.get("status", "unknown"),
        "layer_distribution": deployment.get("deployment_map", {}),
        "peer_completion_status": deployment.get("completion_status", {}),
        "started_at": deployment.get("started_at"),
        "instructions_sent_at": deployment.get("instructions_sent_at")
    }

class DeploymentCompleteData(BaseModel):
    model_name: str
    peer_id: str
    success: bool

@app.post("/deployment_complete")
async def deployment_complete(data: DeploymentCompleteData):
    global active_deployments
    """
    Receive deployment‐done reports from peers.
    """
    print(f"🔍 [DEBUG] Deployment complete received for model {data.model_name} from peer {data.peer_id}")
    if data.model_name not in active_deployments:
        raise HTTPException(404, f"No deployment found for model {data.model_name}")

    status_map = active_deployments[data.model_name]["completion_status"]
    if data.peer_id not in status_map:
        raise HTTPException(400, f"Peer {data.peer_id} not in deployment")

    status_map[data.peer_id] = "success" if data.success else "failed"
    # If every peer succeeded, mark the whole deployment ready; if any failed, fail it.
    if all(s == "success" for s in status_map.values()):
        active_deployments[data.model_name]["status"] = "ready"
        print(f"✅ Deployment for {data.model_name} is now READY")
    elif any(s == "failed" for s in status_map.values()):
        active_deployments[data.model_name]["status"] = "failed"
        print(f"❌ Deployment for {data.model_name} has FAILED")

    return {"status": "ok"}

class CompletionData(BaseModel):
    request_id: str
    output_text: str
    peer_id: str
    timestamp: int

@app.post("/completion")
async def completion(completion: CompletionData):
    """
    Receive completion data from a peer.
    """
    request_id = completion.request_id
    if request_id in active_inferences:
        inference_state = active_inferences[request_id]
        inference_state["status"] = "completed"
        inference_state["result"] = completion.output_text
        inference_state["completed_at"] = time.time()
        inference_state["processing_time"] = inference_state["completed_at"] - inference_state["started_at"]
        print(f"Result: {completion.output_text}")
        print(f"✅ Inference {request_id} completed in {inference_state['processing_time']:.2f}s")
    return {"status": "ok"}

# new inference request model
class InferenceRequest(BaseModel):
    model_name: str
    input_text: str
    max_tokens: int = 100

#IROH STARTS HERE
@app.post("/infer")
async def infer(request: InferenceRequest):
    "Start Distributed Inference for a given model, across the peer network"
    global trigger_gossip_sink, active_deployments, active_inferences

    #0. Check if model deployed or not
    if request.model_name not in active_deployments:
        raise HTTPException(status_code=404, detail=f"No deployment found for model {request.model_name}")
    
    # check if deployment is ready
    if active_deployments[request.model_name]["status"] != "ready":
        raise HTTPException(status_code=404, detail=f"Deployment for model {request.model_name} is not ready")
    
    #1. Get the deployment instructions
    deployment_map = active_deployments[request.model_name]["deployment_map"]
    if not deployment_map:
        raise HTTPException(status_code=404, detail=f"No deployment map found for model {request.model_name}")

    #2. Send the trigger to the first peer
    request_id = f"req_{int(time.time() * 1000)}_{len(active_inferences)}"
    active_inferences[request_id] = {
        "status": "processing",
        "model_name": request.model_name,
        "request_id": request_id,
        "started_at": time.time(),
        "result": None
    }
    
    #3. Construct the pipeline from the deployment map
    # The map is {peer_id: [layer_indices]}, we need an ordered list of peer_ids
    pipeline = list(deployment_map.keys())

    #4. Prepare input text and sampling parameters
    sampling_params = {"max_tokens": request.max_tokens}
    #4. Create the instruction payload
    inference_payload = {
        "action": "start_inference",
        "request_id": request_id,
        "model_name": request.model_name,
        "input_text": request.input_text,
        "pipeline": pipeline,
        "sampling_params": sampling_params,
        "assigned_layers": deployment_map,
        "timestamp": time.time()
    }
    instruction_payload = json.dumps(inference_payload).encode()
    

    #5. Broadcast the instruction payload to the first peer
    await refresh_trigger_sink() 
    await trigger_gossip_sink.broadcast(instruction_payload)

    print(f"🚀 Inference {request_id} started for model {request.model_name}")
    return InferenceResponse(
        request_id=request_id,
        status="processing",
        result=None,
        processing_time=None
    )
#IROH ENDS HERE




