import asyncio
import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from colorama import Fore  # ADD
from colorama import init as colorama_init
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, Response
from pydantic import BaseModel

from src.utils.deployment_utils import (
    create_deployment_instructions,
    create_distribution_plan,
    get_peers_with_vram,
    load_model_metadata,
)
from src.utils.sharding_utils import shard_model_by_layers_safetensors

## Tensor_Iroh Starts here ###########################
from src.utils.tensor_protocol_adapter import TensorTransport

######################################################

# Ensure accelerated downloads from Hugging Face hub are enabled for this process
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from src.config.settings import DEFAULT_QBITS, SERVER_HOST
from src.utils.db_utils import (
    PEERS_COLLECTION,
    cleanup_inactive_peers,
    get_active_peers,
    get_peer_metrics,
    setup_collections,
    update_peer_metrics,
)
from src.utils.db_utils import db as _db

# from src.utils.gpu_utils import get_system_metrics, format_metrics_for_db
from src.utils.model_utils import (
    calculate_max_layers_for_peer,
    distribute_layers_across_peers,
    download_config,
    estimate_parameters,
    estimate_vram,
)

# Initialize FastAPI application
app = FastAPI(title="Iroh Tandemn Server")

# SERVER_IP = "172.16.1.249"
SERVER_IP = SERVER_HOST  # ?

# Global Variables for Iroh Node and Gossip Management

# IROH STARTS HERE
active_inferences = {}
central_server_ticket = None
# IROH ENDS HERE

# ? What is the difference between the following 3 peer lists?
peer_table: Dict[str, str] = {}
# server_id = None  # Unique identifier for this server instance
# Peer tracking for topology changes
active_peer_ids = set()  # Currently active peer node_ids
peer_last_seen = {}  # Track when each peer was last seen
PEER_TIMEOUT = 30  # Seconds before considering a peer dead
PEER_CLEANUP_INTERVAL = 5  # Seconds
# Task tracking for background operations
background_tasks = {}  # Dict to track running background tasks

# model_name -> Dict of metadata
active_deployments = {}

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
    cpu: float
    ram: float
    gpu_count: int
    gpu_info: Optional[List[Dict[str, Any]]] = None
    total_free_vram_gb: Optional[float] = None
    timestamp: int


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
    # NEW: optional quantization settings to minimize code changes
    quantization: Optional[str] = None  # e.g. "bitsandbytes", "awq", "gptq"
    qbits: Optional[int] = None  # e.g. 4 or 8 for VRAM plan
    dtype: Optional[str] = None  # e.g. "bfloat16", "float16", "auto"
    # NEW: optional async engine selection & scheduler tuning
    engine_type: Optional[str] = (
        None  # "async" to enable v0 AsyncLLMEngine; default None/"llm"
    )
    use_async_engine: Optional[bool] = None  # alternative toggle; True = async engine
    max_num_seqs: Optional[int] = None
    max_num_batched_tokens: Optional[int] = None


@app.on_event("startup")
async def startup():
    """Initialize the Iroh node and document on server startup"""
    global central_server_ticket, tensor_transport
    print("🚀 Starting Iroh Tandemn server...")

    # Set up MongoDB collections
    await setup_collections()
    print("✅ MongoDB collections configured")

    # Clear all peer records on fresh server start to prevent stale data; comment all this for prod
    delete_result = await _db[PEERS_COLLECTION].delete_many({})
    if delete_result.deleted_count > 0:
        print(
            f"🧹 Cleared {delete_result.deleted_count} stale peer record(s) from previous sessions"
        )
    else:
        print("🧹 No previous peer records found to clear")
    # only need to do this when we are testing ^^^

    # Start the TensorTransport for the central server ##############################################
    tensor_transport = TensorTransport()
    await tensor_transport.start()
    central_server_ticket = tensor_transport.ticket
    print(
        f"🪪 TensorTransport for the central server started – ticket:\n{central_server_ticket}\n"
    )
    #################################################################################################

    # Start the background task for cleaning up inactive peers
    asyncio.create_task(periodic_peer_cleanup())


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
                metrics_history = await get_peer_metrics(
                    peer_id, time_window=60
                )  # Last minute
                if metrics_history:
                    latest_metrics = metrics_history[0][
                        "metrics"
                    ]  # Most recent metrics
                    results.append(
                        {
                            "machine_id": peer_id,
                            "metrics": latest_metrics,
                            "timestamp": metrics_history[0]["timestamp"],
                        }
                    )
            except Exception as inner_e:
                print(f"❌ Failed reading metrics for {peer_id}: {inner_e}")
                results.append({"machine_id": peer_id, "error": str(inner_e)})

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
        current_time = time.time()
        peer_id_of_requester = hb.peer_id
        peer_last_seen[peer_id_of_requester] = current_time

        if peer_id_of_requester not in peer_table:
            # New peer - add to Iroh network
            peer_table[peer_id_of_requester] = peer_ip
            active_peer_ids.add(peer_id_of_requester)
            _ = True
            print(f"🗒️ New peer detected: {peer_id_of_requester}")

        # Compact metrics object similar to previous format
        formatted_metrics = {
            "cpu_percent": hb.cpu,
            "ram_percent": hb.ram,
            "total_free_vram_gb": hb.total_free_vram_gb,
            "gpu_count": hb.gpu_count,
            "gpu_info": hb.gpu_info or [],
            "timestamp": datetime.fromtimestamp(hb.timestamp),
        }
        # uncomment when we need to do things with the database
        # Update MongoDB (time-series) using existing helper
        await update_peer_metrics(hb.peer_id, formatted_metrics)
        # Upsert peer record
        await _db[PEERS_COLLECTION].update_one(
            {"peer_id": peer_id_of_requester},
            {
                "$set": {
                    "peer_id": peer_id_of_requester,
                    "ip": peer_ip,
                    "is_active": True,
                    "last_seen": datetime.utcnow(),
                }
            },
            upsert=True,
        )
        # Colored log
        _ = _get_peer_color(hb.peer_id)
        # print(f"{color}💓 HB from {hb.peer_id[:6]} @ {peer_ip} | CPU {hb.cpu:.1f}% RAM {hb.ram:.1f}% VRAM {hb.total_free_vram_gb:.1f} GB {Style.RESET_ALL}")
        # print(f"🔗 Active peers: {len(active_peer_ids)} ({list(peer_table.keys())})")

        return {
            "status": "ok",
            "central_server_ticket": central_server_ticket,
        }
    except Exception as e:
        print(f"❌ Heartbeat processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def periodic_peer_cleanup():
    """Periodically check for and remove inactive peers."""
    while True:
        await asyncio.sleep(PEER_CLEANUP_INTERVAL)
        current_time = time.time()

        # Find dead peers
        dead_peers = [
            peer_id
            for peer_id, last_seen in peer_last_seen.items()
            if current_time - last_seen > PEER_TIMEOUT
        ]

        if dead_peers:
            print(
                f"🧹 Running periodic cleanup, found {len(dead_peers)} dead peer(s)..."
            )
            for dead_node_id in dead_peers:
                if dead_node_id in peer_table:
                    del peer_table[dead_node_id]
                active_peer_ids.discard(dead_node_id)
                if dead_node_id in peer_last_seen:
                    del peer_last_seen[dead_node_id]

                # Also mark as inactive in the database
                await _db[PEERS_COLLECTION].update_one(
                    {"peer_id": dead_node_id}, {"$set": {"is_active": False}}
                )
                print(
                    f"💀 Peer {dead_node_id[:8]} timed out and removed during periodic cleanup"
                )
            print(
                f"🔗 Active peers after cleanup: {len(active_peer_ids)} ({list(peer_table.keys())})"
            )

            # After cleaning up peers, check for orphaned deployments
            for model_name, deployment_info in list(active_deployments.items()):
                # Check if deployment has a map
                if (
                    "deployment_map" not in deployment_info
                    or not deployment_info["deployment_map"]
                ):
                    continue

                peers_for_deployment = set(deployment_info["deployment_map"].keys())

                # If the set of peers for this deployment is not empty and has no intersection
                # with the currently active peers, then all its peers are gone.
                if peers_for_deployment and peers_for_deployment.isdisjoint(
                    active_peer_ids
                ):
                    print(
                        f"🗑️ All peers for model '{model_name}' have timed out. Clearing deployment."
                    )
                    del active_deployments[model_name]


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
        return {"status": "success", "peer_id": peer_id, "metrics_history": metrics}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


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
        # config = await download_config(
        #     request.model_id,
        #     request.hf_token,
        #     request.filename
        # )
        total_params = estimate_parameters(request.model_id, request.hf_token)
        # Convert qbits to appropriate dtype string for the VRAM calculation
        dtype_map = {4: "int4", 8: "int8", 16: "float16", 32: "float32"}
        dtype = dtype_map.get(request.qbits, "float16")
        parameters_billions = total_params / 1e9
        vram_gb = estimate_vram(parameters_billions, dtype)

        return {
            "status": "success",
            "total_params": total_params,
            "estimated_vram_gb": vram_gb,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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

                    peers_with_vram.append(
                        {
                            "peer_id": peer_id,
                            "free_vram_gb": free_vram,
                            "metrics": latest_metrics,
                        }
                    )
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
                "quantization": f"{request.qbits}-bit",
            },
            "network_info": {
                "total_peers": len(active_peers),
                "peers_with_vram": len(peers_with_vram),
                "selected_peers": len(selected_peers),
                "total_available_vram_gb": selected_vram,
                "can_host_model": can_host_model,
                "vram_utilization_percentage": (required_vram / selected_vram * 100)
                if selected_vram > 0
                else 0,
            },
            "selected_peers": selected_peers,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to identify suitable peers: {str(e)}"
        )


@app.post("/calculate_peer_layers")
async def calculate_peer_layers(
    request: ModelEstimationRequest, peer_id: str, available_vram_gb: float
):
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
            request.model_id, request.hf_token, request.filename
        )

        # Calculate layer capacity for this peer
        layer_calculation = calculate_max_layers_for_peer(
            config, available_vram_gb, request.qbits
        )

        return {
            "status": "success",
            "peer_id": peer_id,
            "model_id": request.model_id,
            "quantization": f"{request.qbits}-bit",
            "calculation": layer_calculation,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to calculate layers for peer {peer_id}: {str(e)}",
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
            request.model_id, request.hf_token, request.filename
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
                status_code=400, detail="No peers with available VRAM found"
            )

        # Create distribution plan
        distribution_plan = distribute_layers_across_peers(
            config, peers_vram, request.qbits
        )

        return {
            "status": "success",
            "model_id": request.model_id,
            "quantization": f"{request.qbits}-bit",
            "distribution_plan": distribution_plan,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to create distribution plan: {str(e)}"
        )


@app.get("/peer_layer_capacity/{peer_id}")
async def get_peer_layer_capacity(
    peer_id: str, model_id: str, qbits: int = DEFAULT_QBITS
):
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
                status_code=404, detail=f"No recent metrics found for peer {peer_id}"
            )

        latest_metrics = metrics_history[0]["metrics"]

        if "total_free_vram_gb" not in latest_metrics:
            raise HTTPException(
                status_code=400,
                detail=f"No VRAM information available for peer {peer_id}",
            )

        available_vram = latest_metrics["total_free_vram_gb"]

        # Download model configuration (using default HF token)
        config = await download_config(model_id)

        # Calculate layer capacity
        layer_calculation = calculate_max_layers_for_peer(config, available_vram, qbits)

        return {
            "status": "success",
            "peer_id": peer_id,
            "model_id": model_id,
            "quantization": f"{qbits}-bit",
            "current_metrics": latest_metrics,
            "layer_capacity": layer_calculation,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get layer capacity for peer {peer_id}: {str(e)}",
        )


async def run_sharding_safetensors_task(
    task_id: str, request: ModelShardingSafetensorsRequest
):
    """Background task function to run safetensors-based model sharding."""
    # Update task status
    background_tasks[task_id]["status"] = "running"
    background_tasks[task_id]["started_at"] = datetime.now().isoformat()

    # Prepare output directory
    model_name_safe = request.model_id.replace("/", "_").replace("\\", "_")
    output_dir = f"./shards/{model_name_safe}"

    print(
        f"🔪 Starting background safetensors-based layer sharding for {request.model_id}"
    )
    print(f"📁 Output directory: {output_dir}")

    try:
        # Offload the blocking sharding call to a thread pool
        result = await asyncio.to_thread(
            shard_model_by_layers_safetensors,
            request.model_id,
            output_dir,
            request.hf_token,
            request.cache_dir,
        )

        # Update task with successful result
        background_tasks[task_id].update(
            {
                "status": "completed",
                "completed_at": datetime.now().isoformat(),
                "result": {
                    "model_id": request.model_id,
                    "output_directory": result["output_dir"],
                    "total_components": result["total_components"],
                    "metadata": result["metadata"],
                    "message": f"Successfully created {result['total_components']} layer components using safetensors processing",
                },
            }
        )

        print(
            f"✅ Background safetensors-based sharding completed for {request.model_id}"
        )
    except Exception as e:
        # Update task with error
        background_tasks[task_id].update(
            {
                "status": "failed",
                "completed_at": datetime.now().isoformat(),
                "error": str(e),
            }
        )
        print(
            f"❌ Background safetensors-based sharding failed for {request.model_id}: {e}"
        )


@app.post("/create_layer_shards_safetensors")
async def create_layer_shards_safetensors(
    request: ModelShardingSafetensorsRequest, background_tasks_runner: BackgroundTasks
):
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
        "method": "safetensors",  # Mark this as safetensors-based
    }

    # Start background task
    background_tasks_runner.add_task(run_sharding_safetensors_task, task_id, request)

    print(
        f"🚀 Queued safetensors-based layer sharding task {task_id} for {request.model_id}"
    )

    return {
        "status": "queued",
        "task_id": task_id,
        "model_id": request.model_id,
        "method": "safetensors",
        "message": "Safetensors-based layer sharding task started in background. Use /task_status/{task_id} to check progress.",
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
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    task = background_tasks[task_id]

    # Calculate duration if task is running or completed
    duration = None
    if task["started_at"]:
        start_time = datetime.fromisoformat(task["started_at"])
        end_time = (
            datetime.fromisoformat(task["completed_at"])
            if task["completed_at"]
            else datetime.now()
        )
        duration = str(end_time - start_time)

    response = {
        "task_id": task_id,
        "model_id": task["model_id"],
        "status": task["status"],
        "created_at": task["created_at"],
        "started_at": task["started_at"],
        "completed_at": task["completed_at"],
        "duration": duration,
    }

    if task["status"] == "completed" and task["result"]:
        response["result"] = task["result"]
    elif task["status"] == "failed" and task["error"]:
        response["error"] = task["error"]

    return response


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
                    "Content-Type": "application/octet-stream",
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
                "Content-Length": str(
                    full_path.stat().st_size
                ),  # Explicit content length
            },
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
    global active_deployments, tensor_transport

    try:
        # Check if deployment is already in progress
        if request.model_name in active_deployments:
            # Check if deployment is stale (older than 5 minutes)
            deployment_age = (
                time.time() - active_deployments[request.model_name]["started_at"]
            )

            # implement the logic for deployment timeout #? Why is the deployment timeout triggered by /deploy
            if deployment_age > 300:  # 5 minutes
                print(f"🧹 Cleaning up stale deployment (age: {deployment_age:.1f}s)")
                del active_deployments[request.model_name]
            else:
                print(
                    f"⚠️ Deployment already in progress for {request.model_name} (age: {deployment_age:.1f}s)"
                )
                return {
                    "status": "deployment_in_progress",
                    "model_name": request.model_name,
                    "message": f"Deployment for {request.model_name} is already in progress. Please wait.",
                }

        # Mark deployment as active
        active_deployments[request.model_name] = {
            "started_at": time.time(),
            "status": "in_progress",
        }

        print(f"🚀 Starting deployment for {request.model_name}")

        # 1. Validate shard folder exists and read metadata
        metadata = load_model_metadata(request.shard_folder)

        # 2. Get active peers and their VRAM availability
        peers_vram = await get_peers_with_vram()

        # 3. Create distribution plan (respect optional qbits)
        qbits_for_plan = request.qbits if request.qbits is not None else DEFAULT_QBITS
        distribution_plan = create_distribution_plan(
            metadata, peers_vram, q_bits=qbits_for_plan
        )

        # 4. Create optimized deployment instructions for each peer
        # ? dicts are ordered, but do we know what's the order being returned? Impt since this ordering is used for the pipeline later in /infer
        deployment_instructions = create_deployment_instructions(
            request, distribution_plan, peer_table, SERVER_IP
        )

        # Propagate async engine preferences (if provided) into each peer's instructions
        if isinstance(request.engine_type, str) or isinstance(
            request.use_async_engine, bool
        ):
            for peer_id, instr in deployment_instructions.items():
                if request.engine_type is not None:
                    instr["engine_type"] = request.engine_type
                if request.use_async_engine is not None:
                    instr["use_async_engine"] = bool(request.use_async_engine)
                if request.max_num_seqs is not None:
                    instr["max_num_seqs"] = int(request.max_num_seqs)
                if request.max_num_batched_tokens is not None:
                    instr["max_num_batched_tokens"] = int(
                        request.max_num_batched_tokens
                    )

        # 5. Persist deployment tracking information BEFORE broadcasting
        active_deployments[request.model_name].update(
            {
                "instructions_sent_at": time.time(),
                "deployment_map": {
                    peerid: instr["assigned_layers"]
                    for peerid, instr in deployment_instructions.items()
                },
                "completion_status": {
                    peerid: "pending" for peerid in deployment_instructions.keys()
                },
            }
        )

        # 6. Send deployment instructions to each peer via Iroh
        send_tasks = []
        peer_ids = []
        for peer_id, instructions in deployment_instructions.items():
            payload_dict = {
                "action": "deploy_model",
                "target_peer_id": peer_id,  # fix this if needed
                "instructions": instructions,
            }
            payload_bytes = json.dumps(payload_dict).encode()
            payload = np.frombuffer(payload_bytes, dtype=np.uint8)
            peer_ids.append(peer_id)
            send_tasks.append(
                asyncio.create_task(tensor_transport.send(peer_id, "deploy", payload))
            )

        results = await asyncio.gather(*send_tasks, return_exceptions=True)
        for peer_id, result in zip(peer_ids, results):
            if isinstance(result, Exception):
                print(f"❌ Failed to send instructions to {peer_id}: {result}")
            else:
                print(f"📤 Sent deployment instructions to {peer_id}")
        # IROH ENDS HERE
        print(f"✅ Deployment initiated for {request.model_name}")
        print(f"📊 Distribution: {len(deployment_instructions)} peers")

        # Keep deployment as in_progress - peers will update status when done
        # DON'T mark as completed here - instructions were just sent, not completed!
        active_deployments[request.model_name].update(
            {
                "instructions_sent_at": time.time(),
                "deployment_map": {
                    peer_id: info["assigned_layers"]
                    for peer_id, info in deployment_instructions.items()
                },  # ? why is this item updated again
                "completion_status": {
                    peer_id: "pending" for peer_id in deployment_instructions.keys()
                },
            }
        )

        return {
            "status": "deployment_initiated",
            "model_name": request.model_name,
            "total_peers": len(deployment_instructions),
            "distribution_plan": distribution_plan,
            "deployment_instructions": deployment_instructions,
            "message": f"Deployment instructions sent to {len(deployment_instructions)} peers. Peers will download required files and load model.",
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
        raise HTTPException(
            status_code=404, detail=f"No deployment found for model {model_name}"
        )

    deployment = active_deployments[model_name]
    return {
        "status": "success",
        "model_name": model_name,
        "deployment_status": deployment.get("status", "unknown"),
        "layer_distribution": deployment.get("deployment_map", {}),
        "peer_completion_status": deployment.get("completion_status", {}),
        "started_at": deployment.get("started_at"),
        "instructions_sent_at": deployment.get("instructions_sent_at"),
    }


class DeploymentCompleteData(BaseModel):
    model_name: str
    peer_id: str
    success: bool
    max_req_in_batch: int


@app.post("/deployment_complete")
async def deployment_complete(data: DeploymentCompleteData):
    global active_deployments
    """
    Receive deployment‐done reports from peers.
    """
    print(
        f"🔍 [DEBUG] Deployment complete received for model {data.model_name} from peer {data.peer_id}"
    )
    if data.model_name not in active_deployments:
        raise HTTPException(404, f"No deployment found for model {data.model_name}")

    status_map = active_deployments[data.model_name]["completion_status"]
    max_req_map = active_deployments[data.model_name]["max_req_in_batch"]
    if data.peer_id not in status_map:
        raise HTTPException(400, f"Peer {data.peer_id} not in deployment")

    status_map[data.peer_id] = "success" if data.success else "failed"
    max_req_map[data.peer_id] = data.max_req_in_batch

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
        inference_state["processing_time"] = (
            inference_state["completed_at"] - inference_state["started_at"]
        )
        print(f"Result: {completion.output_text}")
        print(
            f"✅ Inference {request_id} completed in {inference_state['processing_time']:.2f}s"
        )
    return {"status": "ok"}


# new inference request model
class InferenceRequest(BaseModel):
    model_name: str
    input_text: str
    max_tokens: int = 100


@app.get("/", response_class=HTMLResponse)
async def hello_world():
    return "<html><body><h1>Hello, World!</h1><p>This is a bare debug page.</p></body></html>"


# IROH STARTS HERE
@app.post("/infer")
async def infer(request: InferenceRequest):
    "Start Distributed Inference for a given model, across the peer network"
    global trigger_gossip_sink, active_deployments, active_inferences

    # 0. Check if model deployed or not
    if request.model_name not in active_deployments:
        raise HTTPException(
            status_code=404,
            detail=f"No deployment found for model {request.model_name}",
        )

    # check if deployment is ready
    if active_deployments[request.model_name]["status"] != "ready":
        raise HTTPException(
            status_code=404,
            detail=f"Deployment for model {request.model_name} is not ready",
        )

    # 1. Get the deployment instructions
    deployment_map = active_deployments[request.model_name]["deployment_map"]
    if not deployment_map:
        raise HTTPException(
            status_code=404,
            detail=f"No deployment map found for model {request.model_name}",
        )

    # 2. Send the trigger to the first peer #? comment doesn't match action
    request_id = f"req_{int(time.time() * 1000)}_{len(active_inferences)}"  # ? Is this sustainable as a request_id
    active_inferences[request_id] = {
        "status": "processing",
        "model_name": request.model_name,
        "request_id": request_id,
        "started_at": time.time(),
        "result": None,
    }

    # 3. Construct the pipeline from the deployment map
    # The map is {peer_id: [layer_indices]}, we need an ordered list of peer_ids
    pipeline = list(deployment_map.keys())  # ? noting again the order

    # 4. Prepare input text and sampling parameters
    sampling_params = {"max_tokens": request.max_tokens}
    # 4. Create the instruction payload
    inference_payload = {
        "action": "start_inference",  # ? Is this action spurious since it's sent to the infer endpoint
        "request_id": request_id,
        "model_name": request.model_name,
        "input_text": request.input_text,
        "pipeline": pipeline,
        "sampling_params": sampling_params,
        "assigned_layers": deployment_map,
        "timestamp": time.time(),
    }
    instruction_payload = json.dumps(inference_payload).encode()
    instruction_payload = np.frombuffer(instruction_payload, dtype=np.uint8)

    # 5. Broadcast the instruction payload to ALL peers in the pipeline concurrently
    trigger_tasks = [
        asyncio.create_task(
            tensor_transport.send(peer_ticket, "inference", instruction_payload)
        )
        for peer_ticket in pipeline
    ]
    trigger_results = await asyncio.gather(*trigger_tasks, return_exceptions=True)
    for peer_ticket, res in zip(pipeline, trigger_results):
        if isinstance(res, Exception):
            print(
                f"❌ Failed to send inference trigger to peer: {peer_ticket[:8]}... ({res})"
            )
        else:
            print(f"📤 Sent inference trigger to peer: {peer_ticket[:8]}...")

    print(f"🚀 Inference {request_id} started for model {request.model_name}")
    return InferenceResponse(
        request_id=request_id, status="processing", result=None, processing_time=None
    )


# IROH ENDS HERE
