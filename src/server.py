import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import boto3
import numpy as np
from botocore.exceptions import ClientError
from colorama import Fore  # ADD
from colorama import init as colorama_init
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel

import src.utils.req_batcher as req_batcher
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

from smart_open import open

from src.config.settings import DEFAULT_QBITS, SERVER_HOST
from src.utils.db_utils import (
    PEERS_COLLECTION,
    cleanup_inactive_peers,
    clear_batch_processing_state_for_file,
    get_active_peers,
    get_csv_processing_state_by_file_id,
    get_peer_metrics,
    save_csv_processing_state_by_file_id,
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
SERVER_IP = SERVER_HOST

active_inferences = {}  # Batch ID -> Dict of metadata
active_requests = {}  # Request ID -> Dict of metadata

central_server_ticket = None

logger = logging.getLogger(__name__)

# Initialize S3 client for batch results
s3_client = boto3.client("s3")


# ? What is the difference between the following 3 peer lists?
peer_table: Dict[str, str] = {}
# server_id = None  # Unique identifier for this server instance
# Peer tracking for topology changes
active_peer_ids = set()  # Currently active peer node_ids
peer_last_seen = {}  # Track when each peer was last seen
PEER_TIMEOUT = 30  # Seconds before considering a peer dead
PEER_CLEANUP_INTERVAL = 5  # Seconds
BATCHED_BUFFER_TIMEOUT = 10  # Seconds
latest_effective_buffer_size = 0
# Task tracking for background operations
background_tasks = {}  # Dict to track running background tasks
batch_processing_tasks = {}
_process_batch_continuously = None
# model_name -> Dict of metadata
active_deployments = {}


# Constants for document keys
TRIGGER_KEY = "job_trigger"  # Key for job initiation
FINAL_RESULT_KEY = "final_result"  # Key for final computation result

colorama_init(autoreset=True)
COLORS = [Fore.CYAN, Fore.MAGENTA, Fore.YELLOW, Fore.GREEN, Fore.BLUE]
peer_color_map = {}

# Maximum time we want to wait to fill up batches (in seconds)
MAX_TIME_PER_BATCH = 1


def setup_logging():
    logging.basicConfig(filename="server.log", level=logging.INFO)


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
    current_buffer_size: Optional[int] = None


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
    use_async_engine: Optional[bool] = None  # True = AsyncLLMEngine, False/None = LLM
    max_num_seqs: Optional[int] = None
    max_num_batched_tokens: Optional[int] = None
    engine_args: Optional[Dict[str, Any]] = None


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

    setup_logging()
    logger.info(f"ticket: {central_server_ticket}")

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
            "current_buffer_size": hb.current_buffer_size,
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
        try:
            print("✅ Current Buffer Size: ", hb.current_buffer_size)
        except Exception:
            print("❌ Error printing batch size")
            pass

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
            request.model_id,
            request.hf_token,
            request.filename,  # ? no filename
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
            "max_req_in_batch": {},
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
        # 3.5 Add engine args to distribution plan
        distribution_plan["engine_args"] = request.engine_args
        # 4. Create optimized deployment instructions for each peer
        deployment_instructions = create_deployment_instructions(
            request, distribution_plan, peer_table, SERVER_IP
        )
        # Propagate async engine preferences (if provided) into each peer's instructions
        if request.use_async_engine is not None:
            for peer_id, instr in deployment_instructions.items():
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
                    pid: instr["assigned_layers"]
                    for pid, instr in deployment_instructions.items()
                },
                "completion_status": {
                    pid: "pending" for pid in deployment_instructions.keys()
                },
            }
        )
        # 6. Send deployment instructions to each peer via Iroh
        send_tasks = []
        peer_ids = []
        for peer_id, instructions in deployment_instructions.items():
            print(f"🔍 Engine args: {instructions['engine_args']}")
            payload_dict = {
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
                },
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
    # this is where i am storing what the max can be digested by vllm workers at one time in the pipeline
    max_req_map = active_deployments[data.model_name]["max_req_in_batch"]
    if data.peer_id not in status_map:
        raise HTTPException(400, f"Peer {data.peer_id} not in deployment")

    status_map[data.peer_id] = "success" if data.success else "failed"
    max_req_map[data.peer_id] = data.max_req_in_batch

    # If every peer succeeded, mark the whole deployment ready; if any failed, fail it.
    if all(s == "success" for s in status_map.values()):
        active_deployments[data.model_name]["status"] = "ready"
        max_req = min(max_req_map.values())
        print(
            f"✅ Deployment for {data.model_name} is now READY, Max req / batch is {max_req}"
        )
        active_deployments[data.model_name]["max_req_per_batch"] = max_req

    elif any(s == "failed" for s in status_map.values()):
        active_deployments[data.model_name]["status"] = "failed"
        print(f"❌ Deployment for {data.model_name} has FAILED")

    return {"status": "ok"}


class CompletionData(BaseModel):
    batch_id: str
    output_text: List[str]
    peer_id: str
    timestamp: int


@app.post("/completion")
async def completion(completion: CompletionData):
    """
    Receive completion data from a peer.
    """
    batch_id = completion.batch_id
    # print(f"Completion: {completion}")
    if batch_id in active_inferences:
        inference_state = active_inferences[batch_id]
        inference_state["status"] = "completed"
        inference_state["result"] = completion.output_text
        inference_state["completed_at"] = time.time()
        inference_state["processing_time"] = (
            inference_state["completed_at"] - inference_state["started_at"]
        )
        print(f"Result: {completion.output_text}")
        print(
            f"✅ Inference {batch_id} completed in {inference_state['processing_time']:.2f}s"
        )

        # Process each request
        req_ids = inference_state["inflight"]
        for req_id in req_ids:
            logger.info(
                f"request-id: {req_id}, token: {len(active_requests[req_id]['tokens'])}"
            )
            active_requests[req_id]["complete"] = True
            active_requests[req_id]["event"].set()

    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
async def hello_world():
    return "<html><body><h1>Hello, World!</h1><p>This is a bare debug page.</p></body></html>"


# new inference request model
class InferenceRequest(BaseModel):
    model_name: str
    input_text: str
    max_tokens: int = 100


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

    # 2. Initialize request_id and add it to active_inferences
    request_id = str(uuid.uuid4())

    req = req_batcher.Request(
        id=request_id,
        prompt=request.input_text,
        model_name=request.model_name,
        sampling_params={"max_tokens": request.max_tokens},
    )

    task = asyncio.create_task(
        active_deployments[request.model_name]["batcher"].add(req)
    )
    await task

    return InferenceResponse(
        request_id=request_id, status="batched", result=None, processing_time=None
    )


async def send_batch(batch_id: str, model_name: str, queue: List[req_batcher.Request]):
    global active_deployments, active_inferences

    req_ids = [req.id for req in queue]

    active_inferences[str(batch_id)] = {
        "status": "batch submitted",
        "model_name": model_name,
        "request_id": req_ids,
        "inflight": req_ids,
        "started_at": time.time(),
        "result": None,
    }

    for req_id in req_ids:
        active_requests[req_id].update({"batch_id": batch_id})

    deployment_map = active_deployments[model_name]["deployment_map"]
    pipeline = list(deployment_map.keys())

    inference_payload = {
        "batch_id": batch_id,
        "model_name": model_name,
        "input_text": [req.prompt for req in queue],
        "pipeline": pipeline,
        "sampling_params": [req.sampling_params for req in queue],
        "assigned_layers": deployment_map,
        "timestamp": time.time(),
    }

    instruction_payload = json.dumps(inference_payload).encode()
    instruction_payload = np.frombuffer(instruction_payload, dtype=np.uint8)

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

    print(
        f"🚀 Batch sent out for {batch_id}, batch size is {len(queue)}, model {model_name}"
    )


class StreamingRequest(BaseModel):
    batch_id: str
    tokens: List[str | None]


@app.post("/streaming")
async def streaming(request: StreamingRequest):
    global active_inferences, active_requests

    try:
        req_ids = active_inferences[request.batch_id]["inflight"]

        # print(
        #     f"/streaming - request.tokens: {request.tokens}, len(req_ids): {len(req_ids)}"
        # )

        if len(request.tokens) != len(req_ids):
            raise ValueError(
                "Length of received tokens =/= Number of reqs inflight in batch"
            )

        # Update token lists or delete inflight if requests is done
        to_remove = []
        for i in range(len(req_ids)):
            token = request.tokens[i]
            if token is not None:
                active_requests[req_ids[i]]["tokens"].append(token)
            else:
                active_requests[req_ids[i]]["complete"] = True
                to_remove.append(i)
            active_requests[req_ids[i]]["event"].set()

        # Remove requests that are done
        for i in reversed(to_remove):
            del req_ids[i]

        # print(f"/streaming - {req_ids}")

    except Exception as e:
        print(f"Exception in /streaming endpoint: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()

    return {"status": "ok"}


# Helper function to generate a large string prompt from messages object passed in OpenAI API
def collect_messages(messages: List[Dict], batched: bool = False):
    # Only keeps 'user', 'assistant', 'developer', 'system' messages
    # role_whitelist = ["user", "assistant", "developer", "system"]

    if batched:
        normalized_batched = []
        for prompt in messages:
            normalized = []
            for p in prompt:
                role = p.get("role", "").strip().lower()
                content = p.get("content", "")

                if role == "developer":
                    role = "system"
                if role in ("system", "assistant", "user", "tool"):
                    normalized.append({"role": role, "content": content})
                else:
                    print(f"❌ Skipping message with role: {role}")

            normalized_batched.append(normalized)

        return normalized_batched

    normalized = []
    for m in messages:
        role = m.get("role", "").strip().lower()
        content = m.get("content", "")

        if role == "developer":
            role = "system"
        if role in ("system", "assistant", "user", "tool"):
            normalized.append({"role": role, "content": content})
        else:
            print(f"❌ Skipping message with role: {role}")
    print("Normalized Input looks like this -> ", normalized)
    return normalized
    # return to_return


class ChatCompletionRequest(BaseModel):
    """Schema for accepting chat completion requests from the user"""

    model: str
    stream: bool
    # start of vLLM Sampling Parameters
    messages: List[Dict[str, Any]]
    max_completion_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    min_tokens: Optional[int] = None
    seed: Optional[int] = None
    frequency_penalty: Optional[float] = None
    repetition_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    n: Optional[int] = 1
    eos_token_id: Optional[List[int]] = None
    stop: Optional[List[str]] = None


def build_sampling_params(request: ChatCompletionRequest):
    """Build the sampling parameters for the chat completion request"""
    sampling_params = {}
    if request.max_completion_tokens:
        sampling_params["max_tokens"] = request.max_completion_tokens
    if request.temperature:
        sampling_params["temperature"] = request.temperature
    if request.top_p:
        sampling_params["top_p"] = request.top_p
    if request.top_k:
        sampling_params["top_k"] = request.top_k
    if request.min_p:
        sampling_params["min_p"] = request.min_p
    if request.seed:
        sampling_params["seed"] = request.seed
    if request.frequency_penalty:
        sampling_params["frequency_penalty"] = request.frequency_penalty
    if request.repetition_penalty:
        sampling_params["repetition_penalty"] = request.repetition_penalty
    if request.presence_penalty:
        sampling_params["presence_penalty"] = request.presence_penalty
    if request.eos_token_id:
        sampling_params["stop_token_ids"] = (
            request.eos_token_id
        )  # vllm does not support eos_token_id so we use stop_token_ids
    if request.stop:
        sampling_params["stop"] = request.stop
    # hardcode n =1
    sampling_params["n"] = 1
    return sampling_params


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    # TODO: Ignore the authorization bearer key stuff in the header for now

    global active_requests

    model = request.model
    stream = request.stream
    messages = request.messages
    # max_tokens = request.max_completion_tokens

    input_text = collect_messages(messages)
    sampling_params = build_sampling_params(request)

    # 0. Check if model deployed or not
    if model not in active_deployments:
        raise HTTPException(
            status_code=404,
            detail=f"No deployment found for model {model}",
        )

    # check if deployment is ready
    if active_deployments[model]["status"] != "ready":
        raise HTTPException(
            status_code=404,
            detail=f"Deployment for model {model} is not ready",
        )

    # 1. Get the deployment instructions
    deployment_map = active_deployments[model]["deployment_map"]
    if not deployment_map:
        raise HTTPException(
            status_code=404,
            detail=f"No deployment map found for model {model}",
        )

    request_id = str(uuid.uuid4())
    event = asyncio.Event()
    event.clear()
    active_requests[request_id] = {
        "tokens": [],
        "event": event,
        "complete": False,
        "failed": False,
    }

    max_batch = active_deployments[model]["max_req_per_batch"]
    deployment_map = active_deployments[model]["deployment_map"]
    pipeline = list(deployment_map.keys())

    payload = {
        "request_id": request_id,
        "prompt": input_text,
        "model": model,
        "sampling_params": sampling_params,
        "max_batch_size": max_batch,
    }
    payload = json.dumps(payload).encode()
    payload = np.frombuffer(payload, dtype=np.uint8)

    # Don't wait for completion
    for peer in pipeline:
        asyncio.create_task(tensor_transport.send(peer, "request", payload))

    if not stream:
        pass  # work on stream first

    # TODO: the great GPT gave me these headers, have to verify what they mean
    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        # Helps disable proxy buffering on Nginx (see notes below)
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(
        stream_token_response(request_id, model),
        media_type="text/event-stream",
        headers=headers,
    )


# Helper function to pack data into format for SSE
def sse_pack(obj: dict) -> bytes:
    # SSE requires "data: <text>\n\n" per message
    return f"data: {json.dumps(obj, separators=(',', ':'))}\n\n".encode("utf-8")


# The async generator to give to the StreamingResponse object
async def stream_token_response(request_id: str, model_name: str):
    global active_requests

    try:
        # Yield one chunk first to give request_id
        init_chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": time.time(),
            "model": model_name,
            "choices": [{"index": 0, "delta": {"content": ""}}],
        }
        yield sse_pack(init_chunk)

        req_metadata = active_requests[request_id]
        req_metadata["read_index"] = -1
        event = req_metadata["event"]

        while True:
            await event.wait()
            event.clear()  # Resets for next wait

            last_read = req_metadata["read_index"]
            tokens = req_metadata["tokens"]

            if last_read + 1 >= len(tokens):
                to_stream = ""
            else:
                to_stream = "".join(req_metadata["tokens"][last_read + 1 :])

            req_metadata["read_index"] = len(tokens) - 1
            chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": time.time(),
                "model": model_name,
                "choices": [{"index": 0, "delta": {"content": to_stream}}],
            }

            # Stream remaining tokens and end SSE
            if req_metadata["complete"]:
                yield sse_pack(chunk)
                yield b"data: [DONE]\n\n"
                return

            elif req_metadata["failed"]:
                yield sse_pack(chunk)
                yield b"error"
                return

            # Normal streaming
            else:
                yield sse_pack(chunk)

    except Exception as e:
        print(f"[ERR] stream_token_response - {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()


class BatchFailRequest(BaseModel):
    batch_id: str
    peer_id: str
    error: str


@app.post("/batch_failed")
async def batch_failed(request: BatchFailRequest):
    batch_id = request.batch_id

    try:
        batch = active_inferences[batch_id]
        req_ids = batch["request_id"]
        batch["status"] = "failed"

        print(
            f"batch_failed - batch_id: {batch_id}, peer_id: {request.peer_id}, error: {request.error}"
        )

        for req_id in req_ids:
            active_requests[req_id]["failed"] = True
            active_requests[req_id]["event"].set()
    except Exception as e:
        print(f"/batch_failed - {type(e).__name__}: {e}")

    return


class BatchInfoRequest(BaseModel):
    batch_id: str
    request_id: List[str]


@app.post("/batch_info")
async def batch_info(request: BatchInfoRequest):
    active_inferences[request.batch_id] = {
        "status": "inflight",
        "request_id": request.request_id,
        "inflight": request.request_id,
        "started_at": time.time(),
        "result": None,
    }


@app.post("/client_batch_test")
async def client_batch_test(request: Request):
    body = await request.json()
    model_name = body.get("model")
    prompt = body.get("prompt")
    deployment_map = active_deployments[model_name]["deployment_map"]
    pipeline = list(deployment_map.keys())

    payload = {
        "request_id": str(uuid.uuid4()),
        "prompt": prompt,
        "model": model_name,
        "sampling_params": {"max_tokens": 1000},
        "max_batch_size": 100,
    }
    payload = json.dumps(payload).encode()
    payload = np.frombuffer(payload, dtype=np.uint8)

    for peer in pipeline:
        await asyncio.create_task(tensor_transport.send(peer, "request", payload))


class InferenceRequestBatched(BaseModel):
    model_name: str
    path_of_csv: str
    name_of_column: str
    system_prompt: str
    delimiter: str = ","  # CSV delimiter, default comma
    max_buffer_size: int = 1000
    min_buffer_size: int = 500
    starting_id: int = 0
    dry_run: bool = False
    max_completion_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    min_tokens: Optional[int] = None
    seed: Optional[int] = None
    frequency_penalty: Optional[float] = None
    repetition_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    n: Optional[int] = 1
    eos_token_id: Optional[List[int]] = None
    stop: Optional[List[str]] = None
    path_to_save_results: Optional[str] = None


@app.post("/infer_batched")
async def infer_batched(request: InferenceRequestBatched):
    global active_deployments, active_inferences, _process_batch_continuously
    # Check if model is deployed
    if request.model_name not in active_deployments:
        raise HTTPException(
            status_code=404,
            detail=f"Model {request.model_name} is not deployed",
        )

    # CLEAR OLD BATCH STATE FOR THIS FILE - ensures fresh start every time
    await clear_batch_processing_state_for_file(request.path_of_csv)

    # create an entry in the batch_processing_tasks which tracks the state of
    # the batch in the background
    task_id = str(uuid.uuid4())
    batch_processing_tasks[task_id] = {
        "status": "queued",
        "request": request,
        "started_at": time.time(),
        "progress": {
            "lines_processed": 0,
            "batches_sent": 0,
            "current_buffer_size": 0,
        },
    }
    # 1. Get the deployment map
    deployment_map = active_deployments[request.model_name]["deployment_map"]
    if not deployment_map:
        raise HTTPException(
            status_code=404,
            detail=f"No deployment map found for model {request.model_name}",
        )
    # 2. Get the pipeline from the active deployments
    pipeline = list(deployment_map.keys())
    # this is a coroutine in the asyncio loop by adding it to the queue but it doesnt wait
    # if _process_batch_continuously is not None:
    asyncio.create_task(_process_batch_continuously(request, task_id, pipeline))

    return {
        "status": "queued",
        "task_id": task_id,
        "message": "Batch Processing Task Queued - use /batch_task_status/{task_id} to monitor",
    }


@app.get("/batch_task_status/{task_id}")
async def get_batch_task_status(task_id: str):
    """Get real-time status and progress of a batch processing task"""
    if task_id not in batch_processing_tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    task = batch_processing_tasks[task_id]
    progress = task.get("progress", {})

    response = {
        "task_id": task_id,
        "status": task.get("status", "unknown"),
        "started_at": task.get("started_at"),
        "progress": {
            "lines_processed": progress.get("lines_processed", 0),
            "batches_sent": progress.get("batches_sent", 0),
            "current_buffer_size": progress.get("current_buffer_size", 0),
        },
    }

    # Add error if failed
    if task.get("error"):
        response["error"] = task["error"]

    # Add performance metrics if running
    if task.get("started_at") and task.get("status") in ["processing", "completed"]:
        elapsed = time.time() - task["started_at"]
        lines = progress.get("lines_processed", 0)
        response["performance"] = {
            "elapsed_seconds": round(elapsed, 1),
            "lines_per_second": round(lines / elapsed, 2) if elapsed > 0 else 0,
        }

    return response


async def _process_batch_continuously(request, task_id, pipeline):
    global latest_effective_buffer_size
    print(f"🚀 Starting continuous batch processing for task {task_id}")

    # Mark as processing
    batch_processing_tasks[task_id]["status"] = "processing"

    while True:
        try:
            print(f"🔄 Monitored the Effective Buffer Size (task: {task_id})")
            # check the buffer size of the peers
            latest_effective_buffer_size = await get_current_buffer_size_from_peers()
            print(
                f"📊 Buffer: {latest_effective_buffer_size}, Min: {request.min_buffer_size}, Max: {request.max_buffer_size}"
            )

            # Update buffer size in progress
            batch_processing_tasks[task_id]["progress"]["current_buffer_size"] = (
                latest_effective_buffer_size
            )

            # check if the buffer size is below the threshold
            if latest_effective_buffer_size < request.min_buffer_size:
                num_samples_to_send_in_batch = (
                    request.max_buffer_size - latest_effective_buffer_size
                )
                print(
                    f"📤 Sending {num_samples_to_send_in_batch} samples to fill buffer"
                )

                # take the system prompt and the number of samples, and create the prompts to be sent
                (
                    samples_to_send_for_inference,
                    batch_state,
                ) = await process_chunk_from_s3(
                    request.path_of_csv,
                    task_id,
                    num_samples_to_send_in_batch,
                    request.system_prompt,
                    request.name_of_column,
                    request.delimiter,
                )

                # break the infinite loop if there are no samples to send for inference
                if not samples_to_send_for_inference:
                    print("✅ No more samples to send - file processing completed!")
                    batch_processing_tasks[task_id]["status"] = "completed"
                    return

                print(
                    f"✅ Loaded {len(samples_to_send_for_inference)} samples from CSV"
                )
                input_text = collect_messages(
                    samples_to_send_for_inference, batched=True
                )
                sampling_params = build_sampling_params(request)

                # just to know what the max batch size is for vLLM
                max_batch = active_deployments[request.model_name]["max_req_per_batch"]

                # Update progress with latest state from MongoDB
                batch_processing_tasks[task_id]["progress"].update(
                    {
                        "lines_processed": batch_state.get("last_processed_line", 0),
                        "batches_sent": batch_state.get("batch_count", 0),
                    }
                )

                # create the payload to send to the peers
                payload = {
                    "task_id": task_id,
                    "prompt": input_text,
                    "model": request.model_name,
                    "sampling_params": sampling_params,
                    "max_batch_size": max_batch,
                    "file_id": request.path_of_csv,
                    "batch_number": batch_state["batch_count"],
                }
                payload = json.dumps(payload).encode()
                payload = np.frombuffer(payload, dtype=np.uint8)

                # send the payload to the peers
                # only send it to the first peer, the rest of them really dont need the batch information
                if not request.dry_run:
                    print(
                        f"📡 Sending batch {batch_state['batch_count']} to peer {pipeline[0][:8]}..."
                    )
                    await tensor_transport.send(pipeline[0], "batch_inject", payload)
                    # update the batch count after sending it to the peers
                    batch_state["batch_count"] += 1
                    print(
                        f"✅ Batch {batch_state['batch_count'] - 1} sent successfully"
                    )
                else:
                    print(f"🔍 Dry run: Would have sent batch to peer {pipeline[0]}")
            else:
                print(
                    f"⏸️  Buffer is full enough ({latest_effective_buffer_size} >= {request.min_buffer_size}), waiting..."
                )

            print(f"⏳ Sleeping for {BATCHED_BUFFER_TIMEOUT}s before next check...")
            await asyncio.sleep(BATCHED_BUFFER_TIMEOUT)

        except Exception as e:
            print(f"❌ ERROR in _process_batch_continuously: {type(e).__name__}: {e}")
            import traceback

            traceback.print_exc()
            batch_processing_tasks[task_id]["status"] = "failed"
            batch_processing_tasks[task_id]["error"] = str(e)
            return


async def get_current_buffer_size_from_peers():
    """
    Background service to find the current_buffer_size for all peers and find the effective current buffer size.
    This represents the current buffer size of the HEAD peer.
    """
    peer_current_buffer_sizes = []
    effective_current_buffer_size = 0

    # Check each active peer's latest current buffer size
    for peer_id in active_peer_ids:
        try:
            # Get the most recent metrics for this peer
            metrics_history = await get_peer_metrics(peer_id, time_window=60)
            if not metrics_history:
                print(f"⚠️  No metrics found for peer {peer_id[:8]}...")
                continue

            latest_metrics = metrics_history[0]["metrics"]
            current_buffer_size = latest_metrics.get("current_buffer_size", -1)

            # Only count positive buffer sizes (first peer in pipeline)
            if current_buffer_size >= 0:
                peer_current_buffer_sizes.append(current_buffer_size)
                print(f"📊 Peer {peer_id[:8]}: buffer = {current_buffer_size}")
        except Exception as e:
            print(f"⚠️  Error getting metrics for peer {peer_id[:8]}: {e}")
            continue

    if peer_current_buffer_sizes:
        effective_current_buffer_size = max(peer_current_buffer_sizes)
    else:
        effective_current_buffer_size = 0
        print("⚠️  No valid buffer sizes found from any peer, defaulting to 0")

    print(f"🏆 Effective buffer size across all peers: {effective_current_buffer_size}")
    return effective_current_buffer_size


async def process_chunk_from_s3(
    file_id: str,
    task_id: str,
    micro_batch_size: int,
    system_prompt: str,
    column_name: str,
    delimiter: str = ",",
):
    """
    1 - Find the Current State of the File (including column index)
    2 - Read the File in Chunks based on the next micro_batch_size
      - For that, resume the file from the given byte_position (given by Database)
      - For that, seek the file to the given byte_position
    3 - Extract only the specified column from each row
    4 - Return the chunk back to the /infer_batched endpoint
    5 - Update the current state of the file in the MongoDB Database
    """
    # 1 - Find the Current State of the File
    current_state = await get_csv_processing_state_by_file_id(file_id, task_id)
    if not current_state:
        current_state = {
            "next_byte_position": 0,
            "last_processed_line": 0,
            "batch_count": 0,
            "column_index": None,  # Store column index to avoid re-parsing header
        }

    # 2 - Read the File in Chunks based on the next batch_size
    # Open the file
    with open(file_id, "r", encoding="utf-8") as s3_file:
        column_index = current_state.get("column_index")

        # HACK: Seek to where we left off (skip if first batch)
        if current_state["next_byte_position"] > 0:
            print(f"📍 Seeking to byte {current_state['next_byte_position']}")
            s3_file.seek(current_state["next_byte_position"])
        else:
            print("📍 Starting from beginning - parsing header")
            # Read and parse header to find column index
            header_line = s3_file.readline()
            header_columns = [
                col.strip().strip('"') for col in header_line.strip().split(delimiter)
            ]

            try:
                column_index = header_columns.index(column_name)
                print(f"✅ Found column '{column_name}' at index {column_index}")
                print(f"📋 Header columns: {header_columns}")
            except ValueError:
                raise ValueError(
                    f"Column '{column_name}' not found in CSV header. Available columns: {header_columns}"
                )

        # Read the file in chunks based on the next micro_batch_size
        lines_processed = 0
        prompts_to_send_for_inference = []
        current_line = current_state["last_processed_line"]

        while lines_processed < micro_batch_size:
            # Read the raw line
            raw_line = s3_file.readline()

            # Check if we hit end of file
            if not raw_line:
                print("📄 Reached end of file!")
                break

            # Parse the CSV line using the specified delimiter
            # Handle quoted fields by simple strip (works for most cases)
            row = [col.strip().strip('"') for col in raw_line.strip().split(delimiter)]

            # Extract only the specified column
            if column_index < len(row):
                column_value = row[column_index]
                current_line += 1
                print(f"🔥 Processing line {current_line}: {column_value[:50]}...")

                # Add prompt to list with system prompt
                prompts_to_send_for_inference.append(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": column_value},
                    ]
                )
                lines_processed += 1
            else:
                print(
                    f"⚠️ Skipping malformed line {current_line + 1}: not enough columns"
                )
                current_line += 1

        # if no lines processed, return the empty list and the current state
        if lines_processed == 0:
            return [], current_state

        # Get the current byte position
        current_byte_position = s3_file.tell()

    # Save the current state of the file
    new_state = {
        "next_byte_position": current_byte_position,
        "last_processed_line": current_line,
        "batch_count": current_state["batch_count"] + 1,
        "column_index": column_index,  # Save column index for next batch
    }
    await save_csv_processing_state_by_file_id(file_id, task_id, new_state)
    print(f"🔄 Saved the current state of the file: {new_state}")
    return prompts_to_send_for_inference, new_state


@app.post("/batch_upload_complete")
async def batch_upload_complete(request: Request):
    """Endpoint for peers to notify when batch results are uploaded to S3"""
    data = await request.json()
    task_id = data.get("task_id")
    s3_path = data.get("s3_path")

    if not task_id or not s3_path:
        raise HTTPException(status_code=400, detail="task_id and s3_path required")

    if task_id in batch_processing_tasks:
        batch_processing_tasks[task_id]["s3_result_path"] = s3_path
        print(f"✅ Stored S3 result path for task {task_id}: {s3_path}")
        return {"status": "success", "message": "S3 path recorded"}

    return {"status": "warning", "message": "Task not found but path recorded"}


@app.get("/get_results/{task_id}")
async def get_results(task_id: str):
    """Download final results file from S3"""
    # Validate task exists
    if task_id not in batch_processing_tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    task = batch_processing_tasks[task_id]
    if task.get("status") != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Task status: {task.get('status')}. Only completed tasks have results.",
        )

    # Try to get stored S3 path first (most reliable)
    s3_full_path = task.get("s3_result_path")

    if s3_full_path:
        # Parse the S3 path
        if s3_full_path.startswith("s3://"):
            path_parts = s3_full_path[5:].split("/", 1)
            S3_RESULTS_BUCKET = path_parts[0]
            s3_key = path_parts[1] if len(path_parts) > 1 else ""
            print(f"✅ Using stored S3 path: {s3_full_path}")
        else:
            raise HTTPException(status_code=500, detail="Invalid S3 path format")
    else:
        # Fall back to searching (old behavior)
        print(f"⚠️ No stored S3 path for task {task_id}, falling back to search")

        request_data = task.get("request")
        if not request_data:
            raise HTTPException(status_code=404, detail="Request data not found")

        original_file_path = request_data.path_of_csv
        safe_file_id = (
            original_file_path.replace("\\", "_").replace("/", "_").replace(":", "")
        )

        S3_RESULTS_BUCKET = os.environ.get("S3_RESULTS_BUCKET", "tandemn-results")
        prefix = f"results/{safe_file_id}_final_"

        print(f"🔍 Searching S3 for: s3://{S3_RESULTS_BUCKET}/{prefix}*")

        try:
            response = s3_client.list_objects_v2(
                Bucket=S3_RESULTS_BUCKET, Prefix=prefix
            )

            if "Contents" not in response:
                raise HTTPException(
                    status_code=404,
                    detail=f"No results found. Searched: s3://{S3_RESULTS_BUCKET}/{prefix}*",
                )

            csv_files = [
                obj for obj in response["Contents"] if obj["Key"].endswith(".csv")
            ]
            if not csv_files:
                raise HTTPException(
                    status_code=404,
                    detail=f"Found files but none are CSV. Searched: {prefix}*",
                )

            latest_file = max(csv_files, key=lambda x: x["LastModified"])
            s3_key = latest_file["Key"]
            print(f"✅ Found result file via search: s3://{S3_RESULTS_BUCKET}/{s3_key}")

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "NoSuchBucket":
                raise HTTPException(
                    status_code=404, detail=f"S3 bucket not found: {S3_RESULTS_BUCKET}"
                )
            elif error_code == "AccessDenied":
                raise HTTPException(status_code=403, detail="Access denied to S3")
            raise HTTPException(status_code=500, detail=f"S3 error: {error_code}")

    # Generate presigned URL for download
    try:
        # Get file metadata
        head_response = s3_client.head_object(Bucket=S3_RESULTS_BUCKET, Key=s3_key)

        # Generate presigned URL
        signed_url = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": S3_RESULTS_BUCKET, "Key": s3_key},
            ExpiresIn=3600,  # 1 hour
        )

        return {
            "download_url": signed_url,
            "filename": s3_key.split("/")[-1],
            "file_size": head_response["ContentLength"],
            "last_modified": head_response["LastModified"].isoformat(),
        }

    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        if error_code == "NoSuchKey":
            raise HTTPException(
                status_code=404, detail=f"File not found in S3: {s3_key}"
            )
        elif error_code == "AccessDenied":
            raise HTTPException(status_code=403, detail="Access denied to S3")
        raise HTTPException(status_code=500, detail=f"S3 error: {error_code}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
