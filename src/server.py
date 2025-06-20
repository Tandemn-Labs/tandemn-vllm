import asyncio
import iroh
import torch
import json
from fastapi import FastAPI, HTTPException
from iroh.iroh_ffi import uniffi_set_event_loop
from pydantic import BaseModel

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
    distribute_layers_across_peers,
    shard_model_for_peers
)

# Initialize FastAPI application
app = FastAPI(title="Iroh Tandemn Server")

# Global variables for Iroh node and document management
node = None  # Iroh node instance
doc = None   # Iroh document instance
ticket = None  # Document sharing ticket
server_peer_id = None  # Unique identifier for this server instance

# Constants for document keys
TRIGGER_KEY = "job_trigger"  # Key for job initiation
FINAL_RESULT_KEY = "final_result"  # Key for final computation result

class ModelEstimationRequest(BaseModel):
    model_id: str
    hf_token: str
    qbits: int = DEFAULT_QBITS
    filename: str = DEFAULT_CONFIG_FILENAME

class ModelShardingRequest(BaseModel):
    model_id: str
    output_dir: str
    qbits: int = DEFAULT_QBITS
    hf_token: str = None  # Will use environment variable if not provided
    model_layers_key: str = "model.layers"
    config_layers_key: str = "num_hidden_layers"

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

@app.post("/shard_model")
async def shard_model(request: ModelShardingRequest):
    """
    Create an optimal distribution plan and shard a model across available peers.
    
    Args:
        request: ModelShardingRequest containing model_id, output_dir, and other parameters
        
    Returns:
        Information about the created shards including their paths and distribution plan
    """
    try:
        print(f"🔪 Starting model sharding for {request.model_id}")
        print(f"📁 Output directory: {request.output_dir}")

        # Get active peers and their metrics
        active_peers = await get_active_peers()
        if not active_peers:
            raise HTTPException(
                status_code=400,
                detail="No active peers available for sharding"
            )

        # Get VRAM availability for all active peers
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

        # Download model configuration
        config = await download_config(request.model_id, request.hf_token)

        # Create optimal distribution plan
        distribution_plan = distribute_layers_across_peers(
            config=config,
            peers_vram=peers_vram,
            q_bits=request.qbits
        )

        if not distribution_plan or "distribution" not in distribution_plan:
            raise HTTPException(
                status_code=500,
                detail="Failed to create distribution plan"
            )

        distribution = distribution_plan["distribution"]
        if not distribution:
            raise HTTPException(
                status_code=400,
                detail="No peer assignments generated in distribution plan"
            )

        # Create shards using the utility function
        shard_paths = shard_model_for_peers(
            model_id=request.model_id,
            distribution_plan=distribution_plan,
            output_dir=request.output_dir,
            model_layers_key=request.model_layers_key,
            config_layers_key=request.config_layers_key,
            hf_token=request.hf_token
        )

        # Prepare detailed response
        shard_info = {}
        total_shards = 0
        total_layers_assigned = 0

        for peer_id, shard_path in shard_paths.items():
            peer_distribution = distribution[peer_id]
            assigned_layers = peer_distribution["assigned_layers"]
            handles_embeddings = peer_distribution["handles_embeddings"]

            shard_info[peer_id] = {
                "shard_path": shard_path,
                "assigned_layers": assigned_layers,
                "handles_embeddings": handles_embeddings,
                "available_vram_gb": peer_distribution["available_vram_gb"],
                "estimated_vram_usage": peer_distribution["estimated_vram_usage"],
                "vram_utilization_percent": peer_distribution["vram_utilization_percent"]
            }

            total_shards += 1
            total_layers_assigned += assigned_layers

        print(f"✅ Successfully created {total_shards} shards with {total_layers_assigned} total layers")

        return {
            "status": "success",
            "model_id": request.model_id,
            "output_dir": request.output_dir,
            "total_shards": total_shards,
            "total_layers_assigned": total_layers_assigned,
            "model_info": distribution_plan.get("model_info", {}),
            "shard_details": shard_info,
            "distribution_plan": distribution_plan
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Model sharding failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to shard model: {str(e)}"
        )