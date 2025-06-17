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
    get_peer_metrics
)
from src.utils.gpu_utils import get_system_metrics, format_metrics_for_db
from src.utils.model_utils import download_config, estimate_parameters, estimate_vram

# Initialize FastAPI application
app = FastAPI(title="Iroh Tandem Server")

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

@app.on_event("startup")
async def startup():
    """Initialize the Iroh node and document on server startup"""
    global node, doc, ticket, server_peer_id

    print("🚀 Starting Iroh Tandem server...")
    # Configure async event loop for Iroh
    uniffi_set_event_loop(asyncio.get_running_loop())

    # Set up MongoDB collections
    await setup_collections()
    print("✅ MongoDB collections configured")

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