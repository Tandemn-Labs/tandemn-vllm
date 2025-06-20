import asyncio
import socket
import tempfile
import os
import json
import torch
import iroh
import httpx
from iroh.iroh_ffi import uniffi_set_event_loop

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

# Constants for document keys
TRIGGER_KEY = "job_trigger"  # Key used to trigger a new computation job
FINAL_RESULT_KEY = "final_result"  # Key used to store the final computation result

# Predefined matrices for each position in the pipeline
MATRIX_MAP = {
    0: torch.tensor([[2., 0.], [1., 2.]]),  # Matrix A (first machine)
    1: torch.tensor([[0., 1.], [1., 0.]]),  # Matrix B (second machine)
    2: torch.tensor([[1., 1.], [0., 1.]]),  # Matrix C (third machine)
}

async def get_shared_ticket():
    """
    Fetch the shared ticket from the server.
    
    Returns:
        str: The shared ticket for joining the Iroh document
        
    Raises:
        Exception: If unable to fetch the ticket from the server
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get('http://localhost:8000/ticket')
            response.raise_for_status()
            ticket_data = response.json()
            ticket = ticket_data["ticket"]
            print(f"✅ Fetched shared ticket from server")
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

async def upload_metrics(doc, author, peer_id: str):
    """Upload system metrics to the document and database."""
    try:
        metrics = get_system_metrics()
        formatted_metrics = format_metrics_for_db(metrics)
        value = f"CPU: {metrics.cpu_percent}%\nRAM: {metrics.ram_percent}%".encode()
        
        # Store in document
        await doc.set_bytes(author, peer_id.encode(), value)
        
        # Store in database
        await update_peer_metrics(peer_id, formatted_metrics)
        print(f"✅ Uploaded metrics for {peer_id}")
    except Exception as e:
        print(f"❌ Failed to upload metrics: {e}")

async def process_once(doc, author, peer_id: str, next_peer: str, is_first: bool, is_last: bool, local_matrix: torch.Tensor, node):
    """
    Process one computation job in the pipeline.
    
    Args:
        doc: Iroh document
        author: Iroh author for writing
        peer_id: This peer's ID
        next_peer: ID of the next peer in the pipeline
        is_first: Whether this is the first machine in the pipeline
        is_last: Whether this is the last machine in the pipeline
        local_matrix: The matrix assigned to this peer
        node: Iroh node
    """
    try:
        if is_first:
            # First machine waits for job trigger
            trigger = await receive_blob(doc, TRIGGER_KEY, node)
            print(f"📥 Received trigger: {trigger}")
            input_matrix = trigger
        else:
            # Other machines wait for input from previous machine
            input_matrix = await receive_blob(doc, peer_id, node)
            print(f"📥 Received input: {input_matrix}")
        
        # Perform matrix multiplication
        result = torch.matmul(input_matrix, local_matrix)
        print(f"🔢 Computed result: {result}")
        
        if is_last:
            # Last machine stores final result
            await send_blob(doc, author, FINAL_RESULT_KEY, result)
            print("✅ Stored final result")
        else:
            # Pass result to next machine
            await send_blob(doc, author, next_peer, result)
            print(f"📤 Sent result to {next_peer}")
            
    except Exception as e:
        print(f"❌ Error in computation: {e}")

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

async def continuous_heartbeat(doc, author, peer_id: str, interval_ms: int = 1000):
    """
    Continuously send heartbeat with metrics to the server.
    
    Args:
        doc: Iroh document
        author: Iroh author for writing  
        peer_id: This peer's ID
        interval_ms: Heartbeat interval in milliseconds (default: 1000ms = 1 second)
    """
    print(f"💓 Starting continuous heartbeat every {interval_ms}ms")
    
    while True:
        try:
            await upload_metrics(doc, author, peer_id)
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
    
    # Upload initial system metrics
    await upload_metrics(doc, author, peer_id)

    # Start continuous heartbeat as background task
    heartbeat_interval_ms = int(os.getenv("HEARTBEAT_INTERVAL_MS", "1000"))  # Default 1 second
    heartbeat_task = asyncio.create_task(
        continuous_heartbeat(doc, author, peer_id, heartbeat_interval_ms)
    )

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
        local_matrix = MATRIX_MAP.get(index)
        if local_matrix is None:
            print(f"❌ No matrix found for pipeline index {index}")
            return

        print(f"✅ Position: {index} | First: {is_first} | Last: {is_last}")

        # Main processing loop - continuously process computation jobs
        while True:
            await process_once(doc, author, peer_id, next_peer, is_first, is_last, local_matrix, node)
            await asyncio.sleep(2)  # Small delay between processing cycles

    except Exception as e:
        print(f"❌ Error in main loop: {e}")
    finally:
        # Cancel heartbeat task
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass
        
        # Deregister peer when shutting down
        await deregister_peer(peer_id)
        print(f"👋 Deregistered {peer_id} from pipeline")

if __name__ == "__main__":
    asyncio.run(main()) 