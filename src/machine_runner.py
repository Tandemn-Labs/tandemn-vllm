import asyncio
import socket
import tempfile
import os
import json
import torch
import iroh
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

# Shared ticket for all machines to join the same Iroh document
SHARED_TICKET = "docaaacb3jtxaxzwrmdb37cc3dvs2rcs3dipyixneft4kyug5vu4titkg7raf7vjk6n25wipwt4yyosiq3pkxwlgu5mjyzp54fibdsjz7vor6hrqajdnb2hi4dthixs65ltmuys2mjoojswyylzfzuxe33ifzxgk5dxn5zgwlrpamaaik3c5lnpcayavqiqaaoqwebabla7fgb5bmic"

# Constants for document keys
TRIGGER_KEY = "job_trigger"  # Key used to trigger a new computation job
FINAL_RESULT_KEY = "final_result"  # Key used to store the final computation result

# Predefined matrices for each position in the pipeline
MATRIX_MAP = {
    0: torch.tensor([[2., 0.], [1., 2.]]),  # Matrix A (first machine)
    1: torch.tensor([[0., 1.], [1., 0.]]),  # Matrix B (second machine)
    2: torch.tensor([[1., 1.], [0., 1.]]),  # Matrix C (third machine)
}

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

    # Join the shared document and create an author for writing
    doc = await node.docs().join(iroh.DocTicket(SHARED_TICKET.strip()))
    author = await node.authors().create()
    
    # Upload initial system metrics
    await upload_metrics(doc, author, peer_id)

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
        # Deregister peer when shutting down
        await deregister_peer(peer_id)
        print(f"👋 Deregistered {peer_id} from pipeline")

if __name__ == "__main__":
    asyncio.run(main()) 