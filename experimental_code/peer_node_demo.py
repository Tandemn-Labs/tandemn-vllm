#!/usr/bin/env python3
"""
Peer Node Demo - Processes layers in distributed inference pipeline
Uses Iroh blobs for efficient hidden state transfer between peers
"""
import asyncio
import iroh
import json
import time
import torch
import numpy as np
import argparse
import requests
from typing import Optional, Dict, Any
from iroh import LiveEventType, Hash
from iroh import MessageType


# Global Iroh objects
node = None
doc = None
peer_id = None
author = None

# Peer configuration
peer_config = {
    "peer_id": None,
    "layer_idx": None,
    "layer_type": None,
    "model_weights": None  # Simulated model weights
}

# Track processed requests to avoid duplicates
processed_triggers = set()
processed_hidden_states = set()

# Add global gossip sinks
trigger_gossip_sink = None
hidden_state_gossip_sink = None
completion_gossip_sink = None

# No-op callback
class NoopCallback(iroh.GossipMessageCallback):
    async def on_message(self, msg):
        return

class LayerProcessor:
    """Simulates a transformer layer processor"""
    
    def __init__(self, layer_idx: int, layer_type: str):
        self.layer_idx = layer_idx
        self.layer_type = layer_type
        
        # Simulate some model weights (in reality, these would be loaded from disk)
        if layer_type == "embedding":
            self.weights = torch.randn(512, 768)  # vocab_size x hidden_size
        elif layer_type == "transformer":
            self.weights = torch.randn(768, 768)  # hidden_size x hidden_size  
        elif layer_type == "output":
            self.weights = torch.randn(768, 512)  # hidden_size x vocab_size
        else:
            self.weights = torch.randn(768, 768)  # default
            
        print(f"✅ Initialized {layer_type} layer {layer_idx} with weights {self.weights.shape}")
    
    def process(self, input_data: torch.Tensor) -> torch.Tensor:
        """Process input through this layer"""
        
        if self.layer_type == "embedding":
            # Simulate token embedding lookup
            # input_data is token IDs (1, 10), weights is (512, 768)
            # We need to use the token IDs as indices to select from embedding table
            # For demo, we'll use one-hot encoding approach
            batch_size, seq_len = input_data.shape
            # Create one-hot encoding of token IDs
            one_hot = torch.zeros(batch_size, seq_len, 512)  # (1, 10, 512)
            one_hot.scatter_(2, input_data.unsqueeze(-1), 1)  # Set 1s at token positions
            # Now multiply: (1, 10, 512) × (512, 768) = (1, 10, 768)
            result = torch.matmul(one_hot, self.weights)
            
        elif self.layer_type == "transformer":
            # Simulate transformer layer computation
            # input_data is hidden states (1, 10, 768), weights is (768, 768)
            # For transformer layers, we typically apply the weight matrix to the last dimension
            # (1, 10, 768) × (768, 768) = (1, 10, 768)
            result = torch.matmul(input_data, self.weights)
            result = torch.relu(result)  # Simple activation
            
        elif self.layer_type == "output":
            # Simulate final output projection
            # input_data is hidden states (1, 10, 768), weights is (768, 512)
            # (1, 10, 768) × (768, 512) = (1, 10, 512)
            result = torch.matmul(input_data, self.weights)
            
        else:
            # Default: simple linear transformation
            result = torch.matmul(input_data, self.weights)
        
        # Add some processing delay to simulate computation
        time.sleep(0.1)
        
        print(f"🔢 Processed {self.layer_type} layer {self.layer_idx}: {input_data.shape} -> {result.shape}")
        return result

class TriggerCallback(iroh.GossipMessageCallback):
    def __init__(self, processor):
        self.processor = processor
    async def on_message(self, msg):
        t = msg.type()
        if t == MessageType.JOINED:
            print("🔎 Mesh membership:", msg.as_joined())
            return
        
        if t == MessageType.RECEIVED:
            rc = msg.as_received()
            instruction = json.loads(rc.content.decode())
            request_id = instruction["request_id"]
            if request_id in processed_triggers:
                return
            processed_triggers.add(request_id)
            print(f"📥 Received inference trigger for request {request_id}")
            if self.processor.layer_type == "embedding":
                input_tokens = torch.randint(0, 512, (1, 10))
                hidden_state = self.processor.process(input_tokens)
                await forward_to_next_peer(instruction, hidden_state)

class HiddenStateCallback(iroh.GossipMessageCallback):
    def __init__(self, processor):
        self.processor = processor
    async def on_message(self, msg):
        print(f"🔍 [DEBUG] HiddenStateCallback received message")
        t = msg.type()
        print(f"🔍 [DEBUG] Message type: {t}")
        if t == MessageType.JOINED:
            print("🔎 Hidden state mesh membership:", msg.as_joined())
            return
            
        if t == MessageType.RECEIVED:
            print(f"🔍 [DEBUG] Processing RECEIVED message")
            rc = msg.as_received()
            print(f"🔍 [DEBUG] Message content length: {len(rc.content)} bytes")
            
            try:
                ref = json.loads(rc.content.decode())
                print(f"🔍 [DEBUG] Parsed JSON reference: {list(ref.keys())}")
                request_id = ref["request_id"]
                print(f"🔍 [DEBUG] Request ID: {request_id}")
                
                if request_id in processed_hidden_states:
                    print(f"🔍 [DEBUG] Hidden state for request {request_id} already processed, skipping")
                    return
                processed_hidden_states.add(request_id)
                print(f"📥 Received hidden state for request {request_id}")
                
                blob_ticket = ref["blob_ticket"]
                print(f"🔍 [DEBUG] Blob ticket: {blob_ticket[:100]}...")  # First 100 chars
                shape = tuple(ref["tensor_shape"])
                print(f"🔍 [DEBUG] Expected tensor shape: {shape}")
                
                hidden_array = await download_hidden_state(request_id, blob_ticket)
                print(f"🔍 [DEBUG] Downloaded array shape: {hidden_array.shape}")
                
                # Reconstruct tensor with proper shape
                if self.processor.layer_type == "transformer":
                    hidden_state = torch.from_numpy(hidden_array).reshape(*shape)
                elif self.processor.layer_type == "output":
                    hidden_state = torch.from_numpy(hidden_array).reshape(*shape)
                else:
                    hidden_state = torch.from_numpy(hidden_array).reshape(-1)
                    
                print(f"🔍 [DEBUG] Reconstructed tensor shape: {hidden_state.shape}")
                output_state = self.processor.process(hidden_state)
                print(f"🔍 [DEBUG] Processed output shape: {output_state.shape}")
                
                instruction = ref["instruction"]
                await forward_to_next_peer(instruction, output_state)
                
            except Exception as e:
                print(f"❌ [DEBUG] Error processing hidden state message: {e}")
                print(f"❌ [DEBUG] Exception type: {type(e)}")
                import traceback
                traceback.print_exc()

class SimpleDownloadCallback(iroh.DownloadCallback):
    async def progress(self, progress):
        # For debugging, we could print some progress types, but keep 
        print(f"🔍 [DEBUG] Download progress: {progress}%")
        return

async def forward_to_next_peer(instruction, hidden_state):
    pipeline = instruction["pipeline"]
    current_idx = None
    for i, layer_info in enumerate(pipeline):
        if layer_info["peer_id"] == peer_config["peer_id"]:
            current_idx = i
            break
    if current_idx is None:
        print("⚠️ Could not find our position in pipeline")
        return
    request_id = instruction["request_id"]
    if current_idx == len(pipeline) - 1:
        await send_final_result(request_id, hidden_state)
    else:
        next_peer = pipeline[current_idx + 1]["peer_id"]
        await send_hidden_state(request_id, next_peer, hidden_state, instruction)

async def download_hidden_state(request_id, blob_ticket_str):
    """Download hidden state blob using ticket and reconstruct tensor"""
    try:
        print(f"🔍 [DEBUG] Starting blob download for request {request_id}")
        print(f"🔍 [DEBUG] Blob ticket string: {blob_ticket_str[:100]}...")  # First 100 chars
        
        # Parse the blob ticket
        blob_ticket = iroh.BlobTicket(blob_ticket_str)
        print(f"🔍 [DEBUG] Parsed blob ticket successfully")
        
        # Build download parameters from ticket
        blob_hash = blob_ticket.hash()
        opts = blob_ticket.as_download_options()
        print(f"🔍 [DEBUG] Hash from ticket: {blob_hash}")
        
        # Download the blob using the ticket-derived options
        print(f"🔍 [DEBUG] Starting blob download via node.blobs().download ...")
        await node.blobs().download(blob_hash, opts, SimpleDownloadCallback())
        print(f"🔍 [DEBUG] Download finished, reading bytes ...")
        
        # Read the blob data
        hidden_bytes = await node.blobs().read_to_bytes(blob_hash)
        print(f"🔍 [DEBUG] Read {len(hidden_bytes)} bytes")
        hidden_array = np.frombuffer(hidden_bytes, dtype=np.float32)
        print(f"🔍 [DEBUG] Converted to numpy array: {hidden_array.shape}")
        
        return hidden_array
        
    except Exception as e:
        print(f"❌ [DEBUG] Failed to download hidden state for {request_id}: {e}")
        print(f"❌ [DEBUG] Exception type: {type(e)}")
        import traceback
        traceback.print_exc()
        raise

async def send_hidden_state(request_id, next_peer_id, hidden_state, instruction):
    global hidden_state_gossip_sink
    print(f"🔍 [DEBUG] Starting to send hidden state for request {request_id} to {next_peer_id}")
    print(f"🔍 [DEBUG] Hidden state shape: {hidden_state.shape}")
    
    hidden_bytes = hidden_state.detach().cpu().numpy().tobytes()
    print(f"🔍 [DEBUG] Converted to {len(hidden_bytes)} bytes")
    
    try:
        print(f"🔍 [DEBUG] Adding bytes to blob store...")
        blob_outcome = await node.blobs().add_bytes(hidden_bytes)
        print(f"🔍 [DEBUG] Blob add outcome: {blob_outcome}")
        blob_hash = blob_outcome.hash  # Extract hash from BlobAddOutcome
        print(f"🔍 [DEBUG] Extracted blob hash: {blob_hash}")
        
        # Create blob ticket for sharing
        print(f"🔍 [DEBUG] Creating blob ticket...")
        blob_ticket = await node.blobs().share(blob_hash, iroh.BlobFormat.RAW, iroh.AddrInfoOptions.RELAY_AND_ADDRESSES)
        print(f"🔍 [DEBUG] Created blob ticket: {str(blob_ticket)[:100]}...")  # First 100 chars
        
        ref = {
            "request_id": request_id,
            "blob_ticket": str(blob_ticket),  # Send ticket instead of just hash
            "tensor_shape": list(hidden_state.shape),
            "tensor_dtype": "float32",
            "instruction": instruction
        }
        ref_bytes = json.dumps(ref).encode()
        print(f"🔍 [DEBUG] Created reference message: {len(ref_bytes)} bytes")
        
        if not hidden_state_gossip_sink:
            print(f"🔍 [DEBUG] Creating hidden state gossip sink...")
            topic = bytes("hidden_state_topic".ljust(32), 'utf-8')[:32]
            hidden_state_gossip_sink = await node.gossip().subscribe(topic, [], NoopCallback())
            print(f"🔍 [DEBUG] Hidden state gossip sink created")
            
        print(f"🔍 [DEBUG] Broadcasting hidden state reference...")
        await hidden_state_gossip_sink.broadcast(ref_bytes)
        print(f"📤 Sent hidden state to {next_peer_id} for request {request_id}")
        print(f"🔍 [DEBUG] Broadcast completed successfully")
        
    except Exception as e:
        print(f"❌ [DEBUG] Error in send_hidden_state: {e}")
        print(f"❌ [DEBUG] Exception type: {type(e)}")
        import traceback
        traceback.print_exc()
        raise

async def send_final_result(request_id, final_output):
    # Send HTTP POST to server instead of gossip
    output_text = f"Generated text for request {request_id} (shape: {final_output.shape})"
    completion_data = {
        "request_id": request_id,
        "output_text": output_text,
        "peer_id": peer_config["peer_id"],
        "timestamp": time.time()
    }
    
    try:
        # Get server URL from args (we'll need to pass it down)
        server_url = "http://localhost:8000"  # Default for now
        response = requests.post(f"{server_url}/completion", json=completion_data, timeout=5)
        if response.ok:
            print(f"✅ Sent final result to server for request {request_id}")
        else:
            print(f"❌ Failed to send completion: {response.status_code}")
    except Exception as e:
        print(f"❌ Error sending completion: {e}")

async def main():
    """Main function to run peer node"""
    global node, doc, peer_id, author
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Peer Node for Distributed Inference")
    parser.add_argument("--peer-id", required=True, help="Unique peer identifier (e.g., peer_1)")
    parser.add_argument("--layer-idx", type=int, required=True, help="Layer index (0, 1, 2, ...)")
    parser.add_argument("--layer-type", required=True, choices=["embedding", "transformer", "output"], help="Type of layer")
    parser.add_argument("--ticket", required=True, help="Iroh document ticket from central server")
    parser.add_argument("--server-url", default="http://localhost:8000", help="Central server base URL for heartbeat")
    
    args = parser.parse_args()
    
    # Configure peer
    peer_config["peer_id"] = args.peer_id
    peer_config["layer_idx"] = args.layer_idx
    peer_config["layer_type"] = args.layer_type
    
    print(f"🚀 Starting peer: {args.peer_id}")
    print(f"   Layer: {args.layer_idx} ({args.layer_type})")
    
    # Setup Iroh
    iroh.iroh_ffi.uniffi_set_event_loop(asyncio.get_running_loop())
    
    options = iroh.NodeOptions()
    options.enable_gossip = True
    # options.enable_docs = True
    node = await iroh.Iroh.memory_with_options(options)
    peer_id = await node.net().node_id()
    
    # Register with central server via heartbeat
    try:
        node_addr = await node.net().node_addr()
        payload = {
            "node_id": str(peer_id),
            "addresses": node_addr.direct_addresses(),
            "relay_url": node_addr.relay_url()
        }
        resp = requests.post(f"{args.server_url}/heartbeat", json=payload, timeout=5)
        if resp.ok:
            data = resp.json()
            srv_pk = iroh.PublicKey.from_string(data["server_id"])
            srv_addr = iroh.NodeAddr(srv_pk, data["relay_url"], data["server_addresses"])
            await node.net().add_node_addr(srv_addr)
            # add other peers addresses
            for p in data.get("peers", []):
                pk = iroh.PublicKey.from_string(p["node_id"])
                pa = iroh.NodeAddr(pk, p["relay_url"], p["addresses"])
                await node.net().add_node_addr(pa)
            
            # Build bootstrap list with server + other peers
            server_id = data["server_id"]
            other_peers = [p["node_id"] for p in data.get("peers", [])]
            bootstrap = [server_id] + other_peers
            
            print("📡 Heartbeat registered with server and peer addresses added")
        else:
            print(f"⚠️ Heartbeat failed: {resp.status_code}")
            bootstrap = []
    except Exception as e:
        print(f"⚠️ Heartbeat error: {e}")
        bootstrap = []
    
    # # Join shared document
    # doc = await node.docs().join(iroh.DocTicket(args.ticket))
    # author = await node.authors().create()
    
    # print(f"✅ Connected to shared document as {peer_id}")
    
    # Initialize layer processor
    processor = LayerProcessor(args.layer_idx, args.layer_type)
    
    # Subscribe to gossip topics
    trigger_topic = bytes("trigger_topic".ljust(32), 'utf-8')[:32]
    trigger_cb = TriggerCallback(processor)
    await node.gossip().subscribe(trigger_topic, bootstrap, trigger_cb)
    
    hidden_state_topic = bytes("hidden_state_topic".ljust(32), 'utf-8')[:32]
    hidden_cb = HiddenStateCallback(processor)
    await node.gossip().subscribe(hidden_state_topic, bootstrap, hidden_cb)
    
    print(f"🎯 Peer {args.peer_id} ready for inference requests (gossip mode)")
    print("   Waiting for inference triggers...")
    
    # Keep running
    try:
        while True:
            await asyncio.sleep(10)
            # Optionally clean up old processed requests
            if len(processed_triggers) > 1000 or len(processed_hidden_states) > 1000:
                processed_triggers.clear()
                processed_hidden_states.clear()
                print("🧹 Cleaned up processed requests cache")
                
    except KeyboardInterrupt:
        print(f"\n👋 Shutting down peer {args.peer_id}")

if __name__ == "__main__":
    asyncio.run(main()) 