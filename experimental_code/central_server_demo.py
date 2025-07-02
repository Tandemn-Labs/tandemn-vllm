#!/usr/bin/env python3
"""
Central Server Demo - Orchestrates distributed inference across peers
Uses Iroh blobs for efficient hidden state transfer between peers
"""
import asyncio
import iroh
import json
import time
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn
from datetime import datetime
from iroh import PublicKey, NodeAddr

# Initialize FastAPI
app = FastAPI(title="Central Server - Distributed Inference Demo")

# Global Iroh objects
node = None
doc = None
ticket = None
server_id = None

# Track active inference requests
active_inferences = {}  # request_id -> inference_state

# Add global gossip sink and callback
trigger_gossip_sink = None
completion_gossip_sink = None

# Constants for 32-byte gossip topics
TRIGGER_TOPIC = bytes("trigger_topic".ljust(32), "utf-8")[:32]
COMPLETION_TOPIC = bytes("completion_topic".ljust(32), "utf-8")[:32]

# Registered peers {pubkey_str: NodeAddr}
peer_table: Dict[str, NodeAddr] = {}

class CompletionCallback(iroh.GossipMessageCallback):
    async def on_message(self, msg):
        from iroh import MessageType
        t = msg.type()
        if t == MessageType.JOINED:
            print("🔎 Completion mesh membership:", msg.as_joined())
            return
            
        if t == MessageType.RECEIVED:
            rc = msg.as_received()
            result_data = json.loads(rc.content.decode())
            await handle_inference_completion(result_data)
            print(f"📥 Processed completion for request: {result_data.get('request_id')}")

# No-op callback for sinks where we do not need to process inbound messages
class NoopCallback(iroh.GossipMessageCallback):
    async def on_message(self, msg):
        return

class InferenceRequest(BaseModel):
    model_name: str
    input_text: str
    max_tokens: int = 100

class InferenceResponse(BaseModel):
    request_id: str
    status: str  # "processing", "completed", "failed"
    result: Optional[str] = None
    processing_time: Optional[float] = None

# Simple demo: 3-layer model distributed across 3 peers
DEMO_MODEL_CONFIG = {
    "demo_model": {
        "layers": [
            {"peer_id": "peer_1", "layer_idx": 0, "layer_type": "embedding"},
            {"peer_id": "peer_2", "layer_idx": 1, "layer_type": "transformer"},
            {"peer_id": "peer_3", "layer_idx": 2, "layer_type": "output"}
        ]
    }
}

@app.on_event("startup")
async def startup():
    """Initialize Iroh node and create shared document"""
    global node, doc, ticket, server_id, trigger_gossip_sink, completion_gossip_sink
    
    print("🚀 Starting Central Server...")
    
    # Setup Iroh
    iroh.iroh_ffi.uniffi_set_event_loop(asyncio.get_running_loop())
    
    options = iroh.NodeOptions()
    options.enable_gossip = True
    
    # options.enable_docs = True # probably not needed
    node = await iroh.Iroh.memory_with_options(options)
    server_id = await node.net().node_id()
    
    # Create shared document for orchestration
    # doc = await node.docs().create()
    # ticket = await doc.share(iroh.ShareMode.WRITE, iroh.AddrInfoOptions.RELAY_AND_ADDRESSES)
    
    print(f"✅ Central Server started: {server_id}")
    # print(f"📎 Share this ticket with peers:\n{ticket}\n")
    
    # Setup gossip topics
    # trigger_topic = b"inference_triggers" + bytes(server_id, 'utf-8')[:16]
    trigger_topic = bytes("trigger_topic".ljust(32), 'utf-8')[:32]  # Convert text to 32 bytes
    # completion_topic = b"inference_completions" + bytes(server_id, 'utf-8')[:16]
    completion_topic = bytes("completion_topic".ljust(32), 'utf-8')[:32]  # Convert text to 32 bytes
    # Subscribe to completions (updated on heartbeat)
    completion_cb = CompletionCallback()
    completion_gossip_sink = await node.gossip().subscribe(COMPLETION_TOPIC, [], completion_cb)
    print(f"📡 Subscribed to completions on topic: {COMPLETION_TOPIC} (will refresh on peer joins)")
    
    # Initialise trigger sink with zero peers for now
    await refresh_trigger_sink()
    # Initialise completion sink with zero peers
    await refresh_completion_sink()
    
    # Start background task to debug document state
    # asyncio.create_task(debug_doc_state())

# Removed old monitoring function - now using subscription callback

async def handle_inference_completion(result_data: dict):
    """Handle completed inference from final peer"""
    request_id = result_data.get("request_id")
    
    if request_id in active_inferences:
        inference_state = active_inferences[request_id]
        inference_state["status"] = "completed"
        inference_state["result"] = result_data.get("output_text", "")
        inference_state["completed_at"] = time.time()
        inference_state["processing_time"] = inference_state["completed_at"] - inference_state["started_at"]
        
        print(f"✅ Inference {request_id} completed in {inference_state['processing_time']:.2f}s")

@app.get("/ticket")
async def get_ticket():
    """Get the shared document ticket for peers to join"""
    return {"ticket": str(ticket)}

@app.post("/infer", response_model=InferenceResponse)
async def start_inference(request: InferenceRequest):
    """Start distributed inference across peer network"""
    
    # Generate unique request ID
    request_id = f"req_{int(time.time() * 1000)}_{len(active_inferences)}"
    
    # Check if model is configured
    if request.model_name not in DEMO_MODEL_CONFIG:
        raise HTTPException(status_code=400, detail=f"Model {request.model_name} not configured")
    
    model_config = DEMO_MODEL_CONFIG[request.model_name]
    
    # Create inference state
    inference_state = {
        "request_id": request_id,
        "model_name": request.model_name,
        "input_text": request.input_text,
        "max_tokens": request.max_tokens,
        "status": "processing",
        "started_at": time.time(),
        "result": None,
        "processing_time": None
    }
    
    active_inferences[request_id] = inference_state
    
    # Send inference trigger to first peer
    await send_inference_trigger(request_id, model_config, request)
    
    print(f"🚀 Started inference {request_id} for model {request.model_name}")
    
    return InferenceResponse(
        request_id=request_id,
        status="processing"
    )

async def send_inference_trigger(request_id: str, model_config: dict, request: InferenceRequest):
    """Send inference trigger to the first peer in the pipeline"""
    
    global trigger_gossip_sink, node, server_id
    # Create inference instruction
    inference_instruction = {
        "request_id": request_id,
        "model_name": request.model_name,
        "input_text": request.input_text,
        "max_tokens": request.max_tokens,
        "pipeline": model_config["layers"],
        "current_layer": 0,
        "timestamp": time.time()
    }
    instruction_bytes = json.dumps(inference_instruction).encode()
    print(instruction_bytes)
    # Ensure sink exists and up-to-date 
    await refresh_trigger_sink()
    await trigger_gossip_sink.broadcast(instruction_bytes)
    print(f"📤 Sent inference trigger via gossip for request {request_id}")

@app.get("/status/{request_id}", response_model=InferenceResponse)
async def get_inference_status(request_id: str):
    """Get status of a specific inference request"""
    
    if request_id not in active_inferences:
        raise HTTPException(status_code=404, detail="Inference request not found")
    
    inference_state = active_inferences[request_id]
    
    return InferenceResponse(
        request_id=request_id,
        status=inference_state["status"],
        result=inference_state.get("result"),
        processing_time=inference_state.get("processing_time")
    )

# @app.get("/active_inferences")
# async def list_active_inferences():
#     """List all active inference requests"""
    
#     summary = []
#     for req_id, state in active_inferences.items():
#         summary.append({
#             "request_id": req_id,
#             "model_name": state["model_name"],
#             "status": state["status"],
#             "started_at": state["started_at"],
#             "processing_time": state.get("processing_time")
#         })
    
#     return {
#         "total_active": len(active_inferences),
#         "inferences": sorted(summary, key=lambda x: x["started_at"], reverse=True)
#     }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "server_id": str(server_id),
        "active_inferences": len(active_inferences),
        "timestamp": datetime.now().isoformat()
    }

async def refresh_trigger_sink():
    """(Re)create the trigger gossip sink with current peer list."""
    global trigger_gossip_sink
    peer_ids = list(peer_table.keys())  # list of node_id strings
    # Recreate sink each time to update peer set
    trigger_gossip_sink = await node.gossip().subscribe(TRIGGER_TOPIC, peer_ids, NoopCallback())
    print(f"📡 Trigger sink refreshed with peers: {peer_ids}")

async def refresh_completion_sink():
    """(Re)create the completion gossip sink with current peer list."""
    global completion_gossip_sink
    peer_ids = list(peer_table.keys())
    completion_gossip_sink = await node.gossip().subscribe(COMPLETION_TOPIC, peer_ids, CompletionCallback())
    print(f"📡 Completion sink refreshed with peers: {peer_ids}")

class Heartbeat(BaseModel):
    node_id: str
    addresses: List[str]
    relay_url: Optional[str] = None

@app.post("/heartbeat")
async def register_peer(hb: Heartbeat):
    """Peer registration / heartbeat endpoint."""
    pk = PublicKey.from_string(hb.node_id)
    addr = NodeAddr(pk, hb.relay_url, hb.addresses)
    await node.net().add_node_addr(addr)
    peer_table[hb.node_id] = addr
    print(f"🗒️ Active peers: {list(peer_table.keys())}")
    await refresh_trigger_sink()
    await refresh_completion_sink()
    # Return server info so peer can connect back to us via gossip
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
    return {
        "status": "ok",
        "server_id": server_id,
        "server_addresses": srv_addr.direct_addresses(),
        "relay_url": srv_addr.relay_url(),
        "peers": peers_payload
    }

if __name__ == "__main__":
    print("🎯 Central Server Demo - Distributed Inference")
    print("=" * 50)
    print("Usage:")
    print("1. Start this server")
    print("2. Copy the ticket and start peer nodes")
    print("3. POST to /infer to trigger distributed inference")
    print("4. GET /status/{request_id} to check progress")
    print("=" * 50)
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info") 