import asyncio
import psutil
import iroh
import tempfile
import os
import socket
import torch
import numpy as np
import json
from pathlib import Path
from iroh.iroh_ffi import uniffi_set_event_loop

SHARED_TICKET = "docaaacajuwrxbzowdxfy2l3gakhubkd37v7weccmhylr2jdgkabt4fkkq6aeeck4iy2gtqqthfto26v7rfvlm2iye7sxlvtg6gqv4mqc5vaodj2ajdnb2hi4dthixs65ltmuys2mjoojswyylzfzuxe33ifzxgk5dxn5zgwlrpamaavqca7kfncayaqj7p6sgmvqbabat675eivuid"
PIPELINE_FILE = Path("pipeline.txt")
TRIGGER_KEY = "job_trigger"
FINAL_RESULT_KEY = "final_result"

MATRIX_MAP = {
    0: torch.tensor([[2., 0.], [1., 2.]]),  # A
    1: torch.tensor([[0., 1.], [1., 0.]]),  # B
    2: torch.tensor([[1., 1.], [0., 1.]]),  # C
}

def load_pipeline():
    if not PIPELINE_FILE.exists():
        print(f"❌ pipeline.txt not found at {PIPELINE_FILE.resolve()}")
        return []
    with open(PIPELINE_FILE, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
        return list(dict.fromkeys(lines))  # remove duplicates while preserving order

async def send_blob(doc, author, peer_id: str, data: torch.Tensor):
    try:
        encoded = json.dumps(data.tolist()).encode()
        await doc.set_bytes(author, peer_id.encode(), encoded)
        print(f"📤 Sent to {peer_id}: {data}")
    except Exception as e:
        print(f"❌ Failed to send to {peer_id}: {e}")

async def receive_blob(doc, peer_id: str, node):
    seen = set()
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
        await asyncio.sleep(2)

async def wait_for_trigger(doc, node):
    seen = set()
    while True:
        try:
            entries = await doc.get_many(iroh.Query.all(None))
            for entry in entries:
                key = entry.key().decode()
                if key != TRIGGER_KEY:
                    continue
                hash = entry.content_hash()
                if hash in seen:
                    continue
                seen.add(hash)
                content = await node.blobs().read_to_bytes(hash)
                tensor = torch.tensor(json.loads(content.decode()))
                return tensor
        except Exception as e:
            print(f"❌ Error while waiting for trigger: {e}")
        await asyncio.sleep(2)

async def upload_metrics(doc, author, peer_id):
    key = peer_id.encode()
    cpu = psutil.cpu_percent(interval=1)
    ram = psutil.virtual_memory().percent
    value = f"CPU: {cpu}%\nRAM: {ram}%".encode()
    try:
        await doc.set_bytes(author, key, value)
        print(f"✅ Metrics uploaded for {peer_id}")
    except Exception as e:
        print(f"❌ Failed to upload metrics: {e}")

async def process_once(doc, author, peer_id, next_peer, is_first, is_last, local_matrix, node):
    if is_first:
        tensor = await wait_for_trigger(doc, node)
    else:
        tensor = await receive_blob(doc, peer_id, node)
    result = local_matrix @ tensor
    if is_last:
        await doc.set_bytes(author, FINAL_RESULT_KEY.encode(), json.dumps(result.tolist()).encode())
        print(f"✅ {peer_id[:6]} pushed result to EC2")
    else:
        await send_blob(doc, author, next_peer, result)
        print(f"➡️ {peer_id[:6]} forwarded to {next_peer[:6]}")

async def main():
    uniffi_set_event_loop(asyncio.get_running_loop())

    hostname = socket.gethostname()
    data_dir = os.path.join(tempfile.gettempdir(), f"iroh_node_{hostname}")
    options = iroh.NodeOptions()
    options.enable_docs = True

    node = await iroh.Iroh.memory_with_options(options)
    peer_id = await node.net().node_id()
    print(f"🤖 Running as peer: {peer_id}")

    doc = await node.docs().join(iroh.DocTicket(SHARED_TICKET.strip()))
    author = await node.authors().create()
    await upload_metrics(doc, author, peer_id)

    print("⏳ Waiting for pipeline.txt to include me...")
    while True:
        pipeline = load_pipeline()
        if peer_id in pipeline:
            break
        await asyncio.sleep(2)

    unique_pipeline = list(dict.fromkeys(pipeline))
    index = unique_pipeline.index(peer_id)
    is_first = index == 0
    is_last = index == len(unique_pipeline) - 1
    next_peer = unique_pipeline[index + 1] if not is_last else None
    local_matrix = MATRIX_MAP.get(index)
    if local_matrix is None:
        print(f"❌ No matrix found for pipeline index {index}")
        return

    print(f"✅ Position: {index} | First: {is_first} | Last: {is_last}")

    while True:
        await process_once(doc, author, peer_id, next_peer, is_first, is_last, local_matrix, node)
        await asyncio.sleep(2)

if __name__ == "__main__":
    asyncio.run(main())