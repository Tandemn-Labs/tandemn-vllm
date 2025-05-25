import asyncio
import psutil
import iroh
from fastapi import FastAPI
from iroh.iroh_ffi import uniffi_set_event_loop
from pathlib import Path

app = FastAPI()

node = None
doc = None
ticket = None
ec2_peer_id = None
seen_peers = []
pipeline_file = Path("pipeline.txt")

@app.on_event("startup")
async def startup():
    global node, doc, ticket, ec2_peer_id

    print("🚀 Starting EC2 FastAPI server with Iroh shared document...")
    uniffi_set_event_loop(asyncio.get_running_loop())

    options = iroh.NodeOptions()
    options.enable_docs = True
    node = await iroh.Iroh.memory_with_options(options)

    # Create shared doc and get ticket
    doc = await node.docs().create()
    ticket = await doc.share(iroh.ShareMode.WRITE, iroh.AddrInfoOptions.RELAY_AND_ADDRESSES)
    ec2_peer_id = await node.net().node_id()

    # Send EC2 metrics to doc (optional)
    author = await node.authors().create()
    key = ec2_peer_id.encode()
    cpu = psutil.cpu_percent(interval=1)
    ram = psutil.virtual_memory().percent
    value = f"CPU: {cpu}%\nRAM: {ram}%".encode()

    try:
        await doc.set_bytes(author, key, value)
        print(f"✅ EC2 metrics stored with key {ec2_peer_id}")
    except Exception as e:
        print(f"❌ Failed to send EC2 metrics: {e}")

    # Join doc for read access
    doc = await node.docs().join(ticket)

    print("✅ EC2 Iroh node started")
    print("📎 SHARE THIS TICKET WITH ALL INTERNAL MACHINES:\n")
    print(str(ticket) + "\n")

@app.get("/health")
async def health():
    global seen_peers, ec2_peer_id

    try:
        entries = await doc.get_many(iroh.Query.all(None))
        results = []

        for entry in entries:
            try:
                key = entry.key().decode()
                content = await node.blobs().read_to_bytes(entry.content_hash())

                if key == ec2_peer_id:
                    continue  # Skip EC2 itself

                if key not in seen_peers:
                    seen_peers.append(key)
                    print(f"🆕 New machine detected: {key}")
                    with open(pipeline_file, "a") as f:
                        f.write(key + "\n")

                results.append({
                    "machine_id": key,
                    "metrics": content.decode()
                })

            except Exception as inner_e:
                print(f"❌ Failed reading entry: {inner_e}")
                results.append({
                    "machine_id": "unknown",
                    "metrics": "error",
                    "detail": str(inner_e)
                })

        return {"status": "success", "machines": results}

    except Exception as e:
        print(f"/health failed: {repr(e)}")
        return {"status": "error", "detail": str(e)}

@app.get("/ticket")
async def get_ticket():
    return {"ticket": ticket}

@app.post("/start_job")
async def start_job():
    global doc, node
    author = await node.authors().create()
    content = b"Process this payload"
    await doc.set_bytes(author, b"job_trigger", content)
    return {"status": "triggered"}

@app.get("/result")
async def get_final_result():
    try:
        query = iroh.Query.all(None)
        entries = await doc.get_many(query)

        for entry in reversed(entries):  # Look from latest to oldest
            if entry.key().decode() == "final_result":
                content = await node.blobs().read_to_bytes(entry.content_hash())
                return {"status": "success", "result": content.decode()}

        return {"status": "waiting", "detail": "No final result found yet."}

    except Exception as e:
        return {"status": "error", "detail": str(e)}
