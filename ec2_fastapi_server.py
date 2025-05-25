import asyncio
import iroh
from fastapi import FastAPI
from iroh.iroh_ffi import uniffi_set_event_loop
import psutil
app = FastAPI()

node = None
doc = None
ticket = None  # Will be generated on startup

@app.on_event("startup")
async def startup():
    global node, doc, ticket

    print("🚀 Starting EC2 FastAPI server with Iroh shared document...")
    uniffi_set_event_loop(asyncio.get_running_loop())

    # Initialize Iroh node
    options = iroh.NodeOptions()
    options.enable_docs = True
    node = await iroh.Iroh.memory_with_options(options)

    # Create new shared doc and share ticket
    doc = await node.docs().create()
    ticket = await doc.share(iroh.ShareMode.WRITE, iroh.AddrInfoOptions.RELAY_AND_ADDRESSES)
    peer_id = await node.net().node_id() 
    key = peer_id.encode()
    print("Created key", key)
    cpu = psutil.cpu_percent(interval=1)
    ram = psutil.virtual_memory().percent
    value = f"CPU: {cpu}%\nRAM: {ram}%".encode()
    print("Created value", value)
    author = await node.authors().create()
    try:
        await doc.set_bytes(author, key, value)
        print(f"Sent metrics as {peer_id}")
    except Exception as e:
        print(f"Failed to send metrics: {e}")
    # Step 2: Immediately join it to activate participation
    doc = await node.docs().join(ticket)


    print("EC2 Iroh node started")
    print("SHARE THIS TICKET WITH ALL INTERNAL MACHINES:")
    print(f"\n{ticket}\n")

@app.get("/health")
async def health():
    try:
        query = iroh.Query.all(None)
        entries = await doc.get_many(query)

        results = []
        for entry in entries:
            key = entry.key().decode()
            content = await node.blobs().read_to_bytes(entry.content_hash())
            results.append({
                "machine_id": key,
                "metrics": content.decode()
            })

        return {"status": "success", "machines": results}
    except Exception as e:
        return {"status": "error", "detail": str(e)}

@app.get("/ticket")
async def get_ticket():
    return {"ticket": ticket}
