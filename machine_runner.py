import asyncio
import psutil
import iroh
import tempfile
import os
import socket
from pathlib import Path
from iroh.iroh_ffi import uniffi_set_event_loop

SHARED_TICKET = "docaaacai6tzy2ictq7jsezznwysh62goidbnux47r2gjga7pfqh6q36g5bagffi6stycitrpx4t2zmyzo6rnfza7kgeinjbozy5uvni6xoholwoajdnb2hi4dthixs65ltmuys2mjoojswyylzfzuxe33ifzxgk5dxn5zgwlrpamaavqca7lf5sayaqj7p6sfs5mbabat675emxwid"
PIPELINE_FILE = Path("pipeline.txt")
TRIGGER_KEY = "job_trigger"
FINAL_RESULT_KEY = "final_result"

async def send_blob(doc, author, peer_id: str, data: bytes):
    try:
        await doc.set_bytes(author, peer_id.encode(), data)
        print(f"\U0001f4e4 Sent to {peer_id}: {data.decode()}")
    except Exception as e:
        print(f"❌ Failed to send to {peer_id}: {e}")

async def receive_blob(node, doc, peer_id: str):
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
                print(f"\U0001f4e5 Received for {peer_id}: {content.decode()}")
                return content
        except Exception as e:
            print(f"❌ Polling error for {peer_id}: {e}")
        await asyncio.sleep(2)

async def wait_for_trigger(node, doc):
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
                print(f"🆕 Trigger received: {content.decode()}")
                return content
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

def load_pipeline():
    if not PIPELINE_FILE.exists():
        print(f"❌ pipeline.txt not found at {PIPELINE_FILE.resolve()}")
        return []

    with open(PIPELINE_FILE, "r") as f:
        return [line.strip() for line in f.readlines() if line.strip()]

async def handle_job(node, doc, author, peer_id, next_peer, is_last):
    while True:
        data = await receive_blob(node, doc, peer_id)
        processed = f"{peer_id[:6]} processed → {data.decode()}".encode()
        if is_last:
            await doc.set_bytes(author, FINAL_RESULT_KEY.encode(), processed)
            print(f"✅ {peer_id[:6]} pushed result to EC2")
        else:
            await send_blob(doc, author, next_peer, processed)
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

    doc = await node.docs().join(iroh.DocTicket(str(SHARED_TICKET).strip()))
    author = await node.authors().create()

    await upload_metrics(doc, author, peer_id)

    print("⏳ Waiting for pipeline.txt to include me...")
    while True:
        pipeline = load_pipeline()
        if peer_id in pipeline:
            break
        await asyncio.sleep(2)

    index = pipeline.index(peer_id)
    is_first = index == 0
    is_last = index == len(pipeline) - 1
    next_peer = pipeline[index + 1] if not is_last else None

    print(f"✅ Pipeline position: {index} | First: {is_first} | Last: {is_last}")

    if is_first:
        while True:
            job = await wait_for_trigger(node, doc)
            result = f"{peer_id[:6]} started job: {job.decode()}".encode()
            await send_blob(doc, author, next_peer, result)
            print(f"🚀 Sent job to {next_peer[:6]}")
    else:
        await handle_job(node, doc, author, peer_id, next_peer, is_last)

if __name__ == "__main__":
    asyncio.run(main())
