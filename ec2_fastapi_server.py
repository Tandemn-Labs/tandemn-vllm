# ec2_fastapi_server.py
import asyncio
from fastapi import FastAPI

app = FastAPI()

# TODO - Replace these with real peer IDs of your internal machines
peer_ids = [
    "12D3KooWRvM1a...",
    "12D3KooWEbs82...",
    "12D3KooWLq5Vc...",
]

@app.on_event("startup")
async def startup_event():
    print("Starting EC2 FastAPI Server")

@app.get("/health")
async def check_all_machines():
    return await get_all_metrics(peer_ids)

async def get_all_metrics(peers):
    async def fetch(peer_id):
        return {"peer_id": peer_id, "status": "ok", "metrics": "metrics"}

    return await asyncio.gather(*(fetch(pid) for pid in peers))
