from pathlib import Path
import json
from fastapi import HTTPException
from src.server import get_active_peers, get_peer_metrics

def load_model_metadata(shard_folder: str):
    shard_path = Path(shard_folder)
    metadata_file = shard_path / "layer_metadata.json"
    if not metadata_file.exists() or not shard_path.exists():
        raise HTTPException(status_code=404, detail=f"layer_metadata.json or shard folder not found: {shard_folder}")
    with open(metadata_file, "r") as f:
        metadata = json.load(f)
    print(f"📊 Model metadata: {metadata['num_layers']} layers, type: {metadata['model_type']}")
    return metadata

async def get_peers_with_vram():
    active_peers = await get_active_peers()
    if len(active_peers) < 1:
        raise HTTPException(status_code=400, detail="No active peers available for deployment")
    
    peers_vram = {}
    for peer_id in active_peers:
        try:
            metrics_history = await get_peer_metrics(peer_id, time_window=60)
            if metrics_history:
                latest_metrics = metrics_history[0]["metrics"]
                if "total_free_vram_gb" in latest_metrics:
                    peers_vram[peer_id] = latest_metrics["total_free_vram_gb"]
        except Exception as e:
            print(f"❌ Error getting metrics for peer {peer_id}: {e}")

    if not peers_vram:
        raise HTTPException(status_code=400, detail="No peers with VRAM information available")
    
    print(f"👥 Found {len(peers_vram)} peers with VRAM data")
    return peers_vram