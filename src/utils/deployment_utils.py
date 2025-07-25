from pathlib import Path
import json
from fastapi import HTTPException
from src.utils.db_utils import get_active_peers, get_peer_metrics
from src.utils.model_utils import distribute_layers_across_peers

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

def create_distribution_plan(metadata, peers_vram):
            config = {
                "num_hidden_layers": metadata['num_layers'],
                "hidden_size": metadata['hidden_size'],
                "vocab_size": 32000,  # Default for demo
                "num_attention_heads": 32,  # Default for demo  
                "num_key_value_heads": 32,  # Default for demo
                "intermediate_size": metadata['hidden_size'] * 4,  # Standard ratio
            }
            
            distribution_plan = distribute_layers_across_peers(
                config=config,
                peers_vram=peers_vram,
                q_bits=32  # Default quantization, for testing for now
            )
            print(f"📋 Distribution plan created:")
            print(f"   • Model can fit: {distribution_plan['can_fit_model']}")
            print(f"   • Total VRAM needed: {distribution_plan['model_info']['total_model_vram_gb']:.1f}GB")
            print(f"   • Available VRAM: {distribution_plan['total_available_vram_gb']:.1f}GB")
            print(f"   • Peers involved: {len(distribution_plan['distribution'])}")
            for peer_id, peer_info in distribution_plan['distribution'].items():
                print(f"   • {peer_id}: {peer_info['assigned_layers']} layers, {peer_info['estimated_vram_usage']:.1f}GB")

            if not distribution_plan["can_fit_model"]:
                raise HTTPException(
                    status_code=400,
                    detail=f"Model cannot fit in available VRAM. Need {distribution_plan['model_info']['total_model_vram_gb']:.1f}GB, have {distribution_plan['total_available_vram_gb']:.1f}GB"
                )
            return distribution_plan

# The distribution_plan returned by distribute_layers_across_peers looks like this:
#
# {
#     "distribution": {
#         "peer_id_1": {
#             "assigned_layers": int,            # Number of layers assigned to this peer
#             "handles_embeddings": bool,        # Whether this peer loads the embedding layer
#             "available_vram_gb": float,        # VRAM available to this peer (GB)
#             "estimated_vram_usage": float,     # Estimated VRAM used by this peer (GB)
#             "vram_utilization_percent": float  # % of VRAM utilized by assigned layers
#         },
#         "peer_id_2": { ... },
#         ...
#     },
#     "model_info": {
#         "total_layers": int,                   # Total number of model layers
#         "total_assigned_layers": int,          # Total layers assigned across all peers
#         "vram_per_layer_gb": float,            # VRAM required per layer (GB)
#         "embedding_vram_gb": float,            # VRAM required for embeddings (GB)
#         "total_model_vram_gb": float           # Total VRAM required for the full model (GB)
#     },
#     "can_fit_model": bool,                     # True if the model fits in available VRAM
#     "remaining_layers": int,                   # Layers left unassigned (should be 0 if can_fit_model)
#     "total_peers": int,                        # Number of peers considered
#     "utilized_peers": int,                     # Number of peers actually assigned layers
#     "total_available_vram_gb": float           # Total VRAM available across all peers (GB)
# }


def create_deployment_instructions(request, distribution_plan, peer_table, SERVER_IP):
            """
            Create deployment instructions for each peer based on the distribution plan.
            """
            deployment_instructions = {}
            peer_list = list(distribution_plan["distribution"].keys())
            
            for i, (peer_id, peer_info) in enumerate(distribution_plan["distribution"].items()):
                is_first_peer = (i == 0)
                is_last_peer = (i == len(peer_list) - 1)
                
                assigned_layers = list(range(
                    sum(p["assigned_layers"] for p in list(distribution_plan["distribution"].values())[:i]),
                    sum(p["assigned_layers"] for p in list(distribution_plan["distribution"].values())[:i+1])
                ))

                # Determine required files based on peer position
                required_files = []
                
                # Config files (all peers need these)
                required_files.extend([
                    "config/config.json",
                    "config/tokenizer.json", 
                    "config/tokenizer_config.json",
                ])
                
                # Essential components based on position
                if is_first_peer:
                    required_files.append("embedding/layer.safetensors")
                    
                if is_last_peer:
                    required_files.extend([
                        "lm_head/layer.safetensors",
                        "norm/layer.safetensors"
                    ])
                
                # Assigned layer files
                for layer_idx in assigned_layers:
                    required_files.append(f"layers/layer_{layer_idx}.safetensors")
                
                # Determine the next peer's ID and address (if any)
                next_peer_ticket = peer_list[i + 1] if (i + 1) < len(peer_list) else None

                deployment_instructions[peer_id] = {
                    "model_name": request.model_name,
                    "assigned_layers": assigned_layers,
                    "is_first_peer": is_first_peer,
                    "is_last_peer": is_last_peer,
                    "required_files": required_files,
                    "server_download_url": f"http://{SERVER_IP}:8000/download_file/{request.model_name.replace('/', '_')}", #TODO: change to server ip
                    "vram_allocation": peer_info,
                    "next_peer_ticket": next_peer_ticket,
                    "pipeline": peer_list
                }
            return deployment_instructions