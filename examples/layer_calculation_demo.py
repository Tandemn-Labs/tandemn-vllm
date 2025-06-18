#!/usr/bin/env python3
"""
Demo script showing how to calculate layer distribution across GPU peers.

This script demonstrates:
1. How to calculate VRAM per layer for a model
2. How many layers can fit in a specific peer
3. How to distribute layers optimally across multiple peers
"""

import asyncio
import json
from src.utils.model_utils import (
    download_config,
    estimate_layer_vram,
    calculate_max_layers_for_peer,
    distribute_layers_across_peers
)

async def demo_layer_calculations():
    """Demonstrate layer calculation functionality"""
    print("🚀 Layer Calculation Demo")
    print("=" * 50)
    
    # Example: Llama-2-7B model
    model_id = "meta-llama/Llama-2-70b-hf"
    qbits = 16  # FP16 quantization
    
    try:
        # 1. Download model configuration
        print(f"📥 Downloading config for {model_id}...")
        config = await download_config(model_id)
        print(f"✅ Model has {config.get('num_hidden_layers')} layers")
        
        # 2. Calculate VRAM per layer
        print(f"\n🔢 Calculating VRAM requirements ({qbits}-bit)...")
        vram_per_layer, embedding_vram, total_vram = estimate_layer_vram(config, qbits)
        print(f"   VRAM per layer: {vram_per_layer:.3f} GB")
        print(f"   Embedding VRAM: {embedding_vram:.3f} GB")
        print(f"   Total model VRAM: {total_vram:.3f} GB")
        
        # 3. Example peer VRAM capacities (simulate different GPU peers)
        example_peers = {
            "peer_rtx_4090": 22.0,    # RTX 4090 with ~22GB free VRAM
            "peer_rtx_3080": 8.0,     # RTX 3080 with ~8GB free VRAM  
            "peer_rtx_3060": 10.0,    # RTX 3060 Ti with ~10GB free VRAM
            "peer_a100": 78.0,        # A100 with ~78GB free VRAM
        }
        
        print(f"\n🖥️  Example GPU Peers:")
        for peer_id, vram in example_peers.items():
            print(f"   {peer_id}: {vram} GB free VRAM")
        
        # 4. Calculate capacity for each peer individually
        print(f"\n📊 Individual Peer Capacities:")
        for peer_id, available_vram in example_peers.items():
            calculation = calculate_max_layers_for_peer(config, available_vram, qbits)
            print(f"\n   {peer_id} ({available_vram} GB free):")
            print(f"      Max layers (layers only): {calculation['max_layers_only']}")
            print(f"      Max layers (with embeddings): {calculation['max_layers_with_embeddings']}")
            print(f"      VRAM utilization: {calculation['vram_used_layers_only']:.2f} GB")
        
        # 5. Create optimal distribution plan
        print(f"\n🎯 Optimal Distribution Plan:")
        distribution = distribute_layers_across_peers(config, example_peers, qbits)
        
        print(f"   Can fit entire model: {distribution['can_fit_model']}")
        print(f"   Total layers to distribute: {distribution['model_info']['total_layers']}")
        print(f"   Assigned layers: {distribution['model_info']['total_assigned_layers']}")
        print(f"   Utilized peers: {distribution['utilized_peers']}/{distribution['total_peers']}")
        
        print(f"\n   Distribution Details:")
        for peer_id, allocation in distribution['distribution'].items():
            handles_emb = "✅" if allocation['handles_embeddings'] else "❌"
            print(f"      {peer_id}:")
            print(f"         Assigned layers: {allocation['assigned_layers']}")
            print(f"         Handles embeddings: {handles_emb}")
            print(f"         VRAM usage: {allocation['estimated_vram_usage']} GB")
            print(f"         Utilization: {allocation['vram_utilization_percent']}%")
        
        # 6. Show what happens with insufficient VRAM
        print(f"\n⚠️  Example: Insufficient VRAM scenario")
        small_peers = {
            "peer_small_1": 2.0,  # Only 2GB free VRAM
            "peer_small_2": 3.0,  # Only 3GB free VRAM
        }
        
        small_distribution = distribute_layers_across_peers(config, small_peers, qbits)
        print(f"   Can fit model with small peers: {small_distribution['can_fit_model']}")
        print(f"   Remaining unassigned layers: {small_distribution['remaining_layers']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in demo: {e}")
        return False

def demo_api_usage():
    """Show example API calls for the new endpoints"""
    print(f"\n📡 API Usage Examples:")
    print("=" * 30)
    
    print("1. Calculate layers for specific peer:")
    print("""
    POST /calculate_peer_layers
    {
        "model_id": "meta-llama/Llama-2-7b-hf",
        "hf_token": "your_token",
        "qbits": 16,
        "peer_id": "peer_rtx_4090",
        "available_vram_gb": 22.0
    }
    """)
    
    print("2. Get current peer capacity:")
    print("""
    GET /peer_layer_capacity/peer_rtx_4090?model_id=meta-llama/Llama-2-7b-hf&qbits=16
    """)
    
    print("3. Create distribution plan:")
    print("""
    POST /distribute_model_layers
    {
        "model_id": "meta-llama/Llama-2-7b-hf", 
        "hf_token": "your_token",
        "qbits": 16
    }
    """)

if __name__ == "__main__":
    print("🎮 Starting Layer Calculation Demo")
    
    # Run the async demo
    success = asyncio.run(demo_layer_calculations())
    
    if success:
        # Show API usage examples
        demo_api_usage()
        print("\n✅ Demo completed successfully!")
    else:
        print("\n❌ Demo failed!") 