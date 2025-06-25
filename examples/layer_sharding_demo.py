#!/usr/bin/env python3
"""
Demo script showing how to use the layer sharding functionality via the REST API.

This script demonstrates:
1. How to create layer shards for a model via the API
2. How the shards are organized on disk
3. How to access the metadata of the created shards
"""

import asyncio
import httpx
import json
from pathlib import Path

async def demo_layer_sharding():
    """Demonstrate the layer sharding API functionality"""
    print("🚀 Layer Sharding API Demo")
    print("=" * 50)
    
    # Configuration
    server_url = "http://localhost:8000"
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Small model for testing
    hf_token = "your_huggingface_token_here"  # Replace with your actual token
    
    # Request payload
    request_data = {
        "model_id": model_id,
        "hf_token": hf_token,
        "model_layers_key": "model.layers",
        "cache_dir": None
    }
    
    try:
        print(f"📥 Requesting layer sharding for {model_id}...")
        
        async with httpx.AsyncClient(timeout=3600.0) as client:  # Long timeout for model loading
            response = await client.post(
                f"{server_url}/create_layer_shards",
                json=request_data
            )
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Layer sharding completed successfully!")
                print(f"📁 Output directory: {result['output_directory']}")
                print(f"🔢 Total components: {result['total_components']}")
                print(f"📝 Message: {result['message']}")
                
                # Display metadata information
                metadata = result['metadata']
                print(f"\n📊 Model Information:")
                print(f"   Model name: {metadata['model_name']}")
                print(f"   Model type: {metadata['model_type']}")
                print(f"   Number of layers: {metadata['num_layers']}")
                print(f"   Hidden size: {metadata['hidden_size']}")
                
                print(f"\n📦 Layer Components:")
                for component in metadata['layer_components']:
                    if component['type'] == 'transformer_layer':
                        print(f"   Layer {component['layer_index']}: {component['path']}")
                    else:
                        print(f"   {component['type'].title()}: {component['path']}")
                
                # Check if files actually exist
                output_dir = Path(result['output_directory'])
                if output_dir.exists():
                    print(f"\n📂 Directory structure created:")
                    for item in sorted(output_dir.rglob("*")):
                        if item.is_file():
                            size_mb = item.stat().st_size / (1024 * 1024)
                            print(f"   {item.relative_to(output_dir)} ({size_mb:.1f} MB)")
                
                return True
                
            else:
                print(f"❌ Request failed with status {response.status_code}")
                print(f"Error: {response.text}")
                return False
                
    except httpx.TimeoutException:
        print("❌ Request timed out - model loading can take a while for large models")
        return False
    except Exception as e:
        print(f"❌ Error during layer sharding: {e}")
        return False

async def demo_check_shards_directory():
    """Check what's in the shards directory"""
    print(f"\n🔍 Checking shards directory contents:")
    shards_dir = Path("./shards")
    
    if shards_dir.exists():
        for model_dir in shards_dir.iterdir():
            if model_dir.is_dir():
                print(f"\n📁 {model_dir.name}:")
                
                # Check for metadata file
                metadata_file = model_dir / "layer_metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    print(f"   ✅ Metadata: {len(metadata['layer_components'])} components")
                
                # Count layer files
                layers_dir = model_dir / "layers"
                if layers_dir.exists():
                    layer_files = list(layers_dir.glob("layer_*.safetensors"))
                    print(f"   📦 Layer files: {len(layer_files)}")
                
                # Check other components
                for component_type in ['embedding', 'lm_head', 'norm', 'config']:
                    component_dir = model_dir / component_type
                    if component_dir.exists():
                        files = list(component_dir.iterdir())
                        print(f"   🔧 {component_type}: {len(files)} files")
                        
                        # Check for dummy model file in config
                        if component_type == 'config':
                            dummy_model = component_dir / "model.safetensors"
                            if dummy_model.exists():
                                size_mb = dummy_model.stat().st_size / (1024 * 1024)
                                print(f"      ✅ Dummy model.safetensors: {size_mb:.1f} MB (for vLLM compatibility)")
    else:
        print("   No shards directory found")

def main():
    """Main function to run the demo"""
    print("This demo shows how to use the layer sharding API.")
    print("Make sure the server is running at http://localhost:8000")
    print("And replace 'your_huggingface_token_here' with your actual HF token.\n")
    
    # Run the demo
    asyncio.run(demo_layer_sharding())
    
    # Check directory contents
    asyncio.run(demo_check_shards_directory())

if __name__ == "__main__":
    main() 