#!/usr/bin/env python3
"""
Demo script for safetensors-based layer sharding.
This approach processes safetensors files directly without loading the entire model into memory.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.utils.sharding_utils import (
    shard_model_by_layers_safetensors,
    get_model_safetensors_files,
    extract_layer_weights_from_safetensors,
)


def demo_safetensors_sharding():
    """Demonstrate safetensors-based sharding."""

    # Example model (use a smaller model for demo)
    model_name = (
        "meta-llama/Meta-Llama-3.1-8B-Instruct"  # Much smaller than Llama-3.1-8B
    )
    hf_token = None  # Set your token if needed

    print("🔪 Safetensors-Based Layer Sharding Demo")
    print("=" * 50)
    print(f"Model: {model_name}")
    print()

    # Step 1: List safetensors files
    print("📋 Step 1: Listing safetensors files...")
    safetensors_files = get_model_safetensors_files(model_name, hf_token)
    print(f"Found {len(safetensors_files)} safetensors files:")
    for file in safetensors_files:
        print(f"  - {file}")
    print()

    # Step 2: Extract a single layer as example
    print("🔍 Step 2: Extracting layer 0 weights...")
    try:
        layer_weights = extract_layer_weights_from_safetensors(
            model_name, layer_idx=0, hf_token=hf_token
        )
        print(f"Extracted {len(layer_weights)} weights for layer 0:")
        for key in layer_weights.keys():
            shape = layer_weights[key].shape
            print(f"  - {key}: {shape}")
        print()
    except Exception as e:
        print(f"❌ Error extracting layer 0: {e}")
        print()

    # Step 3: Full sharding (commented out to avoid long execution)
    print("🚀 Step 3: Full model sharding (commented out for demo)")
    print("To run full sharding, uncomment the code below:")
    print()

    # Uncomment to run full sharding
    output_dir = f"./shards/{model_name.replace('/', '_')}"
    result = shard_model_by_layers_safetensors(
        model_name=model_name, output_dir=output_dir, hf_token=hf_token
    )
    print(f"✅ Sharding completed: {result['total_components']} components")
    print(f"📁 Output: {result['output_dir']}")


def demo_memory_efficiency():
    """Demonstrate memory efficiency of safetensors approach."""

    print("💾 Memory Efficiency Comparison")
    print("=" * 40)
    print()
    print("Traditional Approach:")
    print("  ❌ Loads entire model into memory")
    print("  ❌ Requires 2x model size in RAM")
    print("  ❌ OOM crashes with large models")
    print()
    print("Safetensors Approach:")
    print("  ✅ Processes files one at a time")
    print("  ✅ Extracts only needed tensors")
    print("  ✅ Minimal memory footprint")
    print("  ✅ Works with any model size")
    print()
    print("How it works:")
    print("  1. Download safetensors files individually")
    print("  2. Load each file and extract specific tensors")
    print("  3. Fuse weights (QKV, gate_up) for vLLM compatibility")
    print("  4. Save individual layer files")
    print("  5. Clean up memory after each file")
    print()


if __name__ == "__main__":
    demo_memory_efficiency()
    demo_safetensors_sharding()
