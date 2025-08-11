"""
Minimal implementation for dynamically loading only selected pre-sharded layers.
This replicates Prime Intellect's approach for selective layer loading.

Usage:
python selective_layer_loader.py --layers 0,1,2 --prompt "Hello world"
"""

import json
import argparse
import os
from pathlib import Path
from typing import List, Dict, Any
import time


# FORCE vLLM v0 mode (required for selective layer loading)
os.environ["VLLM_USE_V1"] = "0"
print("🔧 FORCED vLLM v0 mode (VLLM_USE_V1=0) for selective layer loading compatibility")

import torch
from safetensors.torch import load_file
from vllm import LLM, SamplingParams
from vllm.config import ModelConfig, CacheConfig, LoadConfig, VllmConfig


# Global hook counter for tracking forward passes
hook_call_count = 0


def create_hidden_state_hooks(layer_idx: int, layer_name: str):
    """Create hooks to visualize hidden states passing through real layers."""
    
    def pre_hook(module, input):
        """Hook called BEFORE layer forward pass - shows input hidden states."""
        global hook_call_count
        hook_call_count += 1
        
        print(f"\n🔍 [{hook_call_count:02d}] PRE-HOOK: {layer_name} (Layer {layer_idx})")
        
        try:
            # For transformer layers, input is typically (positions, hidden_states, residual)
            if len(input) >= 2:
                positions = input[0]
                hidden_states = input[1]
                if hasattr(positions, 'shape'):
                    print(f"   📍 Positions shape: {positions.shape}")
                    print(f"   📍 Positions: {positions}")
                if hasattr(hidden_states, 'shape'):
                    print(f"   🧠 Hidden states shape: {hidden_states.shape}")
                    print(f"   🧠 Hidden states dtype: {hidden_states.dtype}")
                    print(f"   🧠 Hidden states device: {hidden_states.device}")
                    print(f"   🔥 RAW INPUT HIDDEN STATES (first 10 values): {hidden_states.flatten()[:10]}")
                
                if len(input) >= 3 and input[2] is not None:
                    residual = input[2]
                    if hasattr(residual, 'shape'):
                        print(f"   🔄 Residual shape: {residual.shape}")
                        print(f"   🔥 RAW INPUT RESIDUAL (first 10 values): {residual.flatten()[:10]}")
        except Exception as e:
            print(f"   ⚠️ Error in pre-hook: {e}")
        
        return None  # Don't modify input
    
    def post_hook(module, input, output):
        """Hook called AFTER layer forward pass - shows output hidden states."""
        print(f"   📤 OUTPUT from {layer_name}:")
        
        try:
            if isinstance(output, tuple) and len(output) >= 2:
                hidden_states, residual = output[0], output[1]
                if hasattr(hidden_states, 'shape'):
                    print(f"   🧠 Output hidden states shape: {hidden_states.shape}")
                    print(f"   🧠 Output hidden states dtype: {hidden_states.dtype}")
                    print(f"   🔥 RAW OUTPUT HIDDEN STATES (first 10 values): {hidden_states.flatten()[:10]}")
                if residual is not None and hasattr(residual, 'shape'):
                    print(f"   🔄 Output residual shape: {residual.shape}")
                    print(f"   🔥 RAW OUTPUT RESIDUAL (first 10 values): {residual.flatten()[:10]}")
            else:
                print(f"   📤 Output type: {type(output)}")
                if hasattr(output, 'shape'):
                    print(f"   📤 Output shape: {output.shape}")
                    print(f"   🔥 RAW OUTPUT (first 10 values): {output.flatten()[:10]}")
        except Exception as e:
            print(f"   ⚠️ Error in post-hook: {e}")
        
        print(f"   ✅ {layer_name} processing complete!")
        return None  # Don't modify output
    
    return pre_hook, post_hook


def create_embedding_hooks():
    """Create hooks for embedding layer."""
    
    def embedding_hook(module, input, output):
        print(f"\n🎯 EMBEDDING LAYER:")
        print(f"   📥 Input (token_ids) shape: {input[0].shape}")
        print(f"   📥 Input (token_ids): {input[0]}")
        print(f"   📤 Output (embeddings) shape: {output.shape}")
        print(f"   🧠 Embedding dtype: {output.dtype}")
        print(f"   🔥 RAW EMBEDDINGS (first 10 values): {output.flatten()[:10]}")
        return None
    
    return embedding_hook


def create_norm_hooks():
    """Create hooks for norm layer."""
    
    def norm_hook(module, input, output):
        print(f"\n🎚️  NORM LAYER:")
        print(f"   📥 Input shape: {input[0].shape}")
        
        # Handle tuple output (norm layer returns (hidden_states, residual))
        if isinstance(output, tuple):
            hidden_states, residual = output
            print(f"   📤 Output hidden_states shape: {hidden_states.shape}")
            print(f"   📤 Output residual shape: {residual.shape if residual is not None else 'None'}")
            print(f"   🔥 RAW NORM OUTPUT (first 10 values):")
            print(f"      Hidden states: {hidden_states.flatten()[:10]}")
            if residual is not None:
                print(f"      Residual: {residual.flatten()[:10]}")
        else:
            print(f"   📤 Output shape: {output.shape}")
            print(f"   🔥 RAW NORM OUTPUT (first 10 values): {output.flatten()[:10]}")
        return None
    
    return norm_hook


def create_lm_head_hooks():
    """Create hooks for lm_head layer."""
    
    def lm_head_hook(module, input, output):
        print(f"\n🎯 LM_HEAD LAYER:")
        print(f"   📥 Input shape: {input[0].shape}")
        print(f"   🔥 RAW LM_HEAD INPUT (first 10 values): {input[0].flatten()[:10]}")
        print(f"   📤 Output (logits) shape: {output.shape}")
        print(f"   🧠 Logits dtype: {output.dtype}")
        print(f"   🔥 RAW LOGITS (first 10 values): {output.flatten()[:10]}")
        return None
    
    return lm_head_hook


def create_ppmissing_hooks(layer_idx: int):
    """Create hooks to visualize data flowing through PPMissingLayer (passthrough)."""
    
    def ppmissing_pre_hook(module, input):
        """Hook called BEFORE PPMissingLayer - should be passthrough."""
        print(f"\n⭕ PPMissingLayer-{layer_idx} PRE-HOOK (PASSTHROUGH):")
        
        try:
            if len(input) >= 2:
                positions = input[0]
                hidden_states = input[1]
                if hasattr(positions, 'shape'):
                    print(f"   📍 Input positions: {positions}")
                if hasattr(hidden_states, 'shape'):
                    print(f"   🧠 Input hidden states shape: {hidden_states.shape}")
                    print(f"   🔥 RAW INPUT (first 10): {hidden_states.flatten()[:10]}")
                    print(f"   💡 This should be UNCHANGED from previous layer output!")
                
                if len(input) >= 3 and input[2] is not None:
                    residual = input[2]
                    if hasattr(residual, 'shape'):
                        print(f"   🔄 Input residual shape: {residual.shape}")
        except Exception as e:
            print(f"   ⚠️ Error in PPMissingLayer pre-hook: {e}")
        
        return None
    
    def ppmissing_post_hook(module, input, output):
        """Hook called AFTER PPMissingLayer - should be identical to input."""
        print(f"   📤 PPMissingLayer-{layer_idx} OUTPUT:")
        
        try:
            if isinstance(output, tuple) and len(output) >= 2:
                hidden_states, residual = output[0], output[1]
                if hasattr(hidden_states, 'shape'):
                    print(f"   🧠 Output hidden states shape: {hidden_states.shape}")
                    print(f"   🔥 RAW OUTPUT (first 10): {hidden_states.flatten()[:10]}")
                    print(f"   ✅ Should be IDENTICAL to input (passthrough)")
                if residual is not None and hasattr(residual, 'shape'):
                    print(f"   🔄 Output residual shape: {residual.shape}")
            else:
                if hasattr(output, 'shape'):
                    print(f"   📤 Output shape: {output.shape}")
                    print(f"   🔥 RAW OUTPUT (first 10): {output.flatten()[:10]}")
                elif hasattr(output, '__len__'):
                    print(f"   📤 Output (tuple/list): {output}")
                else:
                    print(f"   📤 Output type: {type(output)}")
        except Exception as e:
            print(f"   ⚠️ Error in PPMissingLayer post-hook: {e}")
        
        print(f"   ⭕ PPMissingLayer-{layer_idx} passthrough complete!")
        return None
    
    return ppmissing_pre_hook, ppmissing_post_hook


def add_visualization_hooks(llm: LLM, assigned_layers: List[int]):
    """Add hooks to visualize hidden states flowing through the model."""
    global hook_call_count
    hook_call_count = 0
    
    print("\n🔧 ADDING VISUALIZATION HOOKS...")
    
    # Access the model through vLLM's internal structure (v0 compatible)
    try:
        # Try v0 path first
        model_runner = llm.llm_engine.model_executor.driver_worker.model_runner
        model = model_runner.model
        print("   ✅ Using vLLM v0 model access path")
    except AttributeError:
        try:
            # Try v1 path (compatibility layer)
            model_runner = llm.llm_engine.engine_core.model_executor.driver_worker.model_runner
            model = model_runner.model
            print("   ✅ Using vLLM v1 model access path")
        except AttributeError:
            # Alternative v1 path
            model_runner = llm.llm_engine.model_executor.driver_worker.model_runner
            model = model_runner.model
            print("   ✅ Using alternative vLLM model access path")
    
    # Add hooks to embedding layer
    if hasattr(model.model, 'embed_tokens'):
        embedding_hook = create_embedding_hooks()
        model.model.embed_tokens.register_forward_hook(embedding_hook)
        print(f"   ✅ Added hook to embedding layer")
    
    # Add hooks to both real transformer layers AND PPMissingLayers
    real_layer_count = 0
    missing_layer_count = 0
    for i, layer in enumerate(model.model.layers):
        # Check if this is a real layer (not PPMissingLayer)
        if not layer.__class__.__name__ == 'PPMissingLayer':
            if i in assigned_layers:
                pre_hook, post_hook = create_hidden_state_hooks(i, f"Layer-{i}")
                layer.register_forward_pre_hook(pre_hook)
                layer.register_forward_hook(post_hook)
                real_layer_count += 1
                print(f"   ✅ Added hooks to Layer {i} (real transformer layer)")
            else:
                print(f"   ⚠️  Layer {i} is real but not in assigned_layers: {assigned_layers}")
        else:
            # This is a PPMissingLayer - add passthrough visualization hooks
            ppmissing_pre_hook, ppmissing_post_hook = create_ppmissing_hooks(i)
            layer.register_forward_pre_hook(ppmissing_pre_hook)
            layer.register_forward_hook(ppmissing_post_hook)
            missing_layer_count += 1
            print(f"   ⭕ Added hooks to Layer {i} (PPMissingLayer passthrough)")
    
    # Add hooks to norm layer
    if hasattr(model.model, 'norm') and not model.model.norm.__class__.__name__ == 'PPMissingLayer':
        norm_hook = create_norm_hooks()
        model.model.norm.register_forward_hook(norm_hook)
        print(f"   ✅ Added hook to norm layer")
    
    # Add hooks to lm_head
    if hasattr(model, 'lm_head'):
        lm_head_hook = create_lm_head_hooks()
        model.lm_head.register_forward_hook(lm_head_hook)
        print(f"   ✅ Added hook to lm_head layer")
    
    print(f"\n🎯 HOOK SUMMARY:")
    print(f"   📊 Total real layers hooked: {real_layer_count}")
    print(f"   ⭕ Total PPMissingLayers hooked: {missing_layer_count}")
    print(f"   📊 Assigned layers: {assigned_layers}")
    print(f"   🔍 Ready to visualize hidden state flow!")
    print(f"   💡 Expected: Real layers process data, PPMissingLayers pass it through unchanged!")
    
    return llm


class SelectiveLayerLoader:
    """Loads only specific layers from pre-sharded model files."""
    
    def __init__(self, shard_dir: str):
        self.shard_dir = Path(shard_dir)
        
        # Load metadata
        with open(self.shard_dir / "layer_metadata.json", "r") as f:
            self.metadata = json.load(f)
        
        print(f"Loaded metadata for {self.metadata['model_name']} with {self.metadata['num_layers']} layers")
    
    def load_weights_for_layers(self, layer_indices: List[int]) -> Dict[str, torch.Tensor]:
        """Load weights only for specified layers."""
        all_weights = {}
        
        # Load embedding (always needed)
        embed_path = self.shard_dir / "embedding" / "layer.safetensors"
        if embed_path.exists():
            embed_weights = load_file(str(embed_path))
            all_weights.update(embed_weights)
            print(f"Loaded embedding weights")
        
        # Load lm_head (always needed for generation)
        lm_head_path = self.shard_dir / "lm_head" / "layer.safetensors"
        if lm_head_path.exists():
            lm_head_weights = load_file(str(lm_head_path))
            all_weights.update(lm_head_weights)
            print(f"Loaded lm_head weights")
        
        # Load model.norm (always needed)
        norm_path = self.shard_dir / "norm" / "layer.safetensors"
        if norm_path.exists():
            norm_weights = load_file(str(norm_path))
            all_weights.update(norm_weights)
            print(f"Loaded model.norm weights")
        
        # Load only specified transformer layers
        for layer_idx in layer_indices:
            layer_path = self.shard_dir / "layers" / f"layer_{layer_idx}.safetensors"
            if layer_path.exists():
                layer_weights = load_file(str(layer_path))
                all_weights.update(layer_weights)
                print(f"Loaded layer {layer_idx} weights")
            else:
                print(f"Warning: Layer {layer_idx} not found at {layer_path}")
        
        return all_weights


def create_dynamic_vllm_model(shard_dir: str, assigned_layers: List[int]) -> LLM:
    """Create vLLM model with only assigned layers loaded by monkey-patching make_layers."""
    
    # STEP 1: Monkey-patch vLLM's make_layers function (Prime Intellect's key insight)
    def _selective_make_layers(num_hidden_layers: int, layer_fn, prefix: str):
        """Custom make_layers that creates real layers only for assigned indices."""
        from vllm.model_executor.models.utils import PPMissingLayer, maybe_offload_to_cpu
        
        start_layer = min(assigned_layers) if assigned_layers else 0
        end_layer = max(assigned_layers) + 1 if assigned_layers else 0
        
        modules = []
        for i in range(num_hidden_layers):
            if i in assigned_layers:
                # Create real layer
                layer = layer_fn(prefix=f"{prefix}.{i}")
                modules.append(maybe_offload_to_cpu(layer))
                print(f"  Created REAL layer {i}")
            else:
                # Create passthrough layer (Prime Intellect's memory optimization)
                modules.append(PPMissingLayer())
                print(f"  Created PPMissingLayer for layer {i}")
        
        return start_layer, end_layer, torch.nn.ModuleList(modules)
    
    # Apply the monkey patch
    import vllm.model_executor.models.utils as model_utils
    original_make_layers = model_utils.make_layers
    model_utils.make_layers = _selective_make_layers
    
    try:
        # STEP 2: Create vLLM model (will use our patched make_layers)
        config_dir = str(Path(shard_dir) / "config")
        
        llm = LLM(
            model=config_dir,
            tensor_parallel_size=1,
            enforce_eager=True,  # Required for custom layer loading
            max_model_len=512,   # Small for demo
            disable_log_stats=True,
            skip_tokenizer_init=False,
            gpu_memory_utilization=0.6,  # Use much less memory
            # disable_log_requests=True,
            # Force legacy engine to avoid v1 memory pre-allocation
            use_v2_block_manager=False,
            load_format="dummy"    # ← this is the magic flag

        )
        
        # STEP 3: Load weights for assigned layers + essential components
        loader = SelectiveLayerLoader(shard_dir)
        weights = loader.load_weights_for_layers(assigned_layers)
        
        print(f"\n🔍 DEBUGGING WEIGHT LOADING:")
        print(f"   Loaded weight keys (first 10): {list(weights.keys())[:10]}")
        for key, tensor in list(weights.items())[:3]:
            print(f"   {key}: shape={tensor.shape}, sample_values={tensor.flatten()[:5]}")
        
        # STEP 4: Actually apply the loaded weights to the model!
        print(f"\n🔧 APPLYING LOADED WEIGHTS TO MODEL...")
        model_runner = llm.llm_engine.model_executor.driver_worker.model_runner
        model = model_runner.model
        
        # Apply weights to the model
        applied_count = 0
        for name, param in model.named_parameters():
            if name in weights:
                with torch.no_grad():
                    param.copy_(weights[name].to(param.dtype))
                    applied_count += 1
                    # print(f"   ✅ Applied {name}: {param.shape}") # Optional: uncomment for verbose logging
            # else:
                # print(f"   ⚠️  Parameter {name} not found in loaded weights")
        
        print(f"✅ Applied weights to {applied_count} parameters")
        
        print(f"\n✅ Successfully created vLLM model with selective layers!")
        print(f"   Our monkey-patch created real layers for: {assigned_layers}")
        print(f"   All other layers are PPMissingLayer (passthrough)")
        print(f"   Loaded weights for {len(weights)} parameters")
        print(f"   Applied {applied_count}/{len(list(model.named_parameters()))} parameters to model")
        
        return llm
        
    finally:
        # Restore original function
        model_utils.make_layers = original_make_layers


def create_full_vllm_model(model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0") -> LLM:
    """Create full vLLM model for comparison."""
    import time
    import psutil
    import torch
    
    print(f"\n🔥 Creating FULL model: {model_name}")
    start_time = time.time()
    start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    
    llm = LLM(
        model=model_name,
        tensor_parallel_size=1,
        enforce_eager=True,
        max_model_len=512,
        disable_log_stats=True,
        gpu_memory_utilization=0.8,  # Use more memory for full model
        use_v2_block_manager=False
    )
    
    end_time = time.time()
    end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    
    print(f"📊 FULL MODEL STATS:")
    print(f"   Loading time: {end_time - start_time:.2f} seconds")
    print(f"   GPU memory used: {(end_memory - start_memory) / 1024**3:.2f} GB")
    print(f"   All 22 layers loaded")
    
    return llm


def demonstrate_selective_loading_with_visualization(shard_dir: str, layers: List[int], prompt: str):
    """Demonstrate loading only specific layers and visualize hidden states during inference."""
    import time
    import torch
    
    print(f"\n⚡ Creating SELECTIVE model with ONLY layers {layers}...")
    start_time = time.time()
    start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    
    llm = create_dynamic_vllm_model(shard_dir, layers)
    
    # Add visualization hooks
    # add_visualization_hooks(llm, layers)
    
    end_time = time.time()
    end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    
    print(f"\n📊 SELECTIVE MODEL STATS:")
    print(f"   Loading time: {end_time - start_time:.2f} seconds")
    print(f"   GPU memory used: {(end_memory - start_memory) / 1024**3:.2f} GB")
    print(f"   Only {len(layers)} out of 22 layers loaded ({len(layers)/22*100:.1f}%)")
    
    print(f"\n" + "="*80)
    print(f"🚀 RUNNING INFERENCE WITH HIDDEN STATE VISUALIZATION")
    print(f"   Prompt: '{prompt}'")
    print(f"   Watch the hidden states flow through layers {layers}!")
    print(f"="*80)
    
    # sampling_params = SamplingParams(temperature=0.0, max_tokens=5)  # Short for demo
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=50)
    
    outputs = llm.generate([prompt], sampling_params)
    
    print(f"\n" + "="*80)
    print(f"🎯 INFERENCE COMPLETE!")
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"   Generated: '{generated_text}'")
    print(f"="*80)
    
    return llm


def compare_full_vs_selective(shard_dir: str, layers: List[int], prompt: str):
    """Compare full model vs selective loading performance."""
    print("=" * 80)
    print("🎯 COMPARISON: FULL MODEL vs SELECTIVE LAYER LOADING")
    print("=" * 80)
    
    # Test 1: Full model
    try:
        full_llm = create_full_vllm_model()
        
        print(f"\n🔥 Running inference on FULL model...")
        sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=50)
        
        start_time = time.time()
        outputs = full_llm.generate([prompt], sampling_params)
        end_time = time.time()
        
        full_output = outputs[0].outputs[0].text
        full_inference_time = end_time - start_time
        
        print(f"Full model output: '{full_output}'")
        print(f"Full model inference time: {full_inference_time:.3f} seconds")
        
    except Exception as e:
        print(f"❌ Full model failed: {e}")
        full_llm = None
        full_output = "FAILED"
        full_inference_time = 0
    
    print("\n" + "-" * 80)
    
    # Test 2: Selective model
    selective_llm = demonstrate_selective_loading_with_visualization(shard_dir, layers, prompt)
    
    print(f"\n⚡ Running inference on SELECTIVE model...")
    start_time = time.time()
    outputs = selective_llm.generate([prompt], sampling_params)
    end_time = time.time()
    
    selective_output = outputs[0].outputs[0].text
    selective_inference_time = end_time - start_time
    
    print(f"Selective model output: '{selective_output}'")
    print(f"Selective model inference time: {selective_inference_time:.3f} seconds")
    
    # Summary comparison
    print("\n" + "=" * 80)
    print("📊 PERFORMANCE COMPARISON SUMMARY")
    print("=" * 80)
    
    if full_llm:
        print(f"🔥 FULL MODEL (22 layers):")
        print(f"   ✓ All layers loaded")
        print(f"   ✓ Complete model functionality")
        print(f"   ✗ High memory usage")
        print(f"   ✗ Slower loading")
    else:
        print(f"🔥 FULL MODEL: ❌ FAILED TO LOAD")
    
    print(f"\n⚡ SELECTIVE MODEL ({len(layers)} layers):")
    print(f"   ✓ {100 * (22 - len(layers)) / 22:.1f}% memory savings")
    print(f"   ✓ Faster loading")
    print(f"   ✓ Distributed inference ready")
    print(f"   ~ Partial model functionality")
    
    print(f"\n🎯 KEY INSIGHT:")
    print(f"   Prime Intellect's approach enables {len(layers)} layers per GPU")
    print(f"   instead of requiring full 22-layer model per GPU!")
    print(f"   This allows consumer GPUs to participate in inference!")
    
    return selective_llm, full_llm

def main():
    parser = argparse.ArgumentParser(description="Demonstrate selective layer loading")
    parser.add_argument("--shard-dir", type=str, default="../shards/meta-llama_Llama-3.2-1B-Instruct", 
                       help="Directory containing pre-sharded layers")
    parser.add_argument("--layers", type=str, default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15",
                       help="Comma-separated list of layer indices to load (e.g., '0,1,2')")
    parser.add_argument("--prompt", type=str, default="Hello, my name is",
                       help="Text prompt for inference")
    parser.add_argument("--compare", action="store_true",
                       help="Compare with full model loading")
    parser.add_argument("--visualize", action="store_true",
                       help="Add visualization hooks to see hidden states")
    
    args = parser.parse_args()
    
    # Parse layer indices
    assigned_layers = [int(x.strip()) for x in args.layers.split(",")]
    
    if args.compare:
        # Run full comparison
        compare_full_vs_selective(args.shard_dir, assigned_layers, args.prompt)
    elif args.visualize:
        # Run with visualization hooks
        print("🎯 RUNNING SELECTIVE LOADING WITH HIDDEN STATE VISUALIZATION!")
        print(f"   Layers: {assigned_layers}")
        print(f"   Prompt: '{args.prompt}'")
        demonstrate_selective_loading_with_visualization(args.shard_dir, assigned_layers, args.prompt)
    else:
        # Just run selective loading
        demonstrate_selective_loading_with_visualization(args.shard_dir, assigned_layers, args.prompt)


def demo_hidden_state_extraction():
    """Simple demo function to show hidden state extraction in action."""
    print("🚀 DEMO: Hidden State Extraction with Selective Layer Loading")
    print("="*80)
    
    # Demo with first 3 layers
    layers = [0, 1, 2]
    prompt = "The quick brown fox"
    
    print(f"📋 Demo Configuration:")
    print(f"   Loading only layers: {layers}")
    print(f"   Prompt: '{prompt}'")
    print(f"   This will show hidden states flowing through each real layer!")
    print(f"   PPMissingLayers (placeholders) will be bypassed silently.")
    print("="*80)
    
    try:
        llm = demonstrate_selective_loading_with_visualization("./layer_shards", layers, prompt)
        print("\n✅ Demo completed successfully!")
        print("   You can see how hidden states flow through only the loaded layers.")
        print("   The other 19 layers are PPMissingLayer placeholders.")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("   Make sure you have run layer_sharding.py first to create the shards.")


if __name__ == "__main__":
    # You can also run the demo directly
    print("🔧 To run hidden state visualization, use one of these commands:")
    print("   python selective_layer_loader.py --layers 0,1,2 --prompt 'Hello' --visualize")
    print("   python -c \"from selective_layer_loader import demo_hidden_state_extraction; demo_hidden_state_extraction()\"")
    
    main()


# ============================================================================
# STANDALONE DEMO FUNCTIONS (you can call these directly)
# ============================================================================

def test_hidden_state_extraction_simple():
    """Simple test function you can call directly to see hidden states."""
    print("🚀 TESTING HIDDEN STATE EXTRACTION")
    print("="*60)
    
    # Ensure v0 mode
    os.environ["VLLM_USE_V1"] = "0"
    
    # Test configuration
    layers = [0, 1, 2]  # First 3 layers
    prompt = "Hello world"
    shard_dir = "./layer_shards"
    
    print(f"📋 Configuration:")
    print(f"   Layers to load: {layers}")
    print(f"   Prompt: '{prompt}'")
    print(f"   Shard directory: {shard_dir}")
    print("="*60)
    
    try:
        # Create the selective model with visualization
        llm = demonstrate_selective_loading_with_visualization(shard_dir, layers, prompt)
        print("\n✅ SUCCESS! Hidden state extraction completed.")
        return llm
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("   Make sure you've run layer_sharding.py first to create the shards.")
        return None


def force_v0_mode():
    """Helper function to explicitly force vLLM v0 mode."""
    import os
    os.environ["VLLM_USE_V1"] = "0"
    print("🔧 Forced VLLM_USE_V1=0 (v0 mode)")
    
    # Verify it's set
    import vllm.envs as envs
    print(f"   Current VLLM_USE_V1 setting: {envs.VLLM_USE_V1}")


# Example usage functions
def run_basic_demo():
    """Run a basic demo without hooks for testing."""
    force_v0_mode()
    
    try:
        llm = create_dynamic_vllm_model("./layer_shards", [0, 1, 2])
        print("✅ Basic selective loading works!")
        
        # sampling_params = SamplingParams(temperature=0.0, max_tokens=5)
        sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=50)
        outputs = llm.generate(["Hello"], sampling_params)
        print(f"Generated: {outputs[0].outputs[0].text}")
        
    except Exception as e:
        print(f"❌ Basic demo failed: {e}")


def run_visualization_demo():
    """Run the full visualization demo."""
    force_v0_mode()
    return test_hidden_state_extraction_simple() 