import asyncio
import iroh
import pickle
import httpx  # type: ignore
import json
from typing import Dict, Any, List, Optional
import time
from vllm import LLM, SamplingParams, TokensPrompt  # type: ignore
import torch
import threading
import uuid
from src.utils.tensor_protocol_adapter import TensorTransport
import numpy as np
from collections import defaultdict

# ADD PROFILING IMPORTS AND GLOBALS
import contextlib
from dataclasses import dataclass, field
from typing import DefaultDict

@dataclass
class ProfileMetrics:
    """Store timing metrics for different pipeline components"""
    pre_hook_wait_times: List[float] = field(default_factory=list)
    pre_hook_processing_times: List[float] = field(default_factory=list)
    post_hook_processing_times: List[float] = field(default_factory=list)
    tensor_send_times: List[float] = field(default_factory=list)
    sampler_hook_times: List[float] = field(default_factory=list)
    vllm_generate_times: List[float] = field(default_factory=list)
    layer_forward_times: List[float] = field(default_factory=list)
    tensor_manipulation_times: List[float] = field(default_factory=list)
    total_step_times: List[float] = field(default_factory=list)

# Global profiling storage per request
PROFILE_METRICS: DefaultDict[str, ProfileMetrics] = defaultdict(ProfileMetrics)

@contextlib.contextmanager
def profile_timer(metric_name: str, request_id: str = "default"):
    """Context manager for timing code blocks"""
    start_time = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start_time
        metrics = PROFILE_METRICS[request_id]
        
        if metric_name == "pre_hook_wait":
            metrics.pre_hook_wait_times.append(elapsed)
        elif metric_name == "pre_hook_processing":
            metrics.pre_hook_processing_times.append(elapsed)
        elif metric_name == "post_hook_processing":
            metrics.post_hook_processing_times.append(elapsed)
        elif metric_name == "tensor_send":
            metrics.tensor_send_times.append(elapsed)
        elif metric_name == "sampler_hook":
            metrics.sampler_hook_times.append(elapsed)
        elif metric_name == "vllm_generate":
            metrics.vllm_generate_times.append(elapsed)
        elif metric_name == "layer_forward":
            metrics.layer_forward_times.append(elapsed)
        elif metric_name == "tensor_manipulation":
            metrics.tensor_manipulation_times.append(elapsed)
        elif metric_name == "total_step":
            metrics.total_step_times.append(elapsed)
        
        print(f"⏱️ [{request_id}] {metric_name}: {elapsed*1000:.2f}ms")

def print_profile_summary(request_id: str):
    """Print comprehensive timing summary for a request"""
    metrics = PROFILE_METRICS.get(request_id)
    if not metrics:
        print(f"❌ No profile metrics found for {request_id}")
        return
    
    print(f"\n{'='*80}")
    print(f"📊 PERFORMANCE PROFILE SUMMARY - {request_id}")
    print(f"{'='*80}")
    
    def stats(times: List[float], name: str):
        if not times:
            return f"{name}: No data"
        avg = sum(times) / len(times) * 1000
        max_time = max(times) * 1000
        min_time = min(times) * 1000
        total = sum(times) * 1000
        return f"{name:25}: avg={avg:7.2f}ms max={max_time:7.2f}ms min={min_time:7.2f}ms total={total:8.2f}ms count={len(times)}"
    
    print(stats(metrics.vllm_generate_times, "🚀 vLLM Generate"))
    print(stats(metrics.pre_hook_wait_times, "⏳ Pre-hook Wait"))
    print(stats(metrics.pre_hook_processing_times, "🔄 Pre-hook Processing"))
    print(stats(metrics.post_hook_processing_times, "📤 Post-hook Processing"))
    print(stats(metrics.tensor_send_times, "📡 Tensor Send"))
    print(stats(metrics.sampler_hook_times, "🎯 Sampler Hook"))
    print(stats(metrics.layer_forward_times, "🧠 Layer Forward"))
    print(stats(metrics.tensor_manipulation_times, "🔧 Tensor Manipulation"))
    print(stats(metrics.total_step_times, "📊 Total Step"))
    
    # Calculate bottlenecks
    bottlenecks = []
    if metrics.vllm_generate_times:
        bottlenecks.append(("vLLM Generate", sum(metrics.vllm_generate_times)))
    if metrics.pre_hook_wait_times:
        bottlenecks.append(("Pre-hook Wait", sum(metrics.pre_hook_wait_times)))
    if metrics.tensor_send_times:
        bottlenecks.append(("Tensor Send", sum(metrics.tensor_send_times)))
    if metrics.pre_hook_processing_times:
        bottlenecks.append(("Pre-hook Processing", sum(metrics.pre_hook_processing_times)))
    
    bottlenecks.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n🔥 TOP BOTTLENECKS:")
    for i, (name, total_time) in enumerate(bottlenecks[:5], 1):
        percentage = (total_time / sum(t for _, t in bottlenecks)) * 100 if bottlenecks else 0
        print(f"   {i}. {name}: {total_time*1000:.2f}ms ({percentage:.1f}%)")
    
    print(f"{'='*80}\n")

def cleanup_profile_metrics(request_id: str):
    """Clean up profiling data for completed request"""
    if request_id in PROFILE_METRICS:
        del PROFILE_METRICS[request_id]
        print(f"🧹 Cleaned up profile metrics for {request_id}")

# This global dictionary holds the actual tensor data, not futures
# Key: request_id (str)
# Value: A dictionary with step-indexed hidden states and residuals
# Structure: {request_id: {step_idx: {"hidden_state": tensor, "residual": tensor}}}
INFERENCE_CONTEXT: Dict[str, Dict[str, Any]] = {}
# the above is just a payload that is sent from one peer to another. 
CONTEXT_LOCK = threading.RLock()  # Thread-safe access to INFERENCE_CONTEXT

# ------------------------------------------------------------------
#  Per–request / per-step Events that tell a waiting peer "data ready"
# ------------------------------------------------------------------
STEP_EVENTS: Dict[str, Dict[int, threading.Event]] = defaultdict(dict)


def cleanup_request_context(request_id: str):
    """Thread-safe cleanup of request context"""
    with CONTEXT_LOCK:
        if request_id in INFERENCE_CONTEXT:
            del INFERENCE_CONTEXT[request_id]
            print(f"🧹 Cleaned up context for {request_id}")
        if request_id in STEP_EVENTS:
            del STEP_EVENTS[request_id]
            print(f"🧹 Cleaned up step events for {request_id}")


async def send_final_result_to_server(
    request_id: str,
    output_obj,
    peer_id: str,
    server_url: str = "http://{SERVER_IP}:8000"):
    try:
        # vLLM ≥0.4 returns CompletionSequenceGroupOutput
        print(f"🔍 Output object type: {type(output_obj)}")
        if isinstance(output_obj, str):
            final_text = output_obj

        elif hasattr(output_obj, "sequences"):           # CompletionSequenceGroupOutput
            final_text = output_obj.sequences[0].text

        elif hasattr(output_obj, "outputs"):             # RequestOutput
            final_text = output_obj.outputs[0].text

        elif hasattr(output_obj, "text"):                # SequenceOutput (older)
            final_text = output_obj.text
        else:
            raise ValueError(f"Unknown output type: {type(output_obj)}")
        completion_data={
            "request_id": request_id,
            "output_text": final_text,
            "peer_id": peer_id,
            "timestamp": int(time.time())
        }
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                f"{server_url}/completion",
                json=completion_data
            )
            if response.status_code == 200:
                print(f"✅ Sent final result to server for request {request_id}")
            else:
                print(f"❌ Failed to send completion: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Error sending completion to server: {e}")


def register_inference_hooks(
    llm:"LLM",
    node: TensorTransport,
    peer_id: str,
    server_url: str = "http://{SERVER_IP}:8000",
    next_peer_ticket: Optional[str] = None,
    pipeline: Optional[List[str]] = None
):
    """
    Create pre and post hooks for the inference pipeline, to transfer hidden states
    """
    # get the model runner worker, model itself and the sampler
    model_runner = llm.llm_engine.model_executor.driver_worker.model_runner
    model=model_runner.model
    sampler=model_runner.sampler
    
    # Thread-safe hook context with locks
    hook_contexts = {}  # Per-request context storage
    context_lock = threading.RLock()

    def _slice_last_token(t: torch.Tensor) -> torch.Tensor:
        """Slices the last token from a tensor, handling multiple dimensions safely."""
        print(f"Dimensions of the tensor: {t.dim()}")
        if t.dim() == 2:
            # Shape (seq_len, hidden_dim) -> (1, hidden_dim)
            return t[-1:, :]
        elif t.dim() == 3:
            # Shape (batch, seq_len, hidden_dim) -> (batch, 1, hidden_dim)
            return t[:, -1:, :]
        elif t.dim() == 4:
            # Shape (batch, seq_len, num_heads, head_dim) -> (batch, 1, num_heads, head_dim)
            return t[:, -1:, :, :]
        else:
            print(f"⚠️ Unexpected tensor dimensions: {t.dim()}, returning unchanged")
            return t

    def _validate_tensor_shapes(positions: torch.Tensor, hidden_states: torch.Tensor, residual: torch.Tensor, step_idx: int) -> bool:
        """Validate that all tensors have compatible shapes for injection."""
        try:
            print(f"Dimensions of the hidden states: {hidden_states.dim()}")
            print(f"Dimensions of the residual: {residual.dim()}") 
            print(f"Dimensions of the positions: {positions.dim()}")
            # Basic dimension checks
            if hidden_states.dim() < 2 or residual.dim() < 2:
                print(f"❌ Hidden states or residual has insufficient dimensions: hidden={hidden_states.dim()}, residual={residual.dim()}")
                return False
            
            # Check that hidden states and residual have same shape
            if hidden_states.shape != residual.shape:
                print(f"❌ Hidden states and residual shape mismatch: hidden={hidden_states.shape}, residual={residual.shape}")
                return False
            
            # Extract sequence and batch dimensions
            if hidden_states.dim() == 2:
                seq_len, hidden_dim = hidden_states.shape
                batch_size = 1
            else:
                batch_size, seq_len, hidden_dim = hidden_states.shape[:3]
            
            # Check positions compatibility
            if positions.dim() == 1:
                pos_batch, pos_seq = 1, positions.shape[0]
            else:
                pos_batch, pos_seq = positions.shape[:2]
            
            # For step 0, allow full sequence. For step 1+, expect single token
            expected_seq_len = seq_len if step_idx == 0 else 1
            
            print(f"🔍 Shape validation for step {step_idx}:")
            print(f"   Hidden states: {hidden_states.shape} (expected seq_len: {expected_seq_len})")
            print(f"   Positions: {positions.shape}")
            print(f"   Batch size: {batch_size}, Seq len: {seq_len}")
            
            # Allow some flexibility but warn about potential issues
            if seq_len != expected_seq_len and step_idx > 0:
                print(f"⚠️ Unexpected sequence length for step {step_idx}: got {seq_len}, expected {expected_seq_len}")
            
            return True
            
        except Exception as e:
            print(f"❌ Shape validation failed: {e}")
            return False

    def pre_hook(module, args):
        """This hook is called BEFORE a layer does its forward pass"""
        step_start_time = time.perf_counter()
        
        # Get request-specific context safely
        with context_lock:
            # Find the current request context (there should be exactly one active)
            active_contexts = [ctx for ctx in hook_contexts.values() if ctx.get("active", False)]
            if not active_contexts:
                print("❌ No active inference context found in pre_hook")
                return args
            
            hook_context = active_contexts[0]  # Should be exactly one
            request_id = hook_context["request_id"]
            current_step = hook_context["current_step"]
        
        if not hook_context["is_first_peer"]:
            print(f"🔍 Detected a non-first peer, waiting for hidden state from previous peer (step {current_step})...")
            
            # TIME THE WAITING PERIOD
            with profile_timer("pre_hook_wait", request_id):
                # Wait for data to arrive for this step using threading.Event
                event = STEP_EVENTS[request_id].setdefault(current_step, threading.Event())
                
                if not event.wait(timeout=30.0):  # blocks without GIL churn
                    print(f"❌ Timeout waiting for hidden state for {request_id} step {current_step}")
                    cleanup_request_context(request_id)
                    return args
            
            # TIME THE TENSOR PROCESSING
            with profile_timer("pre_hook_processing", request_id):
                # Data is ready!
                with CONTEXT_LOCK:
                    step_data = INFERENCE_CONTEXT[request_id][str(current_step)]
                    hidden_states = step_data["hidden_state"]
                    residual = step_data["residual"]
                    print(f"✅ Received hidden state and residual for {request_id} (step {current_step}). Injecting into the next layer.")

                # Process the received tensors with improved shape handling
                orig_positions = args[0]
                
                print(f"🔍 Raw tensor shapes before processing (step {current_step}):")
                print(f"   Original positions: {orig_positions.shape}")
                print(f"   Received hidden states: {hidden_states.shape}")
                print(f"   Received residual: {residual.shape}")

                # MATRIX MANIPULATION STARTS HERE - IMPROVED VERSION
                with profile_timer("tensor_manipulation", request_id):
                    # 1. Validate received tensors
                    if not _validate_tensor_shapes(orig_positions, hidden_states, residual, current_step):
                        print(f"❌ Tensor validation failed for step {current_step}, using original args")
                        return args
                    
                    # 2. Ensure that the hidden states and residuals have batch dimension
                    if hidden_states.dim() == 2:
                        hidden_states = hidden_states.unsqueeze(0)
                        print(f"🔧 Added batch dimension to hidden states: {hidden_states.shape}")
                    if residual.dim() == 2:
                        residual = residual.unsqueeze(0)
                        print(f"🔧 Added batch dimension to residual: {residual.shape}")
                    
                    # 3. Ensure the positions have batch dimension
                    if orig_positions.dim() == 1:
                        orig_positions = orig_positions.unsqueeze(0)
                        print(f"🔧 Added batch dimension to positions: {orig_positions.shape}")
                    elif orig_positions.dim() >= 2:
                        print(f"🔧 Adding batch dimension to positions: {orig_positions.shape}")
                        orig_positions = orig_positions.contiguous()
                        print(f"🔧 Contiguous positions: {orig_positions.shape}")

                    # 4. Extract dimensions after normalization
                    batch_size, seq_len = hidden_states.shape[:2]
                    pos_batch_size, pos_seq_len = orig_positions.shape[:2]
                    
                    print(f"🔍 Normalized shapes:")
                    print(f"   Hidden states: {hidden_states.shape} (batch={batch_size}, seq={seq_len})")
                    print(f"   Positions: {orig_positions.shape} (batch={pos_batch_size}, seq={pos_seq_len})")
                    
                    # 5. Handle sequence length alignment - CRITICAL FIX
                    if current_step == 0:
                        # First step: Positions should match or be sliced to match hidden states
                        if pos_seq_len >= seq_len:
                            positions_to_inject = orig_positions[:, -seq_len:]
                            print(f"🔧 Sliced positions to match hidden states: {positions_to_inject.shape}")
                        else:
                            # Pad positions if needed (shouldn't happen normally)
                            padding_needed = seq_len - pos_seq_len
                            padding = torch.zeros(pos_batch_size, padding_needed, dtype=orig_positions.dtype, device=orig_positions.device)
                            positions_to_inject = torch.cat([padding, orig_positions], dim=1)
                            print(f"🔧 Padded positions to match hidden states: {positions_to_inject.shape}")
                    else:
                        # Step 1+: Hidden states should be single token, but positions might be longer
                        if seq_len == 1:
                            # Received single token as expected
                            positions_to_inject = orig_positions[:, -1:] if pos_seq_len > 1 else orig_positions
                            print(f"🔧 Using last position for single token: {positions_to_inject.shape}")
                        else:
                            # Unexpected: received full sequence in step 1+
                            print(f"⚠️ Step {current_step}: Received full sequence ({seq_len} tokens), expected single token")
                            # Take the last seq_len positions to match
                            positions_to_inject = orig_positions[:, -seq_len:]
                            print(f"🔧 Aligned positions to received sequence: {positions_to_inject.shape}")

                    # 6. Handle batch size mismatches
                    if positions_to_inject.shape[0] != batch_size:
                        if batch_size == 1:
                            positions_to_inject = positions_to_inject[:1]  # Take first batch
                            print(f"🔧 Reduced position batch size to match: {positions_to_inject.shape}")
                        else:
                            positions_to_inject = positions_to_inject.expand(batch_size, -1)
                            print(f"🔧 Expanded position batch size to match: {positions_to_inject.shape}")

                    # 7. Move to correct device - ensure all tensors are on the same device
                    target_device = orig_positions.device
                    positions_to_inject = positions_to_inject.to(target_device, non_blocking=True)
                    hidden_states_to_inject = hidden_states.to(target_device, non_blocking=True)
                    residual_to_inject = residual.to(target_device, non_blocking=True)

                    # 8. Final validation and device consistency check
                    assert positions_to_inject.device == hidden_states_to_inject.device == residual_to_inject.device, \
                        f"Device mismatch: pos={positions_to_inject.device}, hidden={hidden_states_to_inject.device}, residual={residual_to_inject.device}"
                    
                    # Final shape check
                    final_batch_size = positions_to_inject.shape[0]
                    final_seq_len = positions_to_inject.shape[1]
                    hidden_batch_size = hidden_states_to_inject.shape[0]
                    hidden_seq_len = hidden_states_to_inject.shape[1]
                    
                    if final_batch_size != hidden_batch_size or final_seq_len != hidden_seq_len:
                        print(f"❌ FINAL SHAPE MISMATCH:")
                        print(f"   Positions: batch={final_batch_size}, seq={final_seq_len}")
                        print(f"   Hidden states: batch={hidden_batch_size}, seq={hidden_seq_len}")
                        print(f"   Using original args to avoid crash")
                        return args
                    
                    print(f"✅ Final tensor shapes (step {current_step}):")
                    print(f"   Positions: {positions_to_inject.shape}")
                    print(f"   Hidden states: {hidden_states_to_inject.shape}")
                    print(f"   Residual: {residual_to_inject.shape}")

                # Clean up consumed data to prevent memory growth
                with CONTEXT_LOCK:
                    if current_step > 0:
                        INFERENCE_CONTEXT[request_id].pop(str(current_step - 1), None)
                        STEP_EVENTS[request_id].pop(current_step - 1, None)

                return (positions_to_inject, hidden_states_to_inject, residual_to_inject)
        
        # If this is the first peer, just return the original args
        return args

    def post_hook(module, args, output):
        """Runs after the layer forward pass - sends output to the next peer"""
        # Get request-specific context safely
        with context_lock:
            active_contexts = [ctx for ctx in hook_contexts.values() if ctx.get("active", False)]
            if not active_contexts:
                print("❌ No active inference context found in post_hook")
                return
            
            hook_context = active_contexts[0]
            request_id = hook_context["request_id"]
            current_step = hook_context["current_step"]
        
        # TIME THE ENTIRE POST-HOOK PROCESSING
        with profile_timer("post_hook_processing", request_id):
            # For the last peer, we don't need to send anything
            if hook_context["is_last_peer"]:
                return
                
            # Prevent multiple calls per inference step
            context_key = f"sent_step_{current_step}"
            print(f"🔍 Context key: {context_key}")
            if hook_context.get(context_key, False):
                return  # Already sent for this step
            hook_context[context_key] = True
        
        print(f"↪️ POST-HOOK for {request_id}, step {current_step}: Sending hidden states...")
        hidden_states, residual = output
        
        print(f"🔍 Output tensor shapes before processing (step {current_step}):")
        print(f"   Hidden states: {hidden_states.shape}")
        print(f"   Residual: {residual.shape}")
        
        next_peer_id = hook_context["next_peer_id"]
        next_peer_ticket = hook_context["next_peer_ticket"]

        # MATRIX MANIPULATION STARTS HERE - IMPROVED VERSION
        
        # CRITICAL FIX: More conservative and correct slicing strategy
        slice_needed = False
        if current_step > 0:
            if hidden_states.dim() == 2 and hidden_states.shape[0] > 1:
                # Case: (seq_len, hidden_dim)
                slice_needed = True
            elif hidden_states.dim() == 3 and hidden_states.shape[1] > 1:
                # Case: (batch, seq_len, hidden_dim)
                slice_needed = True
            elif hidden_states.dim() > 3 and hidden_states.shape[1] > 1:
                 # Case: (batch, seq_len, num_heads, head_dim) etc.
                 slice_needed = True
        
        if slice_needed:
            print(f"🔧 Slicing to last token for step {current_step}")
            hidden_states = _slice_last_token(hidden_states)
            residual = _slice_last_token(residual)
            print(f"🔧 After slicing - Hidden: {hidden_states.shape}, Residual: {residual.shape}")
        else:
            print(f"🔧 Step {current_step}: No slicing needed (full sequence or already single token).")
        
        # MATRIX MANIPULATION ENDS HERE

        # TIME THE TENSOR SENDING
        with profile_timer("tensor_send", request_id):
            # Send both tensors together in one message
            asyncio.run_coroutine_threadsafe(
                send_inference_tensors(
                    node,
                    request_id,
                    next_peer_id,
                    hidden_states.cpu().numpy(),
                    residual.cpu().numpy(),
                    step_idx=current_step,
                    next_peer_ticket=next_peer_ticket
                ),
                main_loop
            )
        
        # NOTE: Step increment moved to sampler_post_hook to ensure it happens on ALL peers
    
    # Capture the main asyncio loop so we can schedule coroutines from threads
    main_loop = asyncio.get_running_loop()

    def sampler_post_hook(module, args, output):
        """
        Post-hook for the sampler module.
        - For the last peer: broadcasts sampler output to all other peers
        - For non-last peers: waits for sampler output from the last peer
        """
        # Get request-specific context safely
        with context_lock:
            active_contexts = [ctx for ctx in hook_contexts.values() if ctx.get("active", False)]
            if not active_contexts:
                print("❌ No active inference context found in sampler_post_hook")
                return output
            
            hook_context = active_contexts[0]
            request_id = hook_context["request_id"]
            current_step = hook_context.get("current_step", 0)
            is_last_peer = hook_context.get("is_last_peer", False)
            pipeline = hook_context["pipeline"]
            peer_id = hook_context["peer_id"]
        
        # TIME THE SAMPLER HOOK PROCESSING
        with profile_timer("sampler_hook", request_id):
            if is_last_peer:
                # Serialize the entire SamplerOutput object
                sampler_output_bytes = pickle.dumps(output)
                sampler_output_np = np.frombuffer(sampler_output_bytes, dtype=np.uint8)
                
                # Send to all OTHER peers (not ourselves)
                for peer_ticket in pipeline:
                    if peer_ticket != peer_id:
                        asyncio.run_coroutine_threadsafe(
                            send_sampler_output(
                                node,
                                request_id,
                                peer_ticket,
                                sampler_output_np,
                                step_idx=current_step,
                                next_peer_ticket=peer_ticket
                            ),
                            main_loop
                        )
                
                print(f"📢 Last peer sent sampler output to all peers for step {current_step}")
                
                # Increment step for last peer too
                with context_lock:
                    hook_context["current_step"] = current_step + 1
                
                # Clean up old data to prevent memory growth
                with CONTEXT_LOCK:
                    if current_step > 0:
                        INFERENCE_CONTEXT[request_id].pop(str(current_step - 1), None)
                        STEP_EVENTS[request_id].pop(current_step - 1, None)
                    
                return output
            else:
                # Non-last peer: wait for sampler output
                print(f"⏳ Non-last peer waiting for sampler output for step {current_step}")
                
                # Wait for sampler output using threading.Event
                event = STEP_EVENTS[request_id].setdefault(current_step, threading.Event())
                
                if not event.wait(timeout=30.0):
                    cleanup_request_context(request_id)
                    raise RuntimeError(f"Timeout waiting for sampler output for {request_id} step {current_step}")
                
                # Data is ready!
                with CONTEXT_LOCK:
                    received_output = INFERENCE_CONTEXT[request_id][str(current_step)]["sampler_output"]
                    print(f"✅ Received sampler output for step {current_step}")
                
                # Clean up consumed data to prevent memory growth
                with CONTEXT_LOCK:
                    if current_step > 0:
                        INFERENCE_CONTEXT[request_id].pop(str(current_step - 1), None)
                        STEP_EVENTS[request_id].pop(current_step - 1, None)
                
                # Increment step after receiving
                with context_lock:
                    hook_context["current_step"] = current_step + 1
                    
                return received_output


    def start_inference_run(request_id: str, pipeline: List[str], input_text: str, sampling_params: Any, assigned_layers: Dict[str, List[int]]):
        """The main inference runner"""
        # Generate unique execution ID to avoid collisions
        execution_id = str(uuid.uuid4())[:8]
        
        try:
            # Determine this peer's position in the pipeline
            idx = pipeline.index(peer_id)
            is_first = (idx == 0)
            is_last = (idx == len(pipeline) - 1)
            
            # Determine next peer info if not last
            next_peer_id = None
            next_peer_ticket = None
            if not is_last:
                next_peer_id = pipeline[idx + 1]
                next_peer_ticket = pipeline[idx + 1]  # In this implementation, peer_id and ticket are the same
            
            # Initialize thread-safe context for this request
            with context_lock:
                hook_contexts[execution_id] = {
                    "request_id": request_id,
                    "pipeline": pipeline,
                    "input_text": input_text,
                    "is_first_peer": is_first,
                    "is_last_peer": is_last,
                    "peer_id": peer_id,
                    "next_peer_id": next_peer_id,
                    "next_peer_ticket": next_peer_ticket,
                    "current_step": 0,
                    "active": True
                }
            
            # Get this peer's assigned layers
            real_layers = [layer for layer in model.model.layers if "PPMissingLayer" not in layer.__class__.__name__]
            if not real_layers:
                print(f"⚠️ No real layers detected here. Cannot participate in this inference")
                return
            
            # Attach hooks to first and last real layers
            first_layer = real_layers[0]
            last_layer = real_layers[-1]
            print(f"✅ Dynamically attaching hooks to layers: {first_layer.__class__.__name__} -> {last_layer.__class__.__name__}")

            # Register hooks
            pre_hook_handle = first_layer.register_forward_pre_hook(pre_hook)
            post_hook_handle = last_layer.register_forward_hook(post_hook)
            sampler_hook_handle = sampler.register_forward_hook(sampler_post_hook)

            print("Starting the inference run...")
            # TIME THE MAIN vLLM GENERATE CALL
            with profile_timer("vllm_generate", request_id):
                # Run vLLM inference (this is the blocking call)
                completions = llm.generate([input_text], sampling_params=sampling_params)

            # If last peer, send final result to server
            if is_last and completions:
                comp = completions[0]
                final_text = comp.outputs[0].text
                try:
                    # Use a unique sent flag per request to avoid collisions
                    request_sent_key = f"final_sent_{request_id}"
                    with context_lock:
                        if not hook_contexts[execution_id].get(request_sent_key, False):
                            hook_contexts[execution_id][request_sent_key] = True
                            asyncio.run_coroutine_threadsafe(
                                send_final_result_to_server(request_id, final_text, peer_id, server_url),
                                main_loop,
                            )
                        print(f"🎯 Final result sent for {request_id}")
                except Exception as e:
                    print(f"❌ Failed to schedule send_final_result_to_server: {e}")

            # Clean up hooks
            pre_hook_handle.remove()
            post_hook_handle.remove()
            sampler_hook_handle.remove()
            print(f"🎉 Inference run completed for {request_id}")
            
            # PRINT PERFORMANCE PROFILE SUMMARY
            print_profile_summary(request_id)
            
        except Exception as e:
            print(f"❌ Error in inference run for {request_id}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Always clean up context and request data
            with context_lock:
                if execution_id in hook_contexts:
                    hook_contexts[execution_id]["active"] = False
                    del hook_contexts[execution_id]
            cleanup_request_context(request_id)
            # Clean up profiling data
            cleanup_profile_metrics(request_id)
        
        return
    
    # return the start_inference_run function
    return start_inference_run

async def send_hidden_state_tensor(
    tensor_transport: "TensorTransport",
    request_id: str,
    next_peer_id: str,
    data_to_send: "np.ndarray",
    is_residual: bool = False,
    step_idx: int = 0,
    next_peer_ticket: str = "",
):
    """
    Sends the tensor directly to the next peer using TensorTransport.
    No gossip, no blobs.
    """
    try:
        if not next_peer_ticket:
            raise ValueError("next_peer_ticket must be provided for tensor transport send.")

        # Compose a name for the tensor message
        tensor_name = f"{request_id}_step{step_idx}_{'residual' if is_residual else 'hidden_state'}"

        await tensor_transport.send(
            next_peer_ticket,
            name=tensor_name,
            tensor=data_to_send
        )

        print(f"📤 Sent {'residual' if is_residual else 'hidden_state'} tensor for {request_id} to {next_peer_id} ({next_peer_ticket}) via TensorTransport")
    except Exception as e:
        print(f"❌ [DEBUG] Failed to send hidden state tensor for {request_id} to {next_peer_id} ({next_peer_ticket}): {e}")


async def send_inference_tensors(
    tensor_transport: "TensorTransport",
    request_id: str,
    next_peer_id: str,
    hidden_states: "np.ndarray",
    residual: "np.ndarray",
    step_idx: int = 0,
    next_peer_ticket: str = "",
):
    """
    Sends both hidden states and residual tensors in a single message.
    Leverages tensor-iroh's built-in serialization.
    """
    try:
        if not next_peer_ticket:
            raise ValueError("next_peer_ticket must be provided for tensor transport send.")

        # Stack both tensors together - tensor-iroh handles serialization
        # Format: [hidden_states, residual] concatenated along a new dimension
        combined_tensor = np.stack([hidden_states, residual], axis=0)
        
        # Compose a name for the tensor message
        tensor_name = f"{request_id}_step{step_idx}_combined"

        await tensor_transport.send(
            next_peer_ticket,
            name=tensor_name,
            tensor=combined_tensor
        )

        print(f"📤 Sent combined tensors for {request_id} step {step_idx} to {next_peer_id} via TensorTransport")
    except Exception as e:
        print(f"❌ Failed to send tensors for {request_id} to {next_peer_id}: {e}")


async def send_sampler_output(
    tensor_transport: "TensorTransport",
    request_id: str,
    next_peer_id: str,
    sampler_output_bytes: "np.ndarray",
    step_idx: int = 0,
    next_peer_ticket: str = "",
):
    """
    Sends the pickled sampler output to the next peer.
    """
    try:
        if not next_peer_ticket:
            raise ValueError("next_peer_ticket must be provided for tensor transport send.")

        # Compose a name for the tensor message
        tensor_name = f"{request_id}_step{step_idx}_sampler_output"

        await tensor_transport.send(
            next_peer_ticket,
            name=tensor_name,
            tensor=sampler_output_bytes
        )

        print(f"📤 Sent sampler_output for {request_id} step {step_idx} to {next_peer_id[:8]}...")
    except Exception as e:
        print(f"❌ Failed to send sampler output for {request_id} to {next_peer_id}: {e}")
