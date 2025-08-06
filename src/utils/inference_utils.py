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


def create_virtual_sampler_output(request_id: str, step_idx: int) -> Any:
    """
    Create a virtual/mock SamplerOutput that satisfies vLLM's expectations
    for intermediate peers that don't actually need the real sampler state.
    
    This is a lightweight placeholder that contains minimal required fields.
    """
    try:
        from vllm.model_executor.layers.sampler import SamplerOutput
        from vllm.outputs import CompletionSequenceGroupOutput, SequenceOutput
        from vllm.sequence import Logprob
        import torch
        
        # Create minimal mock SequenceOutput
        mock_sequence_output = SequenceOutput(
            parent_seq_id=0,
            output_token=1,  # Dummy token ID
            logprobs=None,  # No logprobs needed for intermediate peers
        )
        
        # Create minimal mock CompletionSequenceGroupOutput
        mock_completion_output = CompletionSequenceGroupOutput(
            samples=[mock_sequence_output],
            prompt_logprobs=None,
        )
        
        # Create virtual SamplerOutput with minimal required fields
        virtual_sampler_output = SamplerOutput(
            outputs=[mock_completion_output],
            sampled_token_probs=None,  # Not needed for intermediate peers
            logprobs=None,  # Not needed for intermediate peers  
            sampled_token_ids=None,  # Not needed for intermediate peers
            sampled_token_ids_cpu=None,
            sampled_token_embeds=None,
            spec_decode_worker_metrics=None,
            hidden_states=None,
            prefill_hidden_states=None,
            model_forward_time=None,
            model_execute_time=None,
        )
        
        print(f"🎭 Created virtual SamplerOutput for intermediate peer (request: {request_id}, step: {step_idx})")
        return virtual_sampler_output
        
    except Exception as e:
        print(f"❌ Failed to create virtual SamplerOutput: {e}")
        # Fallback: return None and let the system handle it
        return None


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

    def pre_hook(module, args):
        """Ultra-minimal pre-hook for maximum performance"""
        # Get request-specific context with minimal overhead
        with context_lock:
            active_contexts = [ctx for ctx in hook_contexts.values() if ctx.get("active", False)]
            if not active_contexts:
                return args
            
            hook_context = active_contexts[0]
            request_id = hook_context["request_id"]
            current_step = hook_context["current_step"]
            
            # Skip ALL checks if first peer
            if hook_context["is_first_peer"]:
                return args
        
        # Wait for data (unavoidable, but optimized)
        event = STEP_EVENTS[request_id].setdefault(current_step, threading.Event())
        if not event.wait(timeout=30.0):
            cleanup_request_context(request_id)
            return args
        
        # Direct memory access (minimal locking)
        with CONTEXT_LOCK:
            step_data = INFERENCE_CONTEXT[request_id][str(current_step)]
            hidden_states = step_data["hidden_state"]
            residual = step_data["residual"]
        
        positions = args[0]
        device = positions.device
        
        # Single conditional for step - ultra optimized reshaping
        if current_step:  # Decode phase
            # Pre-computed shapes for decode (single token)
            # Ensure tensors are on correct device
            hidden_reshaped = hidden_states.view(1, 1, 2048).to(device, non_blocking=True)
            residual_reshaped = residual.view(1, 1, 2048).to(device, non_blocking=True)
            positions_reshaped = positions.view(1, 1).to(device, non_blocking=True) if positions.numel() == 1 else positions.view(1, -1)[:, -1:].to(device, non_blocking=True)
            
            # Clean up old data immediately
            if current_step > 1:
                INFERENCE_CONTEXT[request_id].pop(str(current_step - 2), None)
                STEP_EVENTS[request_id].pop(current_step - 2, None)
                
            return (positions_reshaped, hidden_reshaped, residual_reshaped)
        else:  # Prompt phase
            seq_len = hidden_states.shape[0]  # sequence length
            # Reshape with minimal operations
            hidden_reshaped = hidden_states.view(1, seq_len, 2048).to(device, non_blocking=True)
            residual_reshaped = residual.view(1, seq_len, 2048).to(device, non_blocking=True)
            
            # Handle positions efficiently
            if positions.dim() == 1:
                positions = positions.unsqueeze(0)
            positions_reshaped = positions[:, -seq_len:] if positions.shape[1] >= seq_len else positions
            positions_reshaped = positions_reshaped.to(device, non_blocking=True)
            
            return (positions_reshaped, hidden_reshaped, residual_reshaped)

    def post_hook(module, args, output):
        """Ultra-minimal post-hook for maximum performance"""
        # Get context with minimal overhead
        with context_lock:
            active_contexts = [ctx for ctx in hook_contexts.values() if ctx.get("active", False)]
            if not active_contexts:
                return
            
            hook_context = active_contexts[0]
            
            # Skip if last peer (no sending needed)
            if hook_context["is_last_peer"]:
                return
                
            request_id = hook_context["request_id"]
            current_step = hook_context["current_step"]
            
            # Fast duplicate check
            context_key = f"sent_step_{current_step}"
            if hook_context.get(context_key, False):
                return
            hook_context[context_key] = True
        
        hidden_states, residual = output
        
        # Single slice operation for decode (no validation)
        if current_step > 0:
            # Ultra-fast slicing for single token
            if hidden_states.dim() == 3 and hidden_states.shape[1] > 1:
                hidden_states = hidden_states[:, -1:, :]
                residual = residual[:, -1:, :]
            elif hidden_states.dim() == 2 and hidden_states.shape[0] > 1:
                hidden_states = hidden_states[-1:, :]
                residual = residual[-1:, :]
        
        # Direct tensor sending (skip CPU conversion if possible)
        next_peer_id = hook_context["next_peer_id"]
        next_peer_ticket = hook_context["next_peer_ticket"]
        
        # Async send with minimal conversion
        asyncio.run_coroutine_threadsafe(
            send_inference_tensors_fast(
                node,
                request_id,
                next_peer_id,
                hidden_states,  # Send torch tensor directly
                residual,  # Send torch tensor directly
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
            
        if is_last_peer:
            # Serialize the entire SamplerOutput object
            sampler_output_bytes = pickle.dumps(output)
            sampler_output_np = np.frombuffer(sampler_output_bytes, dtype=np.uint8)
            
            # 🚀 OPTIMIZATION: Only send to FIRST peer instead of all peers
            # Find the first peer in the pipeline
            first_peer_ticket = pipeline[0] if pipeline else None
            
            if first_peer_ticket and first_peer_ticket != peer_id:
                print(f"📤 Sending sampler output ONLY to FIRST peer: {first_peer_ticket[:8]}...")
                asyncio.run_coroutine_threadsafe(
                    send_sampler_output(
                        node,
                        request_id,
                        first_peer_ticket,
                        sampler_output_np,
                        step_idx=current_step,
                        next_peer_ticket=first_peer_ticket
                    ),
                    main_loop
                )
                print(f"✅ Optimized: Sent sampler output only to FIRST peer (not {len(pipeline)-1} peers)")
            else:
                print(f"⚠️ No valid first peer found or first peer is self")
            
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
            # Non-last peer: Use virtual sampler output instead of waiting
            print(f"🎭 Non-last peer using VIRTUAL sampler output for step {current_step}")
            
            # Determine if this peer is the first peer (needs real sampler state)
            peer_position = pipeline.index(peer_id) if peer_id in pipeline else -1
            is_first_peer_in_pipeline = (peer_position == 0)
            
            if is_first_peer_in_pipeline:
                # FIRST peer: Wait for real sampler output from last peer
                print(f"⏳ FIRST peer waiting for REAL sampler output for step {current_step}")
                
                # Wait for sampler output using threading.Event
                event = STEP_EVENTS[request_id].setdefault(current_step, threading.Event())
                
                if not event.wait(timeout=30.0):
                    cleanup_request_context(request_id)
                    raise RuntimeError(f"Timeout waiting for sampler output for {request_id} step {current_step}")
                
                # Data is ready!
                with CONTEXT_LOCK:
                    received_output = INFERENCE_CONTEXT[request_id][str(current_step)]["sampler_output"]
                    print(f"✅ FIRST peer received REAL sampler output for step {current_step}")
                
                # Clean up consumed data to prevent memory growth
                with CONTEXT_LOCK:
                    if current_step > 0:
                        INFERENCE_CONTEXT[request_id].pop(str(current_step - 1), None)
                        STEP_EVENTS[request_id].pop(current_step - 1, None)
                
                # Increment step after receiving
                with context_lock:
                    hook_context["current_step"] = current_step + 1
                    
                return received_output
                
            else:
                # INTERMEDIATE peer: Use virtual sampler output (no waiting!)
                print(f"🚀 INTERMEDIATE peer ({peer_position}) using VIRTUAL sampler output - NO WAITING!")
                
                # Create virtual sampler output
                virtual_output = create_virtual_sampler_output(request_id, current_step)
                
                if virtual_output is None:
                    print(f"⚠️ Failed to create virtual output, falling back to waiting...")
                    # Fallback to old behavior if virtual creation fails
                    event = STEP_EVENTS[request_id].setdefault(current_step, threading.Event())
                    if not event.wait(timeout=30.0):
                        cleanup_request_context(request_id)
                        raise RuntimeError(f"Timeout waiting for sampler output for {request_id} step {current_step}")
                    
                    with CONTEXT_LOCK:
                        received_output = INFERENCE_CONTEXT[request_id][str(current_step)]["sampler_output"]
                        
                    with context_lock:
                        hook_context["current_step"] = current_step + 1
                        
                    return received_output
                
                # Clean up old data to prevent memory growth
                with CONTEXT_LOCK:
                    if current_step > 0:
                        INFERENCE_CONTEXT[request_id].pop(str(current_step - 1), None)
                        STEP_EVENTS[request_id].pop(current_step - 1, None)
                
                # Increment step immediately (no waiting!)
                with context_lock:
                    hook_context["current_step"] = current_step + 1
                    
                print(f"✅ INTERMEDIATE peer bypassed sampler wait - performance optimized!")
                return virtual_output


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


async def send_inference_tensors_fast(
    tensor_transport: "TensorTransport",
    request_id: str,
    next_peer_id: str,
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    step_idx: int = 0,
    next_peer_ticket: str = "",
):
    """
    Ultra-fast tensor sending with minimal conversion overhead.
    Accepts torch tensors directly and does lazy conversion only when needed.
    """
    try:
        if not next_peer_ticket:
            raise ValueError("next_peer_ticket must be provided")

        # Convert to numpy with minimal overhead
        # Use .detach() to avoid autograd overhead, .cpu() only if needed
        if hidden_states.is_cuda:
            hidden_np = hidden_states.detach().cpu().numpy()
            residual_np = residual.detach().cpu().numpy()
        else:
            hidden_np = hidden_states.detach().numpy()
            residual_np = residual.detach().numpy()
        
        # Stack efficiently
        combined_tensor = np.concatenate([hidden_np.reshape(1, *hidden_np.shape), residual_np.reshape(1, *residual_np.shape)], axis=0)
        
        # Fast send
        await tensor_transport.send(
            next_peer_ticket,
            name=f"{request_id}_step{step_idx}_combined",
            tensor=combined_tensor
        )

    except Exception as e:
        # Minimal error handling - no printing in hot path
        pass


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
