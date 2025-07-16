import asyncio
import iroh
import pickle
import httpx  # type: ignore
import json
from typing import Dict, Any, List
import time
from vllm import LLM, SamplingParams, TokensPrompt  # type: ignore
import torch
import threading
import uuid


# This global dictionary holds the actual tensor data, not futures
# Key: request_id (str)
# Value: A dictionary with step-indexed hidden states and residuals
# Structure: {request_id: {step_idx: {"hidden_state": tensor, "residual": tensor}}}
INFERENCE_CONTEXT: Dict[str, Dict[str, Any]] = {}
# the above is just a payload that is sent from one peer to another. 
CONTEXT_LOCK = threading.RLock()  # Thread-safe access to INFERENCE_CONTEXT


def cleanup_request_context(request_id: str):
    """Thread-safe cleanup of request context"""
    with CONTEXT_LOCK:
        if request_id in INFERENCE_CONTEXT:
            del INFERENCE_CONTEXT[request_id]
            print(f"🧹 Cleaned up context for {request_id}")


async def send_hidden_state_blob(
    node: iroh.Node,
    hidden_state_gossip_sink: iroh.Gossip,
    request_id: str,
    next_peer_id: str,
    pipeline: List[str],
    data_to_send: Any,
    is_residual: bool = False,
    step_idx: int = 0
):
    "Serializes the data, stores it as a blob, and sends it to the next peer"
    try:
        # serialize the data
        serialized_data = pickle.dumps(data_to_send)
        # add the data to a blob store
        blob_output = await node.blobs().add_bytes(serialized_data)
        blob_hash = blob_output.hash
        # create a blob ticket
        blob_ticket = await node.blobs().share(blob_hash, iroh.BlobFormat.RAW, iroh.AddrInfoOptions.RELAY_AND_ADDRESSES)

        # now that the data is added to the blob, create a pointer for it that is sent as a chatter over the gossip network
        reference={
            "request_id": request_id,
            "blob_ticket": str(blob_ticket),
            "tensor_shape": list(data_to_send.shape),
            "tensor_dtype": str(data_to_send.dtype),
            "next_peer_id": next_peer_id,
            "pipeline": pipeline,
            "is_residual": is_residual,
            "step_idx": step_idx
        }
        reference_bytes = json.dumps(reference).encode()
        # send it over gossip network
        await hidden_state_gossip_sink.broadcast(reference_bytes)
        print(f"📤 Sent {'residual' if is_residual else 'hidden_state'} blob ticket for {request_id} to {next_peer_id}")
    except Exception as e:
        print(f"❌ [DEBUG] Failed to send hidden state blob for {request_id} to {next_peer_id}: {e}")


async def send_token_output_blob(
    node: iroh.Node,
    token_gossip_sink: iroh.Gossip,
    request_id: str,
    pipeline: List[str],
    sampler_output: Any,
    step_idx: int
):
    "Serializes the sampler output, stores it as a blob, and sends it to all peers"
    try:
        # serialize the data
        serialized_data = pickle.dumps(sampler_output)
        # add the data to a blob store
        blob_output = await node.blobs().add_bytes(serialized_data)
        blob_hash = blob_output.hash
        # create a blob ticket
        blob_ticket = await node.blobs().share(blob_hash, iroh.BlobFormat.RAW, iroh.AddrInfoOptions.RELAY_AND_ADDRESSES)

        # now that the data is added to the blob, create a pointer for it that is sent as a chatter over the gossip network
        reference={
            "request_id": request_id,
            "blob_ticket": str(blob_ticket),
            "pipeline": pipeline,
            "step_idx": step_idx
        }
        reference_bytes = json.dumps(reference).encode()
        # send it over gossip network
        await token_gossip_sink.broadcast(reference_bytes)
        print(f"📤 Sent sampler output blob ticket for {request_id} step {step_idx}")
    except Exception as e:
        print(f"❌ [DEBUG] Failed to send token output blob for {request_id}: {e}")


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
    node: iroh.Node,
    hidden_state_gossip_sink: iroh.Gossip,
    token_gossip_sink: iroh.Gossip,
    peer_id: str,
    server_url: str = "http://{SERVER_IP}:8000"
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
            
            # Wait for data to arrive for this step (blocking)
            max_wait_time = 30.0  # 30 seconds timeout
            start_time = time.time()
            
            while True:
                # Thread-safe check for data availability
                with CONTEXT_LOCK:
                    step_data = INFERENCE_CONTEXT.get(request_id, {}).get(current_step)
                    if step_data and "hidden_state" in step_data and "residual" in step_data:
                        # Data is ready!
                        hidden_states = step_data["hidden_state"]
                        residual = step_data["residual"]
                        print(f"✅ Received hidden state and residual for {request_id} (step {current_step}). Injecting into the next layer.")
                        break
                
                # Check timeout
                if time.time() - start_time > max_wait_time:
                    print(f"❌ Timeout waiting for hidden state for {request_id} step {current_step}")
                    cleanup_request_context(request_id)
                    return args
                    
                # Sleep briefly before checking again
                time.sleep(0.01)

            # Process the received tensors with improved shape handling
            orig_positions = args[0]
            
            print(f"🔍 Raw tensor shapes before processing (step {current_step}):")
            print(f"   Original positions: {orig_positions.shape}")
            print(f"   Received hidden states: {hidden_states.shape}")
            print(f"   Received residual: {residual.shape}")

            # MATRIX MANIPULATION STARTS HERE - IMPROVED VERSION
            
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

            # MATRIX MANIPULATION ENDS HERE

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
        
        pipeline = hook_context["pipeline"]
        current_idx = pipeline.index(peer_id)
        next_peer_id = pipeline[current_idx + 1]

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

        # Schedule sends asynchronously (non-blocking)
        asyncio.run_coroutine_threadsafe(
            send_hidden_state_blob(
                node,
                hidden_state_gossip_sink,
                request_id,
                next_peer_id,
                pipeline,
                hidden_states.cpu(),
                is_residual=False, 
                step_idx=current_step
            ),
            main_loop
        )
        asyncio.run_coroutine_threadsafe(
            send_hidden_state_blob(
                node,
                hidden_state_gossip_sink,
                request_id,
                next_peer_id,
                pipeline,
                residual.cpu(),
                is_residual=True,
                step_idx=current_step
            ),
            main_loop
        )
        
        # NOTE: Step increment moved to sampler_post_hook to ensure it happens on ALL peers
    
    # Capture the main asyncio loop so we can schedule coroutines from threads
    main_loop = asyncio.get_running_loop()

    def sampler_post_hook(module, args, output):
        """
        Runs after the sampler. Last peer broadcasts the chosen token.
        Other peers wait for it and substitute their own sampler output.
        """
        # Get request-specific context safely
        with context_lock:
            active_contexts = [ctx for ctx in hook_contexts.values() if ctx.get("active", False)]
            if not active_contexts:
                print("❌ No active inference context found in sampler_post_hook")
                return output
            
            hook_context = active_contexts[0]
            request_id = hook_context["request_id"]
            current_step = hook_context["current_step"]
            is_last_peer = hook_context["is_last_peer"]
            pipeline = hook_context["pipeline"]

        if is_last_peer:
            # This peer is the decider. Broadcast the result.
            print(f"📢 Last peer broadcasting sampler output for step {current_step}")
            asyncio.run_coroutine_threadsafe(
                send_token_output_blob(
                    node,
                    token_gossip_sink,
                    request_id,
                    pipeline,
                    output,
                    current_step
                ),
                main_loop
            )
            
            # CRITICAL FIX: Increment step on ALL peers, including last peer
            with context_lock:
                hook_context["current_step"] = current_step + 1
                
            return output
        else:
            # Not the last peer, must wait for the definitive sampler output.
            print(f"⏳ Non-last peer waiting for sampler output for step {current_step}")
            max_wait_time = 30.0
            start_time = time.time()
            
            while True:
                with CONTEXT_LOCK:
                    step_data = INFERENCE_CONTEXT.get(request_id, {}).get(current_step)
                    if step_data and "sampler_output" in step_data:
                        received_output = step_data["sampler_output"]
                        print(f"✅ Received sampler output for {request_id} (step {current_step}).")
                        
                        # CRITICAL FIX: Increment step on non-last peers too
                        with context_lock:
                            hook_context["current_step"] = current_step + 1
                            
                        # Here, we are replacing this peer's sampler output with the one from the last peer
                        return received_output
                
                if time.time() - start_time > max_wait_time:
                    print(f"❌ Timeout waiting for sampler output for {request_id} step {current_step}")
                    cleanup_request_context(request_id)
                    # Fail loudly to prevent divergence
                    raise RuntimeError(f"Timeout waiting for sampler output for {request_id} step {current_step}")
                    
                time.sleep(0.01)

    def start_inference_run(request_id: str, pipeline: List[str], input_text: str, sampling_params: Any, assigned_layers: Dict[str, List[int]]):
        """The main inference runner"""
        # Generate unique execution ID to avoid collisions
        execution_id = str(uuid.uuid4())[:8]
        
        try:
            # Determine this peer's position in the pipeline
            idx = pipeline.index(peer_id)
            is_first = (idx == 0)
            is_last = (idx == len(pipeline) - 1)
            
            # Initialize thread-safe context for this request
            with context_lock:
                hook_contexts[execution_id] = {
                    "request_id": request_id,
                    "pipeline": pipeline,
                    "input_text": input_text,
                    "is_first_peer": is_first,
                    "is_last_peer": is_last,
                    "peer_id": peer_id,
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








        
 
