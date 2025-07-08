import asyncio
import iroh
import pickle
import httpx  # type: ignore
import json
from typing import Dict, Any, List
import time
from vllm import LLM, SamplingParams, TokensPrompt  # type: ignore
import torch
from collections import deque


# This global dictionary is the bridge between the async network world and the sync vLLM world.
# Key: request_id (str)
# Value: A dictionary holding a dictionary of asyncio.Future objects, keyed by step_idx
# Basically just a placeholder for the time we do not move on to AsyncVLLM. 
INFERENCE_CONTEXT: Dict[str, Dict[str, Any]] = {}
sent_flag=False


async def send_hidden_state_blob(
    node: iroh.Node,
    hidden_state_gossip_sink: iroh.Gossip, # have to verify if this is correct or not
    request_id: str,
    next_peer_id: str,
    pipeline: List[str],
    data_to_send: Any,
    is_residual: bool = False,
    step_idx: int = 0 # add the tracking for the step so as to only send last token during decoding
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
            "peer_id": peer_id,  # we need to change this and get the actual peer_id
            "timestamp": int(time.time())  # cast to int for Pydantic
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
    peer_id: str,
    server_url: str = "http://{SERVER_IP}:8000"
):
    """
    Create pre and post hooks for the inference pipeline, to transfer hidden states
    """
    # get the model runner worker, model itself and the sampler
    model_runner = llm.llm_engine.model_executor.driver_worker.model_runner
    model=model_runner.model
    sampler=model_runner.sampler # need to check why is this used for
    
    # Just a placeholder for the hooks 
    hook_context: Dict[str, Any] = {}

    def _slice_last_token(t: torch.Tensor) -> torch.Tensor:
        """Slices the last token from a tensor, handling both 2D and 3D cases."""
        if t.dim() == 2:
            return t[-1:, :]  # Shape (T, D) -> (1, D)
        elif t.dim() > 2:
            return t[:, -1:, :]  # Shape (B, T, D) -> (B, 1, D)
        return t

    def pre_hook(module, args):
        "this hook is called BEFORE a layer does its forward pass"
        request_id = hook_context["request_id"]
        if not hook_context["is_first_peer"]:
            print("Detected a non-first peer, injecting previous peer's hidden state")
            
            step_idx = hook_context.get("step_idx", 0)
            futures_map = INFERENCE_CONTEXT.get(request_id, {}).get("futures", {})
            
            pair = futures_map.get(step_idx)
            if not pair:
                print(f"❌ CRITICAL: Futures not found for {request_id} step {step_idx} on non-first peer!")
                return args

            hidden_state_future = pair["h"]
            residual_future = pair["r"]

            # Basically loop until complete is done, and then get the result
            print(f"⏳ Waiting for hidden state for {request_id} (step {step_idx})...")
            while not hidden_state_future.done():
                time.sleep(0.01)
            hidden_states = hidden_state_future.result()
            print(f"⏳ Waiting for residual for {request_id} (step {step_idx})...")
            while not residual_future.done():
                time.sleep(0.01)
            residual = residual_future.result()
            print(f"✅ Received hidden state and residual for {request_id} (step {step_idx}). Injecting into the next layer.")

            # Slice original positions to match the seq_len of hidden_states
            orig_positions = args[0]

            # MATRIX MANIPULATION STARTS HERE
            # 1. Ensure that the hidden states and residuals have the same batch dimension
            if hidden_states.dim() == 2: #(B,S) -> (1,B,S)
                hidden_states = hidden_states.unsqueeze(0)
            if residual.dim() == 2: #(B,S) -> (1,B,S)
                residual = residual.unsqueeze(0)
            
            # 2. Ensure the positions have the same batch dimension
            if orig_positions.dim() == 1: #(S,) -> (1,S)
                orig_positions = orig_positions.unsqueeze(0)
            elif orig_positions.dim() >= 2: #(B,S) -> (1,B,S)
                # flatten the batch dimension
                # orig_positions = orig_positions.view(orig_positions.shape[0], -1)
                orig_positions = orig_positions.contiguous()


            # 3. Matching Sequence length with precision
            batch_size, seq_len = hidden_states.shape[:2]
            
            # 4. Slice the positions to match the hidden states sequence length
            if orig_positions.shape[1] >= seq_len:
                positions_to_inject = orig_positions[:, -seq_len:]
            else:
                positions_to_inject = orig_positions

            # 5. Handle any mismatch in batch sizes
            if positions_to_inject.shape[0] != batch_size:
                # expand the positions to match the batch size
                positions_to_inject = positions_to_inject.expand(batch_size, -1)

            # 6. Move to correct device
            positions_to_inject = positions_to_inject.to(orig_positions.device, non_blocking=True)
            hidden_states_to_inject = hidden_states.to(positions_to_inject.device, non_blocking=True)
            residual_to_inject = residual.to(positions_to_inject.device, non_blocking=True)

            # MATRIX MANIPULATION ENDS HERE

            return (positions_to_inject, hidden_states_to_inject, residual_to_inject)
        
        # If this is the first peer, just return the original args
        return args

    def post_hook(module, args, output):
        "Runs after the pre-hooks in a layer - it sends the output to the next peer"
        request_id = hook_context["request_id"]
        
        # for the last peer, we need to ignore this step
        if hook_context["is_last_peer"]:
            return
            
        step_idx = hook_context.get("step_idx", 0)
        # Prevent multiple calls per inference - only send once per request
        context_key = f"{request_id}_step_{step_idx}_sent"
        if hook_context.get(context_key, False):
            return  # Already sent for this request
        hook_context[context_key] = True
        
        print(f"↪️ POST-HOOK for {request_id}, step {step_idx}: Sending hidden states...")
        hidden_states,residual=output
        pipeline = hook_context["pipeline"]
        current_idx=pipeline.index(peer_id)
        next_peer_id=pipeline[current_idx+1]

        # MATRIX MANIPULATION STARTS HERE
        # 1. Implement Smart Slicing - Only send the necessary tokens to the next peer
        # based on the generation phase. 
        # During decoding, send the last token (incremental)
        # during prefill, send the full sequence
        if step_idx > 0:
            hidden_states = _slice_last_token(hidden_states)
            residual = _slice_last_token(residual)

        # MATRIX MANIPULATION ENDS HERE

        # Schedule sends on the main asyncio loop (thread-safe)
        # not really sure if this is the best way to do this
        asyncio.run_coroutine_threadsafe(
            send_hidden_state_blob(
                node,
                hidden_state_gossip_sink,
                request_id,
                next_peer_id,
                pipeline,
                hidden_states.cpu(),
                is_residual=False, 
                step_idx=step_idx
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
                step_idx=step_idx
            ),
            main_loop
        )

        # Create the future pair for the *next* step and add it to this peer's context
        # This prepares the peer for receiving the next token's data
        next_step_idx = step_idx + 1
        INFERENCE_CONTEXT[request_id]["futures"][next_step_idx] = {
            "h": asyncio.Future(),
            "r": asyncio.Future()
        }
        
        # IMPORTANT: Increment step index for the *next* hook invocation
        hook_context["step_idx"] = next_step_idx
    
    # Capture the main asyncio loop so we can schedule coroutines from threads
    main_loop = asyncio.get_running_loop()

    def final_output_hook(module, args, output):
        global sent_flag
        "Runs on the final peer, sends the final output to the server"
        # do not do anything if we are not the last peer
        if sent_flag or not hook_context.get("is_last_peer"):
            return
        sent_flag=True # this is to avoid sending the final output to the server multiple times
        request_id = hook_context["request_id"]
        peer_id = hook_context["peer_id"]
        print(f"🎯 FINAL-OUTPUT-HOOK for {request_id}: Sending final output to the server...")

        # Send the final output to the server via the main event loop
        try:
            asyncio.run_coroutine_threadsafe(
                send_final_result_to_server(request_id, output,peer_id, server_url),
                main_loop
            )
        except Exception as e:
            print(f"❌ Failed to schedule send_final_result_to_server: {e}")

        # delete the request_id from the INFERENCE_CONTEXT
        if request_id in INFERENCE_CONTEXT:
            del INFERENCE_CONTEXT[request_id]

    # confirm this -> do we need to send in the tokens or do we need to send the prompt? assigned Layers will come from the pipeline
    def start_inference_run(request_id: str, pipeline: List[str], input_text: str, sampling_params: Any, assigned_layers: Dict[str, List[int]]):
        "The runner of inference hehe"
        # Determine this peer's position in the pipeline
        idx = pipeline.index(peer_id)
        is_first = (idx == 0)
        is_last = (idx == len(pipeline) - 1)
        # update the hook context for pre/post hooks
        hook_context.update({
            "request_id": request_id,
            "pipeline": pipeline,
            "input_text": input_text,
            "is_first_peer": is_first,
            "is_last_peer": is_last,
            "peer_id": peer_id,
            "step_idx": 0
        })
        
        # Clear any previous sent flags for this request
        for key in list(hook_context.keys()):
            if key.startswith(f"{request_id}_") and key.endswith("_sent"):
                del hook_context[key]
        
        # Safely get this peer's assigned layers
        real_layers = assigned_layers.get(peer_id, [])
        real_layers=[layer for layer in model.model.layers if "PPMissingLayer" not in layer.__class__.__name__]
        # if no real layers, exit
        if not real_layers:
            print(f"⚠️ No real layers detected here. Cannot participate in this inference")
            return
        # Fix: Attach hooks to FIRST and LAST real layers, regardless of count
        # first_layer, last_layer = real_layers[0], real_layers[1]
        first_layer = real_layers[0]
        last_layer = real_layers[-1]  # Use -1 to get the last layer
        print(f"✅ Dynamically attaching hooks to layers: {first_layer.__class__.__name__} -> {last_layer.__class__.__name__}")

        # attach the pre-hooks and the post hooks
        pre_hook_handle = first_layer.register_forward_pre_hook(pre_hook)
        post_hook_handle = last_layer.register_forward_hook(post_hook)
        # No sampler hook needed; final result will be sent after llm.generate

        print("Starting the inference run...")
        # Run vLLM inference and capture completions
        completions = llm.generate([input_text], sampling_params=sampling_params)

        # If last peer, schedule sending final result back to server
        if hook_context["is_last_peer"] and completions:
            comp = completions[0]
            final_text = comp.outputs[0].text
            try:
                asyncio.run_coroutine_threadsafe(
                    send_final_result_to_server(request_id, final_text, peer_id, server_url),
                    main_loop,
                )
            except Exception as e:
                print(f"❌ Failed to schedule send_final_result_to_server: {e}")

        pre_hook_handle.remove()
        post_hook_handle.remove()
        print(f"🎉 Inference run completed for {request_id}")
        return
    
    # return the start_inference_run function
    return start_inference_run








        
 
