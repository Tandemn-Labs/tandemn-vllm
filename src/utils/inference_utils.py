import asyncio
import iroh
import pickle
import httpx
import json
from typing import Dict, Any, List
import time
from vllm import TokensPrompt


# This global dictionary is the bridge between the async network world and the sync vLLM world.
# Key: request_id (str)
# Value: A dictionary holding asyncio.Future objects for the hidden_state and residual.
# Basically just a placeholder for the time we do not move on to AsyncVLLM. 
INFERENCE_CONTEXT: Dict[str, Dict[str, Any]] = {}


async def send_hidden_state_blob(
    node: iroh.Node,
    hidden_state_gossip_sink: iroh.Gossip, # have to verify if this is correct or not
    request_id: str,
    next_peer_id: str,
    pipeline: List[str],
    data_to_send: Any,
    is_residual: bool = False
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
            "is_residual": is_residual
        }
        reference_bytes = json.dumps(reference).encode()
        # send it over gossip network
        await hidden_state_gossip_sink.broadcast(reference_bytes)
        print(f"📤 Sent {'residual' if is_residual else 'hidden_state'} blob ticket for {request_id} to {next_peer_id}")
    except Exception as e:
        print(f"❌ [DEBUG] Failed to send hidden state blob for {request_id} to {next_peer_id}: {e}")


async def send_final_result_to_server(
    request_id: str,
    output_text: str,
    server_url: str = "http://{SERVER_IP}:8000"):
    try:
        # the output from vLLM distributed inference is sent back to the central server
        final_output=output[0].outputs[0].text
        completion_data={
            "request_id": request_id,
            "output_text": final_output,
            "peer_id": "final_peer", # we need to change this and get the actual peer_id
            "timestamp": time.time()
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
    hook_context={}

    def pre_hook(module,args):
        "this hook is called BEFORE a layer does its forward pass"
        request_id=hook_context.get("request_id")
        if not hook_context.get("is_first_peer"):
            print("Detected a non-first peer, injecting previous peer's hidden state")
            hidden_state_future=asyncio.Future() # because many times the hidden state is not available immediately
            residual_future=asyncio.Future()
            # storing future for the time we are not using AsyncVLLM
            INFERENCE_CONTEXT[request_id]["hidden_state_future"]=hidden_state_future
            INFERENCE_CONTEXT[request_id]["residual_future"]=residual_future
            
            # make an infinite loop to wait for the hidden state and residual to be available
            loop=asyncio.get_running_loop()
            hidden_states=loop.run_until_complete(hidden_state_future)
            residual=loop.run_until_complete(residual_future)
            print(f"✅ Received hidden state and residual for {request_id}. Injecting into the next layer rn")
            # add this within 
            argument = args[0]
            hidden_states_to_inject = hidden_states.to(argument.device)
            residual_to_inject = residual.to(argument.device)
            return (argument, hidden_states_to_inject, residual_to_inject)
        # else just return the args as is, as we do not have to do anything here
        return args

    def post_hook(module, args, output):
        "Runs after the pre-hooks in a layer - it sends the output to the next peer"
        request_id=hook_context.get("request_id")
        # for the last peer, we need to ignore this step, given we have to convert logits to text there
        if hook_context.get("is_last_peer"):
            return 
        print(f"↪️ POST-HOOK for {request_id}: Sending hidden states...")
        hidden_states,residual=output
        pipeline=hook_context.get("pipeline")
        current_idx=pipeline.index(peer_id)
        next_peer_id=pipeline[current_idx+1]

        # have to confirm if this is what we want to do with the residual and the hooks
        asyncio.create_task(send_hidden_state_blob(node, hidden_state_gossip_sink,request_id, next_peer_id, pipeline, hidden_states.cpu(), is_residual=False))
        asyncio.create_task(send_hidden_state_blob(node, hidden_state_gossip_sink,request_id, next_peer_id, pipeline, residual.cpu(), is_residual=False))
    
    def final_output_hook(module, args, output):
        "Runs on the final peer, sends the final output to the server"
        request_id=hook_context.get("request_id")
        # do not do anything if we are not the last peer
        if not hook_context.get("is_last_peer"):
            return
        print(f"🎯 FINAL-OUTPUT-HOOK for {request_id}: Sending final output to the server...")

        # send the final output to the server
        asyncio.create_task(send_final_result_to_server(request_id, output, server_url, peer_id))

        # delete the request_id from the INFERENCE_CONTEXT
        if request_id in INFERENCE_CONTEXT:
            del INFERENCE_CONTEXT[request_id]
    # confirm this -> do we need to send in the tokens or do we need to send the prompt? assigned Layers will come from the pipeline
    def start_inference_run(request_id, pipeline: List[str], input_text: str, sampling_params, assigned_layers: Dict[str, Any]):
        "The runner of inference hehe"
        # update the hook context, in order to make it concurrent (atleast that what i think i should have it)
        hook_context.update({
            "request_id": request_id,
            "pipeline": pipeline,
            "input_text": input_text,
            "is_first_peer": True,
            "is_last_peer": False
        })
        
        # Safely get this peer's layers
        real_layers = assigned_layers.get(peer_id, [])
        real_layers=[layer for layer in model.model.layers if "PPMissingLayer" not in layer.__class__.__name__]
        # if no real layers, exit
        if not real_layers:
            print(f"⚠️ No real layers detected here. Cannot participate in this inference")
            return
        
        first_layer,last_layer=real_layers[0], real_layers[1]
        print(f"✅ Dynamically attaching hooks to layers: {first_layer.__class__.__name__} -> {last_layer.__class__.__name__}")

        # attach the pre-hooks and the post hooks
        pre_hook_handle=first_layer.register_forward_pre_hook(pre_hook)
        post_hook_handle=last_layer.register_forward_hook(post_hook)
        sampler_hook_handle=sampler.register_forward_hook(final_output_hook)
        
        # GPU GO BRRRRRRRRRRRRRRR
        print("Starting the inference run...")
        # Use raw text prompt for vLLM
        llm.generate([input_text], sampling_params)

        pre_hook_handle.remove()
        post_hook_handle.remove()
        sampler_hook_handle.remove()
        print(f"🎉 Inference run completed for {request_id}")
        return
    
    # return the start_inference_run function
    return start_inference_run








        
 
