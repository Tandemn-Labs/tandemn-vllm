import asyncio
import pickle
import threading
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

import httpx  # type: ignore
import numpy as np
import torch

from src.utils.tensor_protocol_adapter import TensorTransport
from vllm import LLM  # type: ignore

# This global dictionary holds the actual tensor data, not futures
# Key: request_id (str)
# Value: A dictionary with step-indexed hidden states and residuals
# Structure: {request_id: {step_idx: {"hidden_state": tensor, "residual": tensor}}}
INFERENCE_CONTEXT: Dict[str, Dict[str, Any]] = {}
CONTEXT_LOCK = threading.Lock()  # Thread-safe access to INFERENCE_CONTEXT

# ------------------------------------------------------------------
#  Per–request / per-step Events that tell a waiting peer "data ready"
# ------------------------------------------------------------------
# Backwards-compatible: STEP_EVENTS keeps signaling for hidden/residual tensors
STEP_EVENTS: Dict[str, Dict[int, threading.Event]] = defaultdict(dict)
# New: separate event map for sampler outputs to avoid cross-signaling
STEP_EVENTS_SAMPLER: Dict[str, Dict[int, threading.Event]] = defaultdict(dict)

# Serialize per-peer inference until hooks are made fully request-aware
INFERENCE_MUTEX = threading.Lock()

# Reference to the async loop, right now it's the main thread's asyncio loop
asyncio_loop = None

# Per-batch context storage, maps batch id to dict of metadata
batch_metadata = {}

# Data wait timeout i.e. How long hooks should wait for data
DATA_TIMEOUT = 5


def cleanup_request_context(request_id: str):
    """Thread-safe cleanup of request context"""
    with CONTEXT_LOCK:
        if request_id in INFERENCE_CONTEXT:
            del INFERENCE_CONTEXT[request_id]
            print(f"🧹 Cleaned up context for {request_id}")
        if request_id in STEP_EVENTS:
            del STEP_EVENTS[request_id]
            print(f"🧹 Cleaned up step events for {request_id}")
        if request_id in STEP_EVENTS_SAMPLER:
            del STEP_EVENTS_SAMPLER[request_id]
            print(f"🧹 Cleaned up sampler step events for {request_id}")
        if threading.get_ident() in batch_metadata:
            del batch_metadata[threading.get_ident()]
            print(
                f"🧹 Cleaned up batch metadata for {request_id}, using key {threading.get_ident()}"
            )


async def stream_token_to_server(
    batch_id: str, tokens: List[str], server_url: str = "http://{SERVER_IP}:8000"
):
    try:
        tokens_data = {"batch_id": batch_id, "tokens": tokens}

        async with httpx.AsyncClient() as client:
            response = await client.post(f"{server_url}/streaming", json=tokens_data)

            response.raise_for_status()

    except Exception as e:
        print(f"[ERROR] stream_tokens_to_server - {e}")


async def send_final_result_to_server(
    batch_id: str,
    final_text: List[str],
    peer_id: str,
    server_url: str = "http://{SERVER_IP}:8000",
):
    try:
        # vLLM ≥0.4 returns CompletionSequenceGroupOutput
        # print(f"🔍 Output object type: {type(final_text)}")
        # if isinstance(output_obj, str):
        #     final_text = output_obj

        # elif hasattr(output_obj, "sequences"):  # CompletionSequenceGroupOutput
        #     final_text = output_obj.sequences[0].text

        # elif hasattr(output_obj, "outputs"):  # RequestOutput
        #     final_text = output_obj.outputs[0].text

        # elif hasattr(output_obj, "text"):  # SequenceOutput (older)
        #     final_text = output_obj.text
        # else:
        #     raise ValueError(f"Unknown output type: {type(output_obj)}")

        completion_data = {
            "batch_id": batch_id,
            "output_text": final_text,
            "peer_id": peer_id,
            "timestamp": int(time.time()),
        }
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                f"{server_url}/completion", json=completion_data
            )
            if response.status_code == 200:
                print(f"🔍 Response: {final_text}")
                print(f"✅ Sent final result to server for batch {batch_id}")
            else:
                print(
                    f"❌ Failed to send completion: {response.status_code} - {response.text}"
                )
    except Exception as e:
        print(f"❌ Error sending completion to server: {e}")


# def create_virtual_sampler_output(request_id: str, step_idx: int) -> Any:
#    """
#     Create a virtual/mock SamplerOutput that satisfies vLLM's expectations
#     for intermediate peers that don't actually need the real sampler state.

#     This is a lightweight placeholder that contains minimal required fields.
#     """
#     try:
#         from vllm.model_executor.layers.sampler import SamplerOutput
#         from vllm.sequence import CompletionSequenceGroupOutput, Logprob, SequenceOutput

#         # Choose a dummy token id and provide a matching logprob entry
#         output_token_id = 1

#         # minimal mock SequenceOutput
#         mock_sequence_output = SequenceOutput(
#             parent_seq_id=0,
#             output_token=output_token_id,
#             logprobs={output_token_id: Logprob(logprob=0.0)},
#         )

#         # minimal mock CompletionSequenceGroupOutput
#         mock_completion_output = CompletionSequenceGroupOutput(
#             samples=[mock_sequence_output],
#             prompt_logprobs=None,
#         )

#         # virtual SamplerOutput with just the required outputs field
#         virtual_sampler_output = SamplerOutput(
#             outputs=[mock_completion_output],
#         )

#         print(
#             f"🎭 Created virtual SamplerOutput for intermediate peer (request: {request_id}, step: {step_idx})"
#         )
#         return virtual_sampler_output

#     except Exception as e:
#         print(f"❌ Failed to create virtual SamplerOutput: {e}")
#         return None


def register_inference_hooks(
    llm: "LLM",
    node: TensorTransport,
    peer_id: str,
    tokenizer: Optional,
    server_url: str = "http://{SERVER_IP}:8000",
    next_peer_ticket: Optional[str] = None,
    pipeline: Optional[List[str]] = None,
):
    """
    Create pre and post hooks for the inference pipeline, to transfer hidden states
    """
    # get the model runner worker, model itself and the sampler
    try:
        if hasattr(llm, "llm_engine"):
            driver_worker = llm.llm_engine.model_executor.driver_worker
            if hasattr(driver_worker, "scorer_worker"):
                # SpecDecodeWorker case - double nested model_runner
                model_runner = driver_worker.scorer_worker.model_runner.model_runner
            else:
                # Regular worker case
                model_runner = driver_worker.model_runner
        elif hasattr(llm, "engine"):
            # AsyncLLMEngine (v0) exposes underlying LLMEngine at .engine
            driver_worker = llm.engine.model_executor.driver_worker
            if hasattr(driver_worker, "scorer_worker"):
                # SpecDecodeWorker case - double nested model_runner
                model_runner = driver_worker.scorer_worker.model_runner.model_runner
            else:
                # Regular worker case
                model_runner = driver_worker.model_runner
        else:
            raise AttributeError(
                "Unsupported engine object passed to register_inference_hooks"
            )
    except Exception as e:
        raise RuntimeError(f"Failed to resolve model_runner from engine: {e}")

    global asyncio_loop

    model = model_runner.model
    sampler = model_runner.sampler

    # context_lock protects hook_contexts and all fields inside each context
    # (request_id, current_step, active, is_first/last, routing info). Hooks
    # from multiple threads consult and update this state during forward passes.
    context_lock = threading.RLock()

    asyncio_loop = asyncio.get_running_loop()
    # print(f"In register hooks - asyncio loop is {id(main_loop)}")

    # Discover hidden/vocab sizes from model config or layers where possible
    def get_model_hidden_size() -> Optional[int]:
        try:
            # Try common config locations
            cfg = getattr(model, "config", None)
            if cfg is not None and hasattr(cfg, "hidden_size"):
                return int(getattr(cfg, "hidden_size"))
            inner_model = getattr(model, "model", None)
            if inner_model is not None:
                cfg = getattr(inner_model, "config", None)
                if cfg is not None and hasattr(cfg, "hidden_size"):
                    return int(getattr(cfg, "hidden_size"))
                # Heuristic via first layer weights
                layers = getattr(inner_model, "layers", None)
                if layers:
                    first_layer = layers[0]
                    for attr_path in [
                        "self_attn.q_proj.weight",
                        "mlp.gate_proj.weight",
                        "mlp.up_proj.weight",
                    ]:
                        try:
                            weight = first_layer
                            for part in attr_path.split("."):
                                weight = getattr(weight, part)
                            if hasattr(weight, "shape"):
                                return int(weight.shape[1])
                        except Exception:
                            continue
        except Exception:
            pass
        return None

    def get_model_vocab_size() -> Optional[int]:
        try:
            cfg = getattr(model, "config", None)
            if cfg is not None and hasattr(cfg, "vocab_size"):
                return int(getattr(cfg, "vocab_size"))
            inner_model = getattr(model, "model", None)
            if inner_model is not None:
                cfg = getattr(inner_model, "config", None)
                if cfg is not None and hasattr(cfg, "vocab_size"):
                    return int(getattr(cfg, "vocab_size"))
        except Exception:
            pass
        return None

    def pre_hook(module, args):
        """Ultra-minimal pre-hook for maximum performance"""
        # Get request-specific context with minimal overhead
        with context_lock:
            hook_context = batch_metadata[threading.get_ident()]
            batch_id = hook_context["batch_id"]
            current_step = hook_context["current_step"]
            # print(
            #     f"🔍 Pre-hook called for request {batch_id} step {current_step} thread {threading.current_thread().name}, {threading.current_thread().ident}"
            # )
            # loop = asyncio.get_running_loop()
            # print(f"asyncio loop - {id(loop)}, {loop}")

        # Skip ALL checks if first peer
        if hook_context["is_first_peer"]:
            return args

        # Wait for data (unavoidable, but optimized)
        with CONTEXT_LOCK:
            event = STEP_EVENTS[batch_id].setdefault(current_step, threading.Event())
        if not event.wait(timeout=DATA_TIMEOUT):
            peer_id = hook_context["peer_id"]
            server_url = hook_context["server_url"]
            handle_failure(
                batch_id=batch_id,
                peer_id=peer_id,
                error="Pre-Hook timed out waiting for tensor data",
                server_url=server_url,
            )
            raise RuntimeError(
                f"Timeout waiting for pre-hook input for {batch_id} step {current_step}"
            )

        # Direct memory access (minimal locking)
        with CONTEXT_LOCK:
            step_data = INFERENCE_CONTEXT[batch_id][str(current_step)]
            hidden_states = step_data["hidden_state"]
            residual = step_data["residual"]
            positions = step_data["positions"]
        # this is just to get the device
        positions_for_device = args[0]
        device = positions_for_device.device

        # Infer payload hidden size and keep it in context for visibility
        payload_hidden_size = int(hidden_states.shape[-1])

        if hook_context.get("hidden_size") is None:
            hook_context["hidden_size"] = payload_hidden_size
            # print(f"🔧 Inferred hidden size from payload: {payload_hidden_size}")
        elif hook_context["hidden_size"] != payload_hidden_size:
            pass

        # Single conditional for step - ultra optimized reshaping
        hidden_states = torch.tensor(hidden_states)
        residual = torch.tensor(residual)
        positions = torch.tensor(positions)
        if current_step:  # Decode phase
            hidden_reshaped = hidden_states.to(device, non_blocking=True)
            positions_reshaped = positions.to(device, non_blocking=True)
            residual_reshaped = residual.to(device, non_blocking=True)
            # Pre-computed shapes for decode (single token)
            # Ensure tensors are on correct device
            # print(f"🔍 Pre-hook reshaping for request {request_id} step {current_step}", hidden_states, hidden_states.shape)
            # hidden_reshaped = hidden_states.view(1, 1, payload_hidden_size).to(
            #     device, non_blocking=True
            # )
            # print(f"🔍 Pre-hook reshaped hidden_states: {hidden_reshaped}", hidden_reshaped.shape)
            # print(f"🔍 Pre-hook residual: {residual}", residual.shape)
            # residual_reshaped = residual.view(1, 1, payload_hidden_size).to(
            #     device, non_blocking=True
            # )
            # print(f"🔍 Pre-hook residual_reshaped: {residual_reshaped}", residual_reshaped.shape)
            # print(f"🔍 Pre-hook positions: {positions}", positions.shape)
            # positions_reshaped = (
            #     positions.view(1, 1).to(device, non_blocking=True)
            #     if positions.numel() == 1
            #     else positions.view(1, -1)[:, -1:].to(device, non_blocking=True)
            # )
            # print(f"🔍 Pre-hook positions_reshaped: {positions_reshaped}", positions_reshaped.shape)

            # Clean up old data immediately
            if current_step > 1:
                with CONTEXT_LOCK:
                    INFERENCE_CONTEXT[batch_id].pop(str(current_step - 2), None)
                    STEP_EVENTS[batch_id].pop(current_step - 2, None)
                    STEP_EVENTS_SAMPLER[batch_id].pop(current_step - 2, None)

            return (positions_reshaped, hidden_reshaped, residual_reshaped)
        else:  # Prompt phase
            _ = hidden_states.shape[0]  # sequence length
            hidden_reshaped = hidden_states.to(device, non_blocking=True)
            positions_reshaped = positions.to(device, non_blocking=True)
            residual_reshaped = residual.to(device, non_blocking=True)
            # Reshape with minimal operations
            # print(f"🔍 Pre-hook reshaping for request {request_id} step {current_step}", hidden_states, hidden_states.shape)
            # hidden_reshaped = hidden_states.view(1, seq_len, payload_hidden_size).to(
            #     device, non_blocking=True
            # )
            # print(f"🔍 Pre-hook reshaped hidden_states: {hidden_reshaped}", hidden_reshaped.shape)
            # print(f"🔍 Pre-hook residual: {residual}", residual.shape)
            # residual_reshaped = residual.view(1, seq_len, payload_hidden_size).to(
            # device, non_blocking=True
            # )
            # print(f"🔍 Pre-hook residual_reshaped: {residual_reshaped}", residual_reshaped.shape)
            # print(f"🔍 Pre-hook positions: {positions}", positions.shape)
            # Handle positions efficiently
            # if positions.dim() == 1:
            # positions = positions.unsqueeze(0)
            # positions_reshaped = (
            # positions[:, -seq_len:] if positions.shape[1] >= seq_len else positions
            # )
            # positions_reshaped = positions_reshaped.to(device, non_blocking=True)
            # print(f"🔍 Pre-hook positions_reshaped: {positions_reshaped}", positions_reshaped.shape)
            return (positions_reshaped, hidden_reshaped, residual_reshaped)

    def post_hook(module, args, output):
        """Ultra-minimal post-hook for maximum performance"""
        # Get context with minimal overhead
        with context_lock:
            hook_context = batch_metadata[threading.get_ident()]

        # Skip if last peer (no sending needed)
        if hook_context["is_last_peer"]:
            return

        request_id = hook_context["batch_id"]
        current_step = hook_context["current_step"]

        # print(
        #     f"post-hook: {request_id}, {current_step} thread {threading.current_thread().name}, {threading.current_thread().ident}"
        # )
        # loop = asyncio.get_running_loop()
        # print(f"asyncio loop - {id(loop)}, {loop}")

        # Fast duplicate check
        context_key = f"sent_step_{current_step}"
        if hook_context.get(context_key, False):
            return
        hook_context[context_key] = True

        # print(f"post-hook: output-type - {type(output)}")
        hidden_states, residual = output
        positions = args[0]
        # print(
        #     f"post-hook: hidden_states type - {type(hidden_states)}, residual type - {type(residual)}"
        # )
        # print(
        #     f"🔍 Post-hook called for request {request_id} step {current_step}",
        #     hidden_states,
        #     hidden_states.shape,
        # )
        # print(f"🔍 Post-hook residual: {residual}", residual.shape)

        # Single slice operation for decode (no validation)
        # if current_step > 0:
        #     print("=" * 80)
        #     print("it is probably failing here for current_step", current_step)
        #     print("=" * 80)
        # Ultra-fast slicing for single token
        # if hidden_states.dim() == 3 and hidden_states.shape[1] > 1:
        #     hidden_states = hidden_states[:, -1:, :]
        #     residual = residual[:, -1:, :]
        # elif hidden_states.dim() == 2 and hidden_states.shape[0] > 1:
        #     hidden_states = hidden_states[-1:, :]
        #     residual = residual[-1:, :]
        # print(
        #     f"🔍 Post-hook hidden_states: {hidden_states}",
        #     hidden_states.shape,
        #     f"id {id(hidden_states)}",
        # )
        # print(f"🔍 Post-hook residual: {residual}", residual.shape)
        # # Normalize ranks before sending to ensure both tensors match
        # if current_step == 0:
        #     # Prompt phase: ensure (seq, hidden)
        #     if hidden_states.dim() == 3 and hidden_states.size(0) == 1:
        #         hidden_states = hidden_states.squeeze(0)
        #     if residual.dim() == 3 and residual.size(0) == 1:
        #         residual = residual.squeeze(0)
        #     if hidden_states.dim() == 3:
        #         hidden_states = hidden_states.view(-1, hidden_states.size(-1))
        #     if residual.dim() == 3:
        #         residual = residual.view(-1, residual.size(-1))
        # else:
        #     # Decode phase: ensure (1, hidden)
        #     if hidden_states.dim() >= 2:
        #         hidden_states = hidden_states.view(-1, hidden_states.size(-1))[-1:, :]
        #     else:
        #         hidden_states = hidden_states.view(1, -1)
        #     if residual.dim() >= 2:
        #         residual = residual.view(-1, residual.size(-1))[-1:, :]
        #     else:
        #         residual = residual.view(1, -1)
        # print(
        #     f"🔍 Post-hook hidden_states: {hidden_states}",
        #     hidden_states.shape,
        #     f"id {id(hidden_states)}",
        # )
        # print(f"🔍 Post-hook residual: {residual}", residual.shape)
        # Direct tensor sending (skip CPU conversion if possible)
        next_peer_id = hook_context["next_peer_id"]
        next_peer_ticket = hook_context["next_peer_ticket"]

        # Async send with minimal conversion
        # print(
        #     f"main_loop given to send_inference_tensors_fast - {id(main_loop)}, {main_loop}"
        # )

        # TODO: Merely for testing
        # if random.randint(1, 10) < 2:
        #     return

        asyncio.run_coroutine_threadsafe(
            send_inference_tensors_fast(
                node,
                request_id,
                next_peer_id,
                hidden_states.clone(),  # Send torch tensor directly
                residual.clone(),  # Send torch tensor directly
                positions.clone(),  # Send torch tensor directly
                step_idx=current_step,
                next_peer_ticket=next_peer_ticket,
            ),
            asyncio_loop,
        )
        # print("post-hook - after calling coro to send data")

        # NOTE: Step increment moved to sampler_post_hook to ensure it happens on ALL peers

    def sampler_post_hook(module, args, output):
        """
        Post-hook for the sampler module.
        - For the last peer: broadcasts sampler output to all other peers
        - For non-last peers: waits for sampler output from the last peer
        """

        # Get request-specific context safely
        with context_lock:
            hook_context = batch_metadata[threading.get_ident()]
            batch_id = hook_context["batch_id"]
            current_step = hook_context.get("current_step", 0)
            is_last_peer = hook_context.get("is_last_peer", False)
            pipeline = hook_context["pipeline"]
            peer_id = hook_context["peer_id"]
            server_url = hook_context["server_url"]

        to_return = None

        # print(
        #     f"sampler-post-hook: {request_id}, {current_step} thread {threading.current_thread().name}, {threading.current_thread().ident}"
        # )
        # loop = asyncio.get_running_loop()
        # print(f"asyncio loop - {id(loop)}, {loop}")

        if is_last_peer:
            # Serialize the entire SamplerOutput object
            sampler_output_bytes = pickle.dumps(output)
            sampler_output_np = np.frombuffer(sampler_output_bytes, dtype=np.uint8)

            # print(f"sampler-post-hook: output type - {type(output)}, data - {output}")

            # Send to all other peers
            peer_tickets = pipeline[:-1] if len(pipeline) > 1 else []

            for peer in peer_tickets:
                asyncio.run_coroutine_threadsafe(
                    send_sampler_output(
                        node,
                        batch_id,
                        peer,
                        sampler_output_np,
                        step_idx=current_step,
                        next_peer_ticket=peer,
                    ),
                    asyncio_loop,
                )

            # Stores parent_seq_id if its first step (current_step = 0)
            if current_step == 0:
                parent_seq_id = [
                    completion.samples[0].parent_seq_id for completion in output.outputs
                ]
                hook_context["parent_seq_id"] = parent_seq_id

            # Decode tokens from number
            parent_seq_id = hook_context["parent_seq_id"]
            curr_seq = []
            token_numbers = []
            for completion in output.outputs:
                curr_seq.append(completion.samples[0].parent_seq_id)
                token_numbers.append([completion.samples[0].output_token])
            if hasattr(tokenizer, "batch_decode"):
                tokens_str = tokenizer.batch_decode(token_numbers)
            else:
                # Assume MistralTokenizer - decode individually
                tokens_str = [
                    tokenizer.decode(token_list) for token_list in token_numbers
                ]

            # Fill in empty string for requests that completed
            tokens_return = []
            curr_i = 0
            to_remove = []  # indexes
            for i in range(len(parent_seq_id)):
                # Current id still inferencing
                if curr_i < len(curr_seq) and curr_seq[curr_i] == parent_seq_id[i]:
                    tokens_return.append(tokens_str[curr_i])
                    curr_i += 1
                # Current id already stopped inferencing
                else:
                    to_remove.append(i)
                    tokens_return.append(None)

            # Now remove the sequences that are complete
            for i in reversed(to_remove):
                del parent_seq_id[i]

            # print(
            #     f"sampler_post_hook - parent_seq_id: {parent_seq_id}, curr_seq: {curr_seq}, tokens_return: {tokens_return}"
            # )
            # print(f"sampler-post-hook - tokens_str {tokens_str}")
            asyncio.run_coroutine_threadsafe(
                stream_token_to_server(
                    batch_id=batch_id, tokens=tokens_return, server_url=server_url
                ),
                asyncio_loop,
            )

            # print("sampler-post-hook: last-peer, returned same sampler")
            to_return = output

        # If not last peer
        else:
            # Wait for sampler output using threading.Event, note that it's indexed by current_step before increment
            with CONTEXT_LOCK:
                event = STEP_EVENTS_SAMPLER[batch_id].setdefault(
                    current_step, threading.Event()
                )

            if not event.wait(timeout=DATA_TIMEOUT):
                handle_failure(
                    batch_id=batch_id,
                    peer_id=peer_id,
                    server_url=server_url,
                    error="Sampler post hook for first peer timed out waiting for sampler output",
                )
                raise RuntimeError(
                    f"Timeout waiting for sampler output for {batch_id} step {current_step}"
                )

            # Data is ready!
            with CONTEXT_LOCK:
                # print(f"🔍 INFERENCE_CONTEXT: {INFERENCE_CONTEXT[request_id][str(current_step)]}, for step {current_step}")
                received_output = INFERENCE_CONTEXT[batch_id][str(current_step)][
                    "sampler_output"
                ]
                # print(
                #     f"✅ FIRST peer received REAL sampler output for step {current_step}"
                # )

            # print(f"sampler-post-hook: first-peer, received sampler {received_output}")
            to_return = received_output

        # Clean up old data to prevent memory growth
        # print("sampler-post-hook - Attempting to grab CONTEXT_LOCK")
        hook_context["current_step"] = current_step + 1
        with CONTEXT_LOCK:
            if current_step > 0:
                INFERENCE_CONTEXT[batch_id].pop(str(current_step - 1), None)
                STEP_EVENTS[batch_id].pop(current_step - 1, None)
                STEP_EVENTS_SAMPLER[batch_id].pop(current_step - 1, None)

        return to_return

    def start_inference_run(
        batch_id: str,
        pipeline: List[str],
        input_text: List[str],
        sampling_params: Any,
        # batcher: req_batcher.Batcher,
        batcher: Any,  # can be either the normal batcher or the mass batcher
        file_id: str = None,  # these are only for the mass batcher, hence optional
        batch_number: int = None,  # these are only for the mass batcher, hence optional
    ):
        """The main inference runner"""

        print(f"start_inference_run - {batch_id}")
        print(
            f"THREAD - {threading.current_thread().name}, {threading.current_thread().ident}"
        )
        print(f"LOOP - {id(asyncio_loop)}, {asyncio_loop}")

        # Predefine hook handles for safe, idempotent cleanup
        pre_hook_handle = None
        post_hook_handle = None
        sampler_hook_handle = None

        # Serialize per-peer inference to avoid concurrent hook contexts
        with INFERENCE_MUTEX:
            try:
                # Determine this peer's position in the pipeline
                idx = pipeline.index(peer_id)
                is_first = idx == 0
                is_last = idx == len(pipeline) - 1

                # Determine next peer info if not last
                next_peer_id = None
                next_peer_ticket = None
                if not is_last:
                    next_peer_id = pipeline[idx + 1]
                    next_peer_ticket = pipeline[
                        idx + 1
                    ]  # In this implementation, peer_id and ticket are the same

                # Initialize thread-safe context for this request
                with context_lock:
                    batch_metadata[threading.get_ident()] = {
                        "batch_id": batch_id,
                        "pipeline": pipeline,
                        "input_text": input_text,
                        "is_first_peer": is_first,
                        "is_last_peer": is_last,
                        "peer_id": peer_id,
                        "next_peer_id": next_peer_id,
                        "next_peer_ticket": next_peer_ticket,
                        "current_step": 0,
                        "active": True,
                        # pre-seed with model-reported sizes when available
                        "hidden_size": get_model_hidden_size(),
                        "server_url": server_url,
                    }

                # Get this peer's assigned layers
                real_layers = [
                    layer
                    for layer in model.model.layers
                    if "PPMissingLayer" not in layer.__class__.__name__
                ]
                if not real_layers:
                    print(
                        "⚠️ No real layers detected here. Cannot participate in this inference"
                    )
                    return

                # Attach hooks to first and last real layers
                first_layer = real_layers[0]
                last_layer = real_layers[-1]
                print(
                    f"✅ Dynamically attaching hooks to layers: {first_layer.__class__.__name__} -> {last_layer.__class__.__name__}"
                )

                # Report sizes once for visibility
                model_hidden = batch_metadata[threading.get_ident()].get("hidden_size")
                if model_hidden is not None:
                    print(f"ℹ️ Model-reported hidden/residual size: {model_hidden}")
                vocab_size = get_model_vocab_size()
                if vocab_size is not None:
                    print(f"ℹ️ Model-reported vocab size (sampler): {vocab_size}")
                # Register hooks
                pre_hook_handle = first_layer.register_forward_pre_hook(pre_hook)
                post_hook_handle = last_layer.register_forward_hook(post_hook)
                sampler_hook_handle = sampler.register_forward_hook(sampler_post_hook)

                print("Starting the inference run...")
                # Run vLLM inference
                if hasattr(llm, "engine"):
                    # AsyncLLMEngine (v0): generate is an async generator and requires request_id
                    async def _collect_async_outputs():
                        last_output = None
                        async for out in llm.generate(
                            input_text,  # prompt as string
                            sampling_params,
                            batch_id,
                        ):
                            last_output = out
                        return last_output

                    # Runs the coro in the event loop belonging in another thread
                    future = asyncio.run_coroutine_threadsafe(
                        _collect_async_outputs(), asyncio_loop
                    )
                    final_output = future.result()
                    completions = [final_output] if final_output is not None else []
                else:
                    # Blocking LLM path
                    print(f"llm-generate: {batch_id}")
                    completions = llm.generate(
                        prompts=input_text, sampling_params=sampling_params
                    )

                # this is only for the mass batcher, hence optional
                if file_id and batch_number:
                    if is_last:
                        # extract results for this batch
                        results = [comp.outputs[0].text for comp in completions]
                        # save to S3 using a callback
                        asyncio.run_coroutine_threadsafe(
                            save_results_to_s3(
                                file_id=file_id,
                                batch_id=batch_id,
                                batch_number=batch_number,
                                results=results,
                            ),
                            asyncio_loop,
                        )
                        # clear busy flag for MassBatcher
                        if hasattr(batcher, "busy_clear"):
                            asyncio.run_coroutine_threadsafe(
                                batcher.busy_clear(), asyncio_loop
                            )
                        # skip streaming as it will straight up exit if we save to S3
                    return

                # If last peer, send final result to server
                if is_last and completions:
                    final_text = [comp.outputs[0].text for comp in completions]
                    print(f"Final text (before sending) - len({len(final_text)})")

                    for i in range(len(final_text)):
                        print(f"\nOutput number {i}\n", "+" * 50)
                        print(f"{final_text[i]}\n")

                    try:
                        # Use a unique sent flag per request to avoid collisions
                        asyncio.run_coroutine_threadsafe(
                            send_final_result_to_server(
                                batch_id, final_text, peer_id, server_url
                            ),
                            asyncio_loop,
                        )
                        print(f"🎯 Final result sent for {batch_id}")
                    except Exception as e:
                        print(f"❌ Failed to schedule send_final_result_to_server: {e}")

                print(f"🎉 Inference run completed for {batch_id}")

            except Exception as e:
                print(f"❌ Error in inference run for {batch_id}: {e}")
                llm.reset_prefix_cache()
                llm.llm_engine.reset_prefix_cache()
                llm.llm_engine.reset_mm_cache()
                print(
                    f"start_inference_run [except] block - has_unfinished_requests: {llm.llm_engine.has_unfinished_requests()}"
                )

                # Abort all pending requests in the LLM engine
                all_request_ids = []

                for scheduler in llm.llm_engine.scheduler:
                    # Get request IDs from all three queues
                    for seq_group in scheduler.waiting:
                        all_request_ids.append(seq_group.request_id)
                    for seq_group in scheduler.running:
                        all_request_ids.append(seq_group.request_id)
                    for seq_group in scheduler.swapped:
                        all_request_ids.append(seq_group.request_id)

                # Abort all collected request IDs at once
                if all_request_ids:
                    llm.llm_engine.abort_request(all_request_ids)

                print(
                    f"start_inference_run [except] block - has_unfinished_requests: {llm.llm_engine.has_unfinished_requests()}"
                )

                import traceback

                traceback.print_exc()

            finally:
                # Mark context inactive early to avoid further hook activity for this execution
                with context_lock:
                    if threading.get_ident() in batch_metadata:
                        batch_metadata[threading.get_ident()]["active"] = False

                # Always clean up hooks idempotently, even on error/early exit
                try:
                    if pre_hook_handle is not None:
                        try:
                            pre_hook_handle.remove()
                        except Exception:
                            pass
                    if post_hook_handle is not None:
                        try:
                            post_hook_handle.remove()
                        except Exception:
                            pass
                    if sampler_hook_handle is not None:
                        try:
                            sampler_hook_handle.remove()
                        except Exception:
                            pass
                finally:
                    # Remove context entry now that hooks are torn down
                    with context_lock:
                        if threading.get_ident() in batch_metadata:
                            del batch_metadata[threading.get_ident()]

                # Always clean up per-request transport/context
                cleanup_request_context(batch_id)

                # Tell main thread to schedule next batch
                if peer_id == pipeline[0] and not (file_id and batch_number):
                    asyncio.run_coroutine_threadsafe(batcher.busy_clear(), asyncio_loop)

        return

    # return the start_inference_run function
    return start_inference_run


async def save_results_to_s3(
    batch_id: str, file_id: str, batch_number: int, results: List[str]
):
    """
    Save results to S3 per batch
    """
    import json
    import os
    from datetime import datetime

    # create a new path
    safe_file_id = file_id.replace("\\", "_").replace("/", "_").replace(":", "")
    S3_BUCKET = os.environ.get("S3_RESULTS_BUCKET", "tandemn-results")
    s3_path = (
        f"s3://{S3_BUCKET}/results/{safe_file_id}/batch_{batch_number}_{batch_id}.json"
    )

    result_data = {
        "batch_id": batch_id,
        "file_id": file_id,
        "batch_number": batch_number,
        "timestamp": datetime.utcnow().isoformat(),
        "results": results,
        "num_results": len(results),
    }

    # Use smart_open to save to S3
    from smart_open import open

    try:
        with open(s3_path, "w") as f:
            json.dump(result_data, f, indent=2)

        print(f"✅ Saved batch {batch_number} with {len(results)} results to S3")
    except Exception as e:
        print(f"❌ Failed to save to S3: {e}")
        raise


# Cleans up data structures and sends failure to server
def handle_failure(batch_id: str, peer_id: str, error: str, server_url: str):
    try:
        cleanup_request_context(batch_id)

        data = {"batch_id": batch_id, "peer_id": peer_id, "error": error}
        with httpx.Client(timeout=5.0) as client:
            client.post(f"{server_url}/batch_failed", json=data)
    except Exception as e:
        print(f"handle_failure - {e}")


async def send_inference_tensors_fast(
    tensor_transport: "TensorTransport",
    request_id: str,
    next_peer_id: str,
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    positions: torch.Tensor,
    step_idx: int = 0,
    next_peer_ticket: str = "",
):
    """
    Ultra-fast tensor sending with minimal conversion overhead.
    Accepts torch tensors directly and does lazy conversion only when needed.
    """
    try:
        # print("send_inference_tensors_fast - beginning of fn")
        if not next_peer_ticket:
            raise ValueError("next_peer_ticket must be provided")

        # Convert to numpy with minimal overhead
        # Use .detach() to avoid autograd overhead, .cpu() only if needed
        # print(
        #     f"🔍 Hidden states: {hidden_states}",
        #     hidden_states.shape,
        #     f"id {id(hidden_states)}",
        # )
        # print(f"🔍 Residual: {residual}", residual.shape)
        # print(f"send_ITF - {hidden_states.detach()}")
        # print(f"send_ITF - {residual.detach()}")
        if hidden_states.is_cuda:
            hidden_np = hidden_states.detach().cpu().numpy()
            residual_np = residual.detach().cpu().numpy()
            positions_np = positions.detach().cpu().numpy()
        else:
            hidden_np = hidden_states.detach().numpy()
            residual_np = residual.detach().numpy()
            positions_np = positions.detach().numpy()

        # Stack efficiently
        # combined_tensor = np.concatenate([hidden_np.reshape(1, *hidden_np.shape), residual_np.reshape(1, *residual_np.shape)], axis=0)
        # print(f"🔍 Combined tensor: {combined_tensor}", combined_tensor.shape)
        # Normalize to 2D (seq, hidden)
        # hidden_np = hidden_np.reshape(-1, hidden_np.shape[-1])
        # residual_np = residual_np.reshape(-1, residual_np.shape[-1])
        # For decode steps, ensure (1, hidden)
        # if step_idx > 0:
        #     hidden_np = hidden_np[-1:, :]
        #     residual_np = residual_np[-1:, :]
        # Stack along a new axis to form (2, seq_or_1, hidden)

        # print(f"send_ITF - hidden_np shape - {hidden_np.shape}")
        # print(f"send_ITF - residual shape - {residual_np.shape}")

        # combined_tensor = np.stack([hidden_np, residual_np], axis=0)
        # combined_tensor = np.array([hidden_np, residual_np, positions_np])
        combined_tensor = [hidden_np, residual_np, positions_np]
        combined_tensor = pickle.dumps(combined_tensor)
        combined_tensor = np.frombuffer(combined_tensor, dtype=np.uint8)

        # Calculate payload size
        # payload_size_bytes = combined_tensor.nbytes
        # payload_size_mb = payload_size_bytes / (1024 * 1024)

        # print(
        #     f"📊 Payload size for {request_id} step {step_idx}: {payload_size_mb:.2f} MB ({payload_size_bytes:,} bytes)"
        # )
        # print(f"   - Hidden states shape: {hidden_np.shape}, dtype: {hidden_np.dtype}")
        # print(f"   - Residual shape: {residual_np.shape}, dtype: {residual_np.dtype}")
        # print(f"   - Combined tensor shape: {combined_tensor.shape}")

        # Fast send
        await tensor_transport.send(
            next_peer_ticket,
            name=f"{request_id}_step{step_idx}_combined",
            tensor=combined_tensor,
        )

    except Exception as e:
        # Minimal error handling - no printing in hot path
        print(f"send_inference_tensors_fast - Error: {e}")
        pass


async def send_sampler_output(
    tensor_transport: "TensorTransport",
    batch_id: str,
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
            raise ValueError(
                "next_peer_ticket must be provided for tensor transport send."
            )

        # Calculate payload size
        # payload_size_bytes = sampler_output_bytes.nbytes
        # payload_size_mb = payload_size_bytes / (1024 * 1024)

        # print(
        #     f"📊 Sampler output payload size for {request_id} step {step_idx}: {payload_size_mb:.2f} MB ({payload_size_bytes:,} bytes)"
        # )
        # print(
        #     f"   - Sampler output shape: {sampler_output_bytes.shape}, dtype: {sampler_output_bytes.dtype}"
        # )

        # Compose a name for the tensor message
        tensor_name = f"{batch_id}_step{step_idx}_sampler_output"

        await tensor_transport.send(
            next_peer_ticket, name=tensor_name, tensor=sampler_output_bytes
        )

        # print(
        #     f"📤 Sent sampler_output for {request_id} step {step_idx} to {next_peer_id[:8]}..."
        # )
    except Exception as e:
        print(f"❌ Failed to send sampler output for {batch_id} to {next_peer_id}: {e}")
