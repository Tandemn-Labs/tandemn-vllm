import asyncio
import pickle
import threading
import time
import uuid
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
# the above is just a payload that is sent from one peer to another.
#
# INFERENCE_CONTEXT (what it is and how it is used):
# ---------------------------------------------------
# Purpose:
# - A per-request scratchpad for cross-peer hand-off of intermediate tensors.
# - It stores step-scoped payloads that allow a downstream stage (peer) to pick
#   up exactly the right data at exactly the right time.
#
# Shape/keys:
# - Top-level key: request_id (str)
# - Second-level key: step index (stored as str for consistency with JSON names)
# - Leaf dict keys:
#     * "hidden_state": tensor with shape [seq_len, hidden_dim] (prompt) or [1, hidden_dim] (decode)
#     * "residual": tensor matching the hidden_state shape
#     * "sampler_output": pickled SamplerOutput (only present on the first peer when the last peer sends it)
#
# Producers (writers):
# - handle_inference_data_message in machine_runner.py
#     * When receiving "..._combined" messages, it unpacks and writes
#       {hidden_state, residual} for a specific (request_id, step).
#     * When receiving "..._sampler_output" messages, it unpickles and writes
#       {sampler_output} for the (request_id, step) the first peer is waiting on.
#
# Consumers (readers):
# - pre_hook (non-first peers): waits on STEP_EVENTS[req][step], then reads
#   {hidden_state, residual} to feed its layers.
# - sampler_post_hook (first peer): waits on STEP_EVENTS_SAMPLER[req][step], then
#   reads {sampler_output} to continue the engine’s control flow.
#
# Synchronization:
# - Read/writes are protected by CONTEXT_LOCK.
# - Arrival is coordinated by two distinct event maps:
#     * STEP_EVENTS → payload (hidden_state/residual) readiness
#     * STEP_EVENTS_SAMPLER → sampler_output readiness (from last peer)
#
# Lifecycle/cleanup:
# - Per-step garbage collection: once a step is consumed and we advance, we drop
#   step-1 data to avoid unbounded growth.
# - Per-request cleanup: cleanup_request_context(request_id) removes the entire
#   request subtree and both event maps, called on completion or timeout.
#
# ========== LOCKING MODEL (READ ME) ==========
# We use three distinct synchronization primitives to keep concurrency sane:
#
# 1) CONTEXT_LOCK (global, RLock):
#    - Protects access to INFERENCE_CONTEXT and step-event dicts when we create,
#      read, or delete per-request/step payloads (hidden/residual/sampler_output)
#    - Use when: storing tensors, reading tensors for a given step, or cleaning up
#      per-request maps to avoid races with other threads/hooks.
#
# 2) context_lock (local to register_inference_hooks, RLock):
#    - Protects the hook_contexts dict (per-execution request metadata) and any
#      updates to fields like current_step, active, next_peer_ticket, etc.
#    - Hooks may run from different threads; this lock ensures we don't read a
#      partially-updated context or increment steps concurrently.
#    - RLock (re-entrant) is used to avoid deadlocks when the same thread needs
#      to re-enter protected sections during nested calls.
#
# 3) INFERENCE_MUTEX (global, Lock):
#    - Serializes start_inference_run per peer. This avoids multiple concurrent
#      hook contexts being attached at the same time on a single peer, which
#      previously led to misrouting and timeouts. This is a temporary safety
#      measure until hooks are made fully request-aware and can run concurrently.
#
# Step events (STEP_EVENTS vs STEP_EVENTS_SAMPLER):
# - STEP_EVENTS: signals arrival of payload tensors (hidden_state/residual)
# - STEP_EVENTS_SAMPLER: signals arrival of the last peer's sampler output
# Keeping these separate prevents accidental cross-wakeup (payload setting the
# sampler wait, or vice versa).
CONTEXT_LOCK = threading.RLock()  # Thread-safe access to INFERENCE_CONTEXT

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


def create_virtual_sampler_output(request_id: str, step_idx: int) -> Any:
    """
    Create a virtual/mock SamplerOutput that satisfies vLLM's expectations
    for intermediate peers that don't actually need the real sampler state.

    This is a lightweight placeholder that contains minimal required fields.
    """
    try:
        from vllm.model_executor.layers.sampler import SamplerOutput
        from vllm.sequence import CompletionSequenceGroupOutput, Logprob, SequenceOutput

        # Choose a dummy token id and provide a matching logprob entry
        output_token_id = 1

        # minimal mock SequenceOutput
        mock_sequence_output = SequenceOutput(
            parent_seq_id=0,
            output_token=output_token_id,
            logprobs={output_token_id: Logprob(logprob=0.0)},
        )

        # minimal mock CompletionSequenceGroupOutput
        mock_completion_output = CompletionSequenceGroupOutput(
            samples=[mock_sequence_output],
            prompt_logprobs=None,
        )

        # virtual SamplerOutput with just the required outputs field
        virtual_sampler_output = SamplerOutput(
            outputs=[mock_completion_output],
        )

        print(
            f"🎭 Created virtual SamplerOutput for intermediate peer (request: {request_id}, step: {step_idx})"
        )
        return virtual_sampler_output

    except Exception as e:
        print(f"❌ Failed to create virtual SamplerOutput: {e}")
        return None


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

    # Per-batch context storage, maps uuid to dict of metadata
    batch_metadata = {}

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
            active_contexts = [
                ctx for ctx in batch_metadata.values() if ctx.get("active", False)
            ]
            if not active_contexts:
                return args

            hook_context = active_contexts[0]  # ?
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
        event = STEP_EVENTS[batch_id].setdefault(current_step, threading.Event())
        if not event.wait(timeout=1000.0):
            cleanup_request_context(batch_id)
            return args

        # Direct memory access (minimal locking)
        with CONTEXT_LOCK:
            step_data = INFERENCE_CONTEXT[batch_id][str(current_step)]
            hidden_states = step_data["hidden_state"]
            residual = step_data["residual"]

        positions = args[0]
        device = positions.device

        # Infer payload hidden size and keep it in context for visibility
        payload_hidden_size = int(hidden_states.shape[-1])
        with context_lock:
            if hook_context.get("hidden_size") is None:
                hook_context["hidden_size"] = payload_hidden_size
                # print(f"🔧 Inferred hidden size from payload: {payload_hidden_size}")
            elif hook_context["hidden_size"] != payload_hidden_size:
                pass
                # print(
                #     f"⚠️ Hidden size mismatch: model={hook_context['hidden_size']} payload={payload_hidden_size}. Using payload size."
                # )

        # Single conditional for step - ultra optimized reshaping
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
            active_contexts = [
                ctx for ctx in batch_metadata.values() if ctx.get("active", False)
            ]
            if not active_contexts:
                return

            hook_context = active_contexts[0]  # ?

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
        asyncio.run_coroutine_threadsafe(
            send_inference_tensors_fast(
                node,
                request_id,
                next_peer_id,
                hidden_states.clone(),  # Send torch tensor directly
                residual.clone(),  # Send torch tensor directly
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
            active_contexts = [
                ctx for ctx in batch_metadata.values() if ctx.get("active", False)
            ]
            if not active_contexts:
                print("❌ No active inference context found in sampler_post_hook")
                return output

            hook_context = active_contexts[0]
            batch_id = hook_context["batch_id"]
            current_step = hook_context.get("current_step", 0)
            is_last_peer = hook_context.get("is_last_peer", False)
            pipeline = hook_context["pipeline"]
            peer_id = hook_context["peer_id"]

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

            # 🚀 OPTIMIZATION: Only send to FIRST peer instead of all peers
            # Find the first peer in the pipeline
            first_peer_ticket = pipeline[0] if pipeline else None

            if first_peer_ticket and first_peer_ticket != peer_id:
                print(
                    f"📤 Sending sampler output ONLY to FIRST peer: {first_peer_ticket[:8]}..."
                )
                asyncio.run_coroutine_threadsafe(
                    send_sampler_output(
                        node,
                        batch_id,
                        first_peer_ticket,
                        sampler_output_np,
                        step_idx=current_step,
                        next_peer_ticket=first_peer_ticket,
                    ),
                    asyncio_loop,
                )
                print(
                    f"✅ Optimized: Sent sampler output only to FIRST peer (not {len(pipeline) - 1} peers)"
                )
            else:
                # print(f"⚠️ No valid first peer found or first peer is self")
                pass

            # Decode tokens from number
            token_numbers = [
                [completion.samples[0].output_token] for completion in output.outputs
            ]
            tokens_str = tokenizer.batch_decode(token_numbers)
            # print(f"sampler-post-hook - tokens_str {tokens_str}")
            asyncio.run_coroutine_threadsafe(
                stream_token_to_server(
                    batch_id=batch_id, tokens=tokens_str, server_url=server_url
                ),
                asyncio_loop,
            )

            # Increment step for last peer too
            with context_lock:
                hook_context["current_step"] = current_step + 1

            # Clean up old data to prevent memory growth
            with CONTEXT_LOCK:
                if current_step > 0:
                    INFERENCE_CONTEXT[batch_id].pop(str(current_step - 1), None)
                    STEP_EVENTS[batch_id].pop(current_step - 1, None)
                    STEP_EVENTS_SAMPLER[batch_id].pop(current_step - 1, None)

            # print("sampler-post-hook: last-peer, returned same sampler")
            return output

        # If first peer
        elif pipeline.index(peer_id) == 0 if peer_id in pipeline else -1:
            # FIRST peer: Wait for real sampler output from last peer
            print(
                f"⏳ FIRST peer waiting for REAL sampler output for step {current_step}"
            )

            # Wait for sampler output using threading.Event, note that it's indexed by current_step before increment
            event = STEP_EVENTS_SAMPLER[batch_id].setdefault(
                current_step, threading.Event()
            )

            if not event.wait(timeout=1000.0):
                cleanup_request_context(batch_id)
                raise RuntimeError(
                    f"Timeout waiting for sampler output for {batch_id} step {current_step}"
                )

            # Data is ready!
            with CONTEXT_LOCK:
                # print(f"🔍 INFERENCE_CONTEXT: {INFERENCE_CONTEXT[request_id][str(current_step)]}, for step {current_step}")
                received_output = INFERENCE_CONTEXT[batch_id][str(current_step)][
                    "sampler_output"
                ]
                print(
                    f"✅ FIRST peer received REAL sampler output for step {current_step}"
                )

            # Clean up consumed data to prevent memory growth
            with CONTEXT_LOCK:
                if current_step > 0:
                    INFERENCE_CONTEXT[batch_id].pop(str(current_step - 1), None)
                    STEP_EVENTS[batch_id].pop(current_step - 1, None)
                    STEP_EVENTS_SAMPLER[batch_id].pop(current_step - 1, None)

            # Increment step after receiving
            with context_lock:
                hook_context["current_step"] = current_step + 1

            print(f"sampler-post-hook: first-peer, received sampler {received_output}")
            return received_output

        # Middle peer
        else:
            # INTERMEDIATE peer: Use virtual sampler output (no waiting!)
            print(
                f"🚀 INTERMEDIATE peer ({pipeline.index(peer_id) if peer_id in pipeline else -1}) using VIRTUAL sampler output - NO WAITING!"
            )

            # Create virtual sampler output
            virtual_output = create_virtual_sampler_output(batch_id, current_step)

            if virtual_output is None:
                print("⚠️ Failed to create virtual output, falling back to waiting...")
                # Fallback to old behavior if virtual creation fails
                event = STEP_EVENTS_SAMPLER[batch_id].setdefault(
                    current_step, threading.Event()
                )
                if not event.wait(timeout=1000.0):
                    cleanup_request_context(batch_id)
                    raise RuntimeError(
                        f"Timeout waiting for sampler output for {batch_id} step {current_step}"
                    )

                with CONTEXT_LOCK:
                    received_output = INFERENCE_CONTEXT[batch_id][str(current_step)][
                        "sampler_output"
                    ]

                with context_lock:
                    hook_context["current_step"] = current_step + 1

                return received_output

            # Clean up old data to prevent memory growth
            # print("sampler-post-hook - Attempting to grab CONTEXT_LOCK")
            with CONTEXT_LOCK:
                if current_step > 0:
                    INFERENCE_CONTEXT[batch_id].pop(str(current_step - 1), None)
                    STEP_EVENTS[batch_id].pop(current_step - 1, None)
                    STEP_EVENTS_SAMPLER[batch_id].pop(current_step - 1, None)

            # Increment step immediately (no waiting!)
            # print("sampler-post-hook - Attempting to grab context_lock")
            with context_lock:
                hook_context["current_step"] = current_step + 1

            # print("sampler-post-hook: middle-peer, returned same sampler")
            return output

    def start_inference_run(
        batch_id: str,
        pipeline: List[str],
        input_text: List[str],
        sampling_params: Any,
    ):
        """The main inference runner"""
        # Generate unique execution ID to avoid collisions
        execution_id = str(uuid.uuid4())[:8]  # ? Need to find where this is used

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
                    batch_metadata[execution_id] = {
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
                model_hidden = batch_metadata[execution_id].get("hidden_size")
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

                # If last peer, send final result to server
                if is_last and completions:
                    final_text = [comp.outputs[0].text for comp in completions]
                    print(f"Final text (before sending) - len({len(final_text)})")

                    for i in range(len(final_text)):
                        print(f"\nOutput number {i}\n", "+" * 50)
                        print(f"{final_text[i]}\n")

                    try:
                        # Use a unique sent flag per request to avoid collisions
                        request_sent_key = f"final_sent_{batch_id}"
                        with context_lock:
                            if not batch_metadata[execution_id].get(
                                request_sent_key, False
                            ):
                                batch_metadata[execution_id][request_sent_key] = True
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
                import traceback

                traceback.print_exc()
            finally:
                # Mark context inactive early to avoid further hook activity for this execution
                with context_lock:
                    if execution_id in batch_metadata:
                        batch_metadata[execution_id]["active"] = False

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
                        if execution_id in batch_metadata:
                            del batch_metadata[execution_id]

                # Always clean up per-request transport/context
                cleanup_request_context(batch_id)

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
            raise ValueError(
                "next_peer_ticket must be provided for tensor transport send."
            )

        # Calculate payload size
        # payload_size_bytes = data_to_send.nbytes
        # payload_size_mb = payload_size_bytes / (1024 * 1024)

        tensor_type = "residual" if is_residual else "hidden_state"
        # print(
        #     f"📊 {tensor_type.capitalize()} tensor payload size for {request_id} step {step_idx}: {payload_size_mb:.2f} MB ({payload_size_bytes:,} bytes)"
        # )
        # print(f"   - Tensor shape: {data_to_send.shape}, dtype: {data_to_send.dtype}")

        # Compose a name for the tensor message
        tensor_name = f"{request_id}_step{step_idx}_{tensor_type}"

        await tensor_transport.send(
            next_peer_ticket, name=tensor_name, tensor=data_to_send
        )

        # print(
        #     f"📤 Sent {tensor_type} tensor for {request_id} to {next_peer_id} ({next_peer_ticket}) via TensorTransport"
        # )
    except Exception as e:
        print(
            f"❌ [DEBUG] Failed to send hidden state tensor for {request_id} to {next_peer_id} ({next_peer_ticket}): {e}"
        )


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
            raise ValueError(
                "next_peer_ticket must be provided for tensor transport send."
            )

        # Stack both tensors together - tensor-iroh handles serialization
        # Format: [hidden_states, residual] concatenated along a new dimension
        combined_tensor = np.stack([hidden_states, residual], axis=0)

        # Calculate payload size
        # payload_size_bytes = combined_tensor.nbytes
        # payload_size_mb = payload_size_bytes / (1024 * 1024)

        # print(
        #     f"📊 Payload size for {request_id} step {step_idx}: {payload_size_mb:.2f} MB ({payload_size_bytes:,} bytes)"
        # )
        # print(
        #     f"   - Hidden states shape: {hidden_states.shape}, dtype: {hidden_states.dtype}"
        # )
        # print(f"   - Residual shape: {residual.shape}, dtype: {residual.dtype}")
        # print(f"   - Combined tensor shape: {combined_tensor.shape}")

        # Compose a name for the tensor message
        tensor_name = f"{request_id}_step{step_idx}_combined"

        await tensor_transport.send(
            next_peer_ticket, name=tensor_name, tensor=combined_tensor
        )

        print(
            f"📤 Sent combined tensors for {request_id} step {step_idx} to {next_peer_id} via TensorTransport"
        )
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
        print("send_inference_tensors_fast - beginning of fn")
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
        else:
            hidden_np = hidden_states.detach().numpy()
            residual_np = residual.detach().numpy()

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

        combined_tensor = np.stack([hidden_np, residual_np], axis=0)
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
        tensor_name = f"{request_id}_step{step_idx}_sampler_output"

        await tensor_transport.send(
            next_peer_ticket, name=tensor_name, tensor=sampler_output_bytes
        )

        # print(
        #     f"📤 Sent sampler_output for {request_id} step {step_idx} to {next_peer_id[:8]}..."
        # )
    except Exception as e:
        print(
            f"❌ Failed to send sampler output for {request_id} to {next_peer_id}: {e}"
        )
