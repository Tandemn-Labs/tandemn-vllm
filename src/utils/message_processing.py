"""
Message Processing Utilities

Common functions for handling tensor-based messages in the distributed inference system.
Eliminates code duplication across message handlers and centralizes error handling.
"""

import json
from typing import Any, Dict, Optional, Tuple

import numpy as np


class MessageProcessingError(Exception):
    """Custom exception for message processing errors"""

    pass


def convert_tensor_to_bytes(tensor) -> bytes:
    """
    Convert tensor (PyTorch or NumPy) to bytes for JSON parsing.

    Args:
        tensor: Input tensor (torch.Tensor or np.ndarray)

    Returns:
        bytes: Raw bytes from tensor

    Raises:
        MessageProcessingError: If tensor has wrong dtype or conversion fails
    """
    try:
        # Handle PyTorch tensor
        if hasattr(tensor, "numpy"):
            arr = tensor.numpy()
        else:
            # Already numpy array
            arr = tensor

        # Validate dtype
        if arr.dtype != np.uint8:
            raise MessageProcessingError(f"Expected uint8 tensor, got {arr.dtype}")

        return arr.tobytes()

    except Exception as e:
        raise MessageProcessingError(f"Failed to convert tensor to bytes: {e}")


def parse_json_from_tensor(tensor) -> Dict[str, Any]:
    """
    Convert tensor to JSON data with validation.

    Args:
        tensor: Input tensor containing JSON data as bytes

    Returns:
        Dict[str, Any]: Parsed JSON data

    Raises:
        MessageProcessingError: If parsing fails
    """
    try:
        tensor_bytes = convert_tensor_to_bytes(tensor)
        json_str = tensor_bytes.decode("utf-8")
        return json.loads(json_str)
    except UnicodeDecodeError as e:
        raise MessageProcessingError(f"Failed to decode tensor bytes as UTF-8: {e}")
    except json.JSONDecodeError as e:
        raise MessageProcessingError(f"Failed to parse JSON from tensor: {e}")
    except Exception as _:
        raise


def extract_request_metadata(name: str) -> Optional[Tuple[str, int, str]]:
    """
    Parse request metadata from tensor names.

    Handles formats like:
    - {request_id}_step{step_idx}_combined
    - {request_id}_step{step_idx}_sampler_output

    Args:
        name: Tensor name string

    Returns:
        Tuple[str, int, str] | None: (request_id, step_idx, message_type) or None if invalid
    """
    try:
        # Find the last occurrence of "_step" to handle request_ids with underscores
        step_marker_idx = name.rfind("_step")
        if step_marker_idx == -1:
            return None

        # Extract request_id (everything before the last _step)
        request_id = name[:step_marker_idx]

        # Extract the step number and message type
        rest = name[step_marker_idx + 5 :]  # Skip "_step"
        parts = rest.split("_")

        if not parts or not parts[0].isdigit():
            return None

        step_idx = int(parts[0])

        # Determine message type
        if "_combined" in name:
            message_type = "combined"
        elif "_sampler_output" in name:
            message_type = "sampler_output"
        elif "_hidden_state" in name:
            message_type = "hidden_state"
        elif "_residual" in name:
            message_type = "residual"
        else:
            message_type = "unknown"

        return request_id, step_idx, message_type

    except (ValueError, IndexError) as e:
        print(f"⚠️ Failed to parse tensor name '{name}': {e}")
        return None


def validate_deployment_message(msg: Dict[str, Any]) -> bool:
    """
    Validate deployment message format.

    Args:
        msg: Deployment message to validate

    Returns:
        bool: True if valid deployment message
    """

    instructions = msg.get("instructions", {})
    required_fields = ["model_name", "assigned_layers", "required_files"]

    return all(field in instructions for field in required_fields)


def validate_inference_trigger_message(msg: Dict[str, Any]) -> bool:
    """
    Validate inference trigger message format.

    Args:
        msg: Inference trigger message to validate

    Returns:
        bool: True if valid inference trigger message
    """

    required_fields = ["batch_id", "pipeline"]
    return all(field in msg for field in required_fields)


def validate_request_message(msg: Dict[str, Any]) -> bool:
    required_fields = [
        "request_id",
        "prompt",
        "model",
        "sampling_params",
        "max_batch_size",
    ]
    return all(field in msg for field in required_fields)


def validate_dispatch_message(msg: Dict[str, Any]) -> bool:
    required_fields = ["batch_id", "request_id"]
    return all(field in msg for field in required_fields)


def log_message_received(message_type: str, msg: Dict[str, Any], extra_info: str = ""):
    """
    Standardized logging for received messages.

    Args:
        message_type: Type of message (deployment, inference_trigger, etc.)
        msg: Parsed message content
        extra_info: Additional info to log
    """
    print(f"\n{'=' * 80}")
    print(f"📨 {message_type.upper()} MESSAGE RECEIVED")
    print(f"🔍 Action: {msg.get('action', 'unknown')}")

    if extra_info:
        print(f"ℹ️  {extra_info}")

    print(f"{'=' * 80}\n")


# Convenience functions for common parsing patterns
def parse_deployment_message(tensor) -> Optional[Dict[str, Any]]:
    """Parse and validate deployment message"""
    msg = parse_json_from_tensor(tensor)
    if msg and validate_deployment_message(msg):
        return msg
    return None


def parse_inference_trigger_message(tensor) -> Optional[Dict[str, Any]]:
    """Parse and validate inference trigger message"""
    msg = parse_json_from_tensor(tensor)
    if msg and validate_inference_trigger_message(msg):
        return msg
    return None


def parse_request_message(tensor) -> Optional[Dict[str, Any]]:
    """Parse and validate incoming request message"""
    msg = parse_json_from_tensor(tensor)
    if msg and validate_request_message(msg):
        return msg
    return None


def parse_dispatch_message(tensor) -> Optional[Dict[str, Any]]:
    """Parse and validate incoming batch dispatch message"""
    msg = parse_json_from_tensor(tensor)
    if msg and validate_dispatch_message(msg):
        return msg
    return None
