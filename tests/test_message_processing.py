import pytest
import json
import numpy as np

from src.utils.message_processing import (
    parse_deployment_message,
    parse_inference_trigger_message,
    extract_request_metadata
)

# Helper function to simulate the tensor creation process
def create_tensor_from_payload(payload: dict):
    """Converts a Python dict to a numpy uint8 array, simulating a tensor message."""
    json_str = json.dumps(payload)
    json_bytes = json_str.encode('utf-8')
    return np.frombuffer(json_bytes, dtype=np.uint8)

# --- Tests for parse_deployment_message ---

def test_parse_deployment_message_valid():
    """Tests parsing a valid deployment message."""
    payload = {
        "action": "deploy_model",
        "instructions": {
            "model_name": "TestModel/v1",
            "assigned_layers": [0, 1, 2],
            "required_files": ["file1.safetensors"]
        }
    }
    tensor = create_tensor_from_payload(payload)
    parsed_msg = parse_deployment_message(tensor)

    assert parsed_msg is not None
    assert parsed_msg == payload

def test_parse_deployment_message_invalid_action():
    """Tests that a message with the wrong action is rejected."""
    payload = {
        "action": "wrong_action",
        "instructions": {"model_name": "TestModel/v1"}
    }
    tensor = create_tensor_from_payload(payload)
    assert parse_deployment_message(tensor) is None

def test_parse_deployment_message_missing_instructions():
    """Tests that a deployment message without an 'instructions' key is rejected."""
    payload = {"action": "deploy_model"}
    tensor = create_tensor_from_payload(payload)
    assert parse_deployment_message(tensor) is None

def test_parse_deployment_message_missing_required_field():
    """Tests that a deployment message with missing fields in instructions is rejected."""
    payload = {
        "action": "deploy_model",
        "instructions": {"model_name": "TestModel/v1"} # Missing assigned_layers, etc.
    }
    tensor = create_tensor_from_payload(payload)
    assert parse_deployment_message(tensor) is None

def test_parse_message_not_json():
    """Tests that a tensor containing non-JSON bytes is handled gracefully."""
    tensor = np.frombuffer(b"this is not json", dtype=np.uint8)
    assert parse_deployment_message(tensor) is None
    assert parse_inference_trigger_message(tensor) is None

# --- Tests for parse_inference_trigger_message ---

def test_parse_inference_trigger_message_valid():
    """Tests parsing a valid inference trigger message."""
    payload = {
        "action": "start_inference",
        "request_id": "req_123",
        "pipeline": ["peer1", "peer2"]
    }
    tensor = create_tensor_from_payload(payload)
    parsed_msg = parse_inference_trigger_message(tensor)

    assert parsed_msg is not None
    assert parsed_msg == payload

def test_parse_inference_trigger_invalid_action():
    """Tests that an inference message with the wrong action is rejected."""
    payload = {
        "action": "another_action",
        "request_id": "req_123"
    }
    tensor = create_tensor_from_payload(payload)
    assert parse_inference_trigger_message(tensor) is None

def test_parse_inference_trigger_missing_field():
    """Tests that an inference message with a missing required field is rejected."""
    payload = {"action": "start_inference", "request_id": "req_123"} # Missing pipeline
    tensor = create_tensor_from_payload(payload)
    assert parse_inference_trigger_message(tensor) is None

# --- Tests for extract_request_metadata ---

@pytest.mark.parametrize("name, expected", [
    ("req_abc_step123_combined", ("req_abc", 123, "combined")),
    ("req_with_underscores_1_step4_sampler_output", ("req_with_underscores_1", 4, "sampler_output")),
    ("req_xyz_step0_hidden_state", ("req_xyz", 0, "hidden_state")),
    ("req_123_step99_residual", ("req_123", 99, "residual")),
    ("req_id_with_step_in_it_step5_combined", ("req_id_with_step_in_it", 5, "combined")),
    ("invalid_name_no_step", None),
    ("req_abc_step_not_a_number_combined", None),
])
def test_extract_request_metadata(name, expected):
    """Tests parsing various tensor names for request metadata."""
    assert extract_request_metadata(name) == expected
