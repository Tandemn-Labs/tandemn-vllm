import pytest
from src.utils.deployment_utils import create_deployment_instructions

@pytest.fixture
def mock_distribution_plan():
    """Provides a mock distribution plan for a two-peer setup."""
    return {
        "distribution": {
            "peer1": {
                "assigned_layers": 11,
                "handles_embeddings": True,
                "estimated_vram_usage": 4.0
            },
            "peer2": {
                "assigned_layers": 11,
                "handles_embeddings": False,
                "estimated_vram_usage": 3.8
            }
        }
    }

@pytest.fixture
def mock_request():
    """Provides a mock request object."""
    class MockRequest:
        def __init__(self):
            self.model_name = "TestModel/v1"
            self.quantization = "bitsandbytes"
            self.dtype = "float16"
            self.qbits = 4

    return MockRequest()

def test_create_deployment_instructions(mock_distribution_plan, mock_request):
    """
    Tests the creation of deployment instructions for all peers in a plan.
    """
    peer_table = {
        "peer1": "192.168.1.101",
        "peer2": "192.168.1.102"
    }
    server_ip = "192.168.1.100"

    instructions = create_deployment_instructions(
        mock_request,
        mock_distribution_plan,
        peer_table,
        server_ip
    )

    # VERIFY
    assert isinstance(instructions, dict)
    assert len(instructions) == 2
    assert "peer1" in instructions
    assert "peer2" in instructions

    # --- Verify instructions for Peer 1 (First Peer) ---
    p1_instr = instructions["peer1"]
    assert p1_instr["model_name"] == "TestModel/v1"
    assert p1_instr["is_first_peer"] is True
    assert p1_instr["is_last_peer"] is False
    assert p1_instr["assigned_layers"] == list(range(0, 11))
    assert p1_instr["next_peer_ticket"] == "peer2"
    assert p1_instr["pipeline"] == ["peer1", "peer2"]

    # Check required files for the first peer
    assert "embedding/layer.safetensors" in p1_instr["required_files"]
    assert "lm_head/layer.safetensors" not in p1_instr["required_files"]
    assert "layers/layer_0.safetensors" in p1_instr["required_files"]
    assert "layers/layer_10.safetensors" in p1_instr["required_files"]
    assert "layers/layer_11.safetensors" not in p1_instr["required_files"]

    # Check download URL format
    expected_url_prefix = f"http://{server_ip}:8000/download_file/TestModel_v1"
    assert p1_instr["server_download_url"] == expected_url_prefix

    # Check that quantization settings are passed through
    assert p1_instr["quantization"] == "bitsandbytes"
    assert p1_instr["dtype"] == "float16"
    assert p1_instr["qbits"] == 4

    # --- Verify instructions for Peer 2 (Last Peer) ---
    p2_instr = instructions["peer2"]
    assert p2_instr["model_name"] == "TestModel/v1"
    assert p2_instr["is_first_peer"] is False
    assert p2_instr["is_last_peer"] is True
    assert p2_instr["assigned_layers"] == list(range(11, 22))
    assert p2_instr["next_peer_ticket"] is None
    assert p2_instr["pipeline"] == ["peer1", "peer2"]

    # Check required files for the last peer
    assert "embedding/layer.safetensors" not in p2_instr["required_files"]
    assert "lm_head/layer.safetensors" in p2_instr["required_files"]
    assert "norm/layer.safetensors" in p2_instr["required_files"]
    assert "layers/layer_11.safetensors" in p2_instr["required_files"]
    assert "layers/layer_21.safetensors" in p2_instr["required_files"]
    assert "layers/layer_10.safetensors" not in p2_instr["required_files"]

    assert p2_instr["server_download_url"] == expected_url_prefix
    assert p2_instr["quantization"] == "bitsandbytes"
    assert p2_instr["dtype"] == "float16"
    assert p2_instr["qbits"] == 4
