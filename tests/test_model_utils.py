import pytest
from src.utils.model_utils import estimate_parameters, estimate_vram
from transformers import AutoConfig

# A known model configuration for consistent testing
TINYLLAMA_CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "hidden_size": 2048,
    "intermediate_size": 5632,
    "num_attention_heads": 32,
    "num_hidden_layers": 22,
    "num_key_value_heads": 4,
    "vocab_size": 32000,
    "head_dim": 64, # Calculated as hidden_size / num_attention_heads
}

@pytest.fixture
def model_config():
    """Provides a mock model configuration for tests."""
    config = AutoConfig.from_dict(TINYLLAMA_CONFIG)
    return config

def test_estimate_parameters(model_config):
    """
    Tests the parameter estimation against a known model's approximate size.
    """
    # This test provides a sanity check but isn't exact, as formulas can be approximations.
    # The real TinyLlama-1.1B has ~1.1B parameters.
    estimated_params = estimate_parameters(model_config.to_dict())

    # Check if the estimation is in a reasonable range (e.g., within 10% of 1.1B)
    assert 1.0e9 < estimated_params < 1.2e9
    # A more precise check based on the formula's output for the given config
    assert estimated_params == 1100092672

def test_estimate_vram_fp16(model_config):
    """
    Tests VRAM estimation for 16-bit precision (FP16).
    """
    total_params = estimate_parameters(model_config.to_dict())

    # For FP16 (16 bits), q_bits = 16
    # Expected VRAM = (total_params / 1e9) * 4 * (16 / 32) * 1.2
    # Expected VRAM = (1.100092672) * 4 * 0.5 * 1.2 = 2.64 GB
    vram_gb = estimate_vram(total_params, q_bits=16)

    assert vram_gb == pytest.approx(2.64, abs=1e-2)

def test_estimate_vram_8bit(model_config):
    """
    Tests VRAM estimation for 8-bit quantization.
    """
    total_params = estimate_parameters(model_config.to_dict())

    # For 8-bit, q_bits = 8
    # Expected VRAM = (1.100092672) * 4 * (8 / 32) * 1.2 = 1.32 GB
    vram_gb = estimate_vram(total_params, q_bits=8)

    assert vram_gb == pytest.approx(1.32, abs=1e-2)

def test_estimate_vram_4bit(model_config):
    """
    Tests VRAM estimation for 4-bit quantization.
    """
    total_params = estimate_parameters(model_config.to_dict())

    # For 4-bit, q_bits = 4
    # Expected VRAM = (1.100092672) * 4 * (4 / 32) * 1.2 = 0.66 GB
    vram_gb = estimate_vram(total_params, q_bits=4)

    assert vram_gb == pytest.approx(0.66, abs=1e-2)

def test_estimate_vram_zero_params():
    """
    Tests VRAM estimation with zero parameters, which should result in zero VRAM.
    """
    vram_gb = estimate_vram(0, q_bits=16)
    assert vram_gb == 0

from src.utils.model_utils import distribute_layers_across_peers

def test_distribute_layers_across_peers_sufficient_vram(model_config):
    """
    Tests the layer distribution logic, accounting for the temporary VRAM hack
    where the peer with the most VRAM is simulated to have 0.25GB.
    """
    # Peer VRAM in GB. peer1 has the most, but will be treated as having 0.25GB.
    # This makes peer2 the most capable for the distribution calculation.
    peers_vram = {
        "peer1_16gb": 16.0,
        "peer2_8gb": 8.0,
        "peer3_4gb": 4.0,
    }

    config_dict = model_config.to_dict()
    q_bits = 4  # 4-bit quantization

    distribution_plan = distribute_layers_across_peers(config_dict, peers_vram, q_bits)

    # VERIFY
    assert distribution_plan["can_fit_model"] is True
    assert distribution_plan["utilized_peers"] == 2
    assert distribution_plan["remaining_layers"] == 0

    dist = distribution_plan["distribution"]
    assert "peer1_16gb" in dist
    assert "peer2_8gb" in dist
    assert "peer3_4gb" not in dist  # Peer 3 should not be used

    # Based on the logic trace:
    # Peer1 (hacked to 0.25GB) gets embeddings and 10 layers.
    assert dist["peer1_16gb"]["assigned_layers"] == 10
    assert dist["peer1_16gb"]["handles_embeddings"] is True

    # Peer2 gets the remaining 12 layers.
    assert dist["peer2_8gb"]["assigned_layers"] == 12
    assert dist["peer2_8gb"]["handles_embeddings"] is False

    total_assigned = sum(p["assigned_layers"] for p in dist.values())
    assert total_assigned == config_dict["num_hidden_layers"]

def test_distribute_layers_insufficient_vram(model_config):
    """
    Tests layer distribution when there is not enough VRAM across all peers.
    Also covers the side-effect of the hack where a low-VRAM peer gets a boost.
    """
    # peer1_vlow has 0.1GB, but the hack will treat it as having 0.25GB.
    peers_vram = {
        "peer1_vlow": 0.1,
        "peer2_vlow": 0.05,
    }

    config_dict = model_config.to_dict()
    # Use 16-bit to make the model larger and ensure it fails to fit
    q_bits = 16

    distribution_plan = distribute_layers_across_peers(config_dict, peers_vram, q_bits)

    assert distribution_plan["can_fit_model"] is False
    assert distribution_plan["remaining_layers"] > 0

    # Check that peer1_vlow was assigned 1 layer due to the VRAM boost from the hack
    dist = distribution_plan["distribution"]
    assert "peer1_vlow" in dist
    assert dist["peer1_vlow"]["assigned_layers"] == 1
    assert dist["peer1_vlow"]["handles_embeddings"] is True
