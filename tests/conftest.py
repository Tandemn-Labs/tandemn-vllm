import pytest
import asyncio
import motor.motor_asyncio
import iroh
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock

from src.server import app
from src.config.settings import MONGODB_URI, MONGODB_DB_NAME

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
async def mock_motor_client():
    """Create a mock MongoDB client."""
    client = AsyncMock(spec=motor.motor_asyncio.AsyncIOMotorClient)
    db = AsyncMock()
    client.__getitem__.return_value = db
    return client

@pytest.fixture
async def mock_iroh_node():
    """Create a mock Iroh node."""
    node = AsyncMock(spec=iroh.Iroh)
    doc = AsyncMock()
    node.docs().create.return_value = doc
    node.net().node_id.return_value = "test_peer_id"
    return node

@pytest.fixture
def test_client():
    """Create a FastAPI test client."""
    return TestClient(app)

@pytest.fixture
def sample_metrics():
    """Sample system metrics for testing."""
    return {
        "cpu": 25.5,
        "ram": 60.0,
        "gpu_info": [
            {
                "name": "NVIDIA GeForce RTX 3080",
                "total_vram_gb": 10.0,
                "used_vram_gb": 4.5,
                "free_vram_gb": 5.5,
                "utilization_percent": 75.0,
                "temperature_celsius": 65.0
            }
        ],
        "total_free_vram_gb": 5.5,
        "timestamp": "2024-01-01T00:00:00Z"
    }

@pytest.fixture
def sample_model_config():
    """Sample model configuration for testing."""
    return {
        "vocab_size": 50257,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "num_key_value_heads": 12,
        "intermediate_size": 3072
    } 