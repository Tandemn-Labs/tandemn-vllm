import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# Server settings
SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))

# MongoDB settings
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "iroh_tandem")

# Model settings
DEFAULT_QBITS = int(os.getenv("DEFAULT_QBITS", "16"))
DEFAULT_CONFIG_FILENAME = os.getenv("DEFAULT_CONFIG_FILENAME", "config.json")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")

# Peer cleanup settings
PEER_CLEANUP_THRESHOLD = int(os.getenv("PEER_CLEANUP_THRESHOLD", "24"))  # hours

# GPU metrics settings
GPU_METRICS_INTERVAL = int(os.getenv("GPU_METRICS_INTERVAL", "60"))  # seconds 