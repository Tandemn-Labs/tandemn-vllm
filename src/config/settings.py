import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Server settings
SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))

# MongoDB settings
MONGODB_URI = os.getenv(
    "MONGODB_URI",
    "mongodb+srv://choprahetarth:helloworld@demo-day.tjaxr2t.mongodb.net/?retryWrites=true&w=majority&appName=demo-day",
)
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "iroh_tandemn")

# Model settings
DEFAULT_QBITS = int(os.getenv("DEFAULT_QBITS", "16"))
DEFAULT_CONFIG_FILENAME = os.getenv("DEFAULT_CONFIG_FILENAME", "config.json")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")

# Peer cleanup settings
PEER_CLEANUP_THRESHOLD = int(
    os.getenv("PEER_CLEANUP_THRESHOLD", "1")
)  # hours -> 1 hour instead of 24

# GPU metrics settings
GPU_METRICS_INTERVAL = int(os.getenv("GPU_METRICS_INTERVAL", "60"))  # seconds

# S3 settings
S3_SHARDS_BASE = os.getenv("S3_SHARDS_BASE", "")
S3_BUCKET = os.getenv("S3_BUCKET", "")
S3_REGION = os.getenv("S3_REGION", "")
