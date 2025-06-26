# Iroh Tandemn

A distributed computing system that enables collaborative model training and inference across multiple machines using Iroh for peer-to-peer communication.

## Features

- Distributed computation pipeline using Iroh for peer-to-peer communication
- Real-time system metrics monitoring (CPU, RAM, GPU)
- Automatic peer discovery and management
- Model parameter and VRAM requirement estimation
- MongoDB-based metrics storage and peer tracking

## Prerequisites

- Python 3.8+
- MongoDB
- NVIDIA GPU with CUDA support (for GPU metrics)
- Iroh node

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/iroh-tandem.git
cd iroh-tandem
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables (or create a .env file):
```bash
export MONGODB_URI="mongodb://localhost:27017"
export MONGODB_DB_NAME="iroh_tandem"
export HUGGINGFACE_TOKEN="your_huggingface_token"
```

## Usage

1. Start the central server:
```bash
python src/server.py
```

2. On each peer machine, run:
```bash
python src/machine_runner.py
```

The server will provide a ticket that needs to be shared with all peer machines.

## API Endpoints

- `GET /health` - Check health and status of all connected machines
- `GET /metrics/{peer_id}` - Get metrics history for a specific peer
- `GET /ticket` - Retrieve the document sharing ticket
- `POST /estimate_model` - Estimate model parameters and VRAM requirements
- `POST /identify_peers` - Identify suitable peers for running a model
- `POST /start_job` - Initiate a distributed computation job
- `GET /result` - Retrieve the final computation result

## Architecture

The system consists of:
- A central server that coordinates the distributed computation
- Multiple peer machines that perform the actual computation
- MongoDB for storing metrics and peer information
- Iroh for peer-to-peer communication

## License

MIT License 