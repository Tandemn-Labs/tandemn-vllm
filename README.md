# Tandemn vLLM

A distributed inference system that splits large language model layers across multiple machines using Iroh for peer-to-peer communication, with vLLM powering each peer's computation.

## Features

- Distributed model layer sharding across peer machines
- OpenAI-compatible `/v1/chat/completions` endpoint
- Real-time system metrics monitoring (CPU, RAM, GPU via pynvml)
- Automatic peer discovery and heartbeat management
- Model parameter and VRAM requirement estimation
- MongoDB-based metrics storage and deployment tracking
- Batched and streaming inference support
- S3-compatible pre-signed URL uploads for result transfer

## Prerequisites

- Python 3.8+
- MongoDB
- NVIDIA GPU with CUDA support (peers only)
- [Iroh](https://iroh.computer/) node

## Installation

Clone the repository:

```bash
git clone https://github.com/Tandemn-Labs/tandemn-vllm.git
cd tandemn-vllm
```

Install dependencies for the server:

```bash
pip install -r requirements.server.txt
```

Install dependencies for each peer machine:

```bash
pip install -r requirements.peer.txt
```

Set up environment variables (or create a `.env` file):

```bash
export MONGODB_URI="mongodb://localhost:27017"
export MONGODB_DB_NAME="iroh_tandem"
export HUGGINGFACE_TOKEN="your_huggingface_token"
```

## Usage

Start the central server:

```bash
python src/server.py
```

On each peer machine, run:

```bash
python src/machine_runner.py
```

Peers will register themselves with the server via heartbeat. The server handles layer distribution, job scheduling, and result aggregation.

## API Endpoints

### Cluster Management
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check and status of all connected peers |
| `POST` | `/heartbeat` | Peer registration and heartbeat |
| `GET` | `/metrics/{peer_id}` | Metrics history for a specific peer |
| `GET` | `/deployments` | List all active deployments |

### Model Deployment
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/estimate_model` | Estimate model parameters and VRAM requirements |
| `POST` | `/identify_peers` | Find suitable peers for a given model |
| `POST` | `/calculate_peer_layers` | Compute layer distribution across peers |
| `POST` | `/distribute_model_layers` | Push model layer shards to peers |
| `POST` | `/create_layer_shards_safetensors` | Create safetensors shards for distribution |
| `GET` | `/peer_layer_capacity/{peer_id}` | Get a peer's available layer capacity |
| `POST` | `/deploy` | Start a full deployment job |
| `GET` | `/deployment_status/{model_name}` | Poll deployment progress |
| `POST` | `/deployment_complete` | Notify server that deployment finished |
| `GET` | `/task_status/{task_id}` | Get status of a background task |

### Inference
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat completions |
| `POST` | `/infer` | Single inference request |
| `POST` | `/streaming` | Streaming inference |
| `POST` | `/infer_batched` | Batched inference |
| `POST` | `/completion` | Raw completion endpoint |
| `GET` | `/get_results/{task_id}` | Retrieve results for a completed task |
| `GET` | `/batch_task_status/{task_id}` | Poll status of a batched task |
| `POST` | `/batch_failed` | Report a failed batch |
| `POST` | `/batch_info` | Submit batch metadata |
| `POST` | `/batch_upload_complete` | Notify server that batch upload finished |

### File Transfer
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET/HEAD` | `/download_file/{model_name}/{file_path}` | Download a model shard file |

## tensor-iroh

Tensor passing between pipeline stages is powered by **[tensor-iroh](https://github.com/Tandemn-Labs/tensor-iroh)**, a Rust-backed Python library that sends PyTorch tensors directly between peers over QUIC using the Iroh protocol.

Each peer starts a `PyTensorNode`, gets a shareable ticket (node address), and uses that ticket to send/receive tensors without routing through the server. This keeps activation traffic off the coordinator and on a direct peer-to-peer path.

Install it alongside the other peer dependencies:

```bash
pip install tensor-iroh
```

See [Tandemn-Labs/tensor-iroh](https://github.com/Tandemn-Labs/tensor-iroh) for build instructions, protocol details, and the Rust source.

## Architecture

```
Client
  └── Server (src/server.py)
        ├── Peer 1 (src/machine_runner.py) — layers 0–N
        ├── Peer 2 (src/machine_runner.py) — layers N–M
        └── Peer 3 (src/machine_runner.py) — layers M–end
```

- **Server** — coordinates deployments, routes inference requests, tracks peer health via heartbeats, stores metrics in MongoDB
- **Peers** — run vLLM on assigned model layers, stream activations to the next peer in the pipeline
- **Iroh** — handles peer-to-peer transport for tensor passing between pipeline stages

## License

MIT License
