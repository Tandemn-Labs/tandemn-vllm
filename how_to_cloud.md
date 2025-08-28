# How to installation for LMCache Branch 


Get AWS Machine First
Run these

```
#!/usr/bin/env bash
# 01_gpu_prereqs_minimal.sh
set -euo pipefail

sudo apt update
sudo apt install -y \
  build-essential git curl wget pkg-config \
  python3 python3-venv python3-pip python3-dev \
  software-properties-common

# Graphics drivers PPA
sudo add-apt-repository -y ppa:graphics-drivers/ppa
sudo apt update

# NVIDIA driver 535 (without DKMS for cloud instances)
sudo apt install -y nvidia-driver-535

# CUDA toolkit for nvcc
sudo apt install -y nvidia-cuda-toolkit

sudo apt -y full-upgrade # bad practice, but works for now
echo "Rebooting..."
sudo reboot
```

Then after rebooting

```
git clone https://github.com/Tandemn-Labs/tandemn-vllm.git
git checkout lmcache
cd ./tandemn_vllm/experimental_code
git clone https://github.com/LMCache/LMCache.git
cd LMCache
git checkout v0.2.1
cd ../../..
```

Now populate your .env

Then after that install uv

```
curl -LsSf https://astral.sh/uv/install.sh | sh
cd tandemn-vllm
uv venv
source .venv/bin/activate

# Ensure we don't keep CUDA libs in runtime path (prevents NVML mismatch)
unset LD_LIBRARY_PATH || true
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu

uv pip install -r requirements.txt
uv pip uninstall -y nvidia-ml-py3 || true
```

Then build lmcache

```
cd ./tandemn_vllm/experimental_code/lmcache
if ! command -v nvcc >/dev/null 2>&1; then
  echo "ERROR: nvcc not found; install nvidia-cuda-toolkit." >&2
  exit 1
fi
NVCC_PATH="$(readlink -f "$(command -v nvcc)")"
CUDA_HOME="$(dirname "$(dirname "$NVCC_PATH")")"
export CUDA_HOME TORCH_CUDA_HOME="$CUDA_HOME"
export CUDACXX="$CUDA_HOME/bin/nvcc" NVCC="$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"

python setup.py clean --all || true
rm -f lmcache/c_ops*.so || true
rm -rf build *.egg-info lmcache/**/__pycache__ || true

uv pip install -e . --no-build-isolation --no-deps -v

python - <<'PY'
import torch, lmcache
import lmcache.c_ops as _c
print("LMCache OK | torch:", torch.__version__, "| CUDA:", torch.version.cuda)
PY
cd ../..


echo "Ready:"
echo "  uvicorn src.server:app --host 0.0.0.0 --port 8000"
echo "  python -m src.machine_runner"
```





