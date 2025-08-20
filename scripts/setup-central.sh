curl -LsSf https://astral.sh/uv/install.sh | sh

export PATH="~/.local/bin:$PATH"
export PATH="~/tandemn-vllm/scripts"

chmod +x ~/tandemn-vllm/scripts/*

uv venv -p 3.12
source .venv/bin/activate

uv pip install -r requirements.server.txt
