#!/bin/bash
# Install vLLM into the project's worker venv (made by `uv sync --frozen` at launch) and fetch Qwen3.5-9B.
# vLLM is not in pyproject.toml: it pins its own torch, and the coordinator has no GPU. vLLM 0.32.0 is the newest
# release with a CUDA 12.9 build (0.33.0 ships CUDA 13.0 only, which needs a newer host driver than Runpod
# hosts reliably have). Run every later command with `uv run --no-sync`, or uv would remove vLLM again.
set -euo pipefail
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
nvidia-smi | grep -o "CUDA Version: [0-9.]*" || true
uv pip install "vllm==0.32.0+cu129" --extra-index-url https://wheels.vllm.ai/0.32.0/cu129 --torch-backend=cu129 2>&1 | tail -n 5
uv run --no-sync python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, torch.version.cuda, torch.cuda.is_available())"
uv run --no-sync python -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3.5-9B'))"
