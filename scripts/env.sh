# Source before any uv command: the network volume is near quota, so the venv and cache live on container disk.
export UV_CACHE_DIR=/root/.cache/uv-sc
export UV_PROJECT_ENVIRONMENT=/root/venvs/swarm-cavity
export UV_LINK_MODE=copy
