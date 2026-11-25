# Source before any uv command. The venv lives on the container disk (fast imports; `uv sync` rebuilds it after a
# restart); the uv cache lives on the network volume (the coordinator's shared default), since the container disk is
# only 20 GB. UV_LINK_MODE=copy keeps the venv independent of the cache across the two filesystems.
export UV_CACHE_DIR=/workspace/.cache/uv
export UV_PROJECT_ENVIRONMENT=/root/venvs/swarm-cavity
export UV_LINK_MODE=copy
