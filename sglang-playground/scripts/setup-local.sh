#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
    echo "Local MLX mode requires an Apple Silicon Mac." >&2
    exit 1
fi
# Keep the source checkout: the environment installs it in editable mode.
if [[ ! -d .local/sglang ]]; then
    mkdir -p .local
    git clone --depth 1 https://github.com/sgl-project/sglang.git .local/sglang
fi
uv venv --python 3.12 --allow-existing .venv-local
cp .local/sglang/python/pyproject_other.toml .local/sglang/python/pyproject.toml
# Text-only MLX serving does not need the optional Rust extensions.
SGLANG_BUILD_RUST_EXTS=none uv pip install --python .venv-local/bin/python -e '.local/sglang/python[srt_mps]'
.venv-local/bin/python -c 'import torch, mlx.core as mx; assert torch.backends.mps.is_available(), "MPS unavailable"; print("MPS available; MLX device:", mx.default_device())'
echo "Installed source revision:"
git -C .local/sglang rev-parse HEAD
echo "Ready. Run just serve, then just chat in another terminal."
