# script to train the model on modal

import modal

app = modal.App("cs336-training")

volume = modal.Volume.from_name("cs336-data")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync()
    .add_local_dir(
        ".",
        remote_path="/root/cs336",
        ignore=[
            ".venv/**",
            ".git/**",
            "checkpoints/*.pt",
            "data/**",
            "wandb/**",
            "__pycache__/**",
            "cs336_basics/__pycache__/**",
            "*.pyc",
            "*.pkl",
            "tests/**",
        ],
    )
)

# Verify the GPU is available
# @app.function(
#     image=image,
#     gpu="L40S",
#     timeout=60 * 10,
#     volumes={"/data": volume},
# )
# def train():
#     import torch

#     print("CUDA:", torch.cuda.is_available())
#     print("GPU:", torch.cuda.get_device_name(0))
#     print("CUDA version:", torch.version.cuda)


@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 60 * 6,
    volumes={
        "/workspace": volume,
    },
    secrets=[
        modal.Secret.from_name("wandb-secret")
    ],
)
def train():
    import subprocess

    subprocess.run(
    [
        "python",
        "-u",
        "/root/cs336/train_llm.py",
        "--dataset-name=owt",
        "--vocab-path=data/vocab_owt.pkl",
        "--merges-path=data/merges_owt.pkl",
        "--config-path=/root/cs336/configs/train_llm_ts.toml",
        "--num-steps=100",
        "--wandb-log-interval=10",
        "--print-interval=10",
        "--eval-interval=10",
        "--checkpoint-interval=100",
        "--device=cuda",
    ],
    cwd="/workspace",
    check=True,
)


@app.local_entrypoint()
def main():
    train.remote()