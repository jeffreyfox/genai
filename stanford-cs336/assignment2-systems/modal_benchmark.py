# Profile the benchmark on a Modal GPU with Nsight Systems.

import modal

app = modal.App("cs336-benchmark")

volume = modal.Volume.from_name("cs336-data")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("wget", "ca-certificates", "gnupg")
    .run_commands(
        "wget -qO- https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub | gpg --dearmor -o /usr/share/keyrings/nvidia-devtools-keyring.gpg",
        "echo 'deb [signed-by=/usr/share/keyrings/nvidia-devtools-keyring.gpg] https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64/ /' > /etc/apt/sources.list.d/nvidia-devtools.list",
        "apt-get update",
        "apt-get install -y nsight-systems-cli",
    )
    # uv sync installs the editable path dependency declared in uv.lock.
    .add_local_dir("cs336-basics", remote_path="/.uv/cs336-basics", copy=True)
    .uv_sync()
    .add_local_dir(
        ".",
        remote_path="/root/cs336",
        ignore=[
            ".venv/**",
            ".git/**",
            "**/__pycache__/**",
            "*.pyc",
            "tests/**",
            "*.nsys-rep",
        ],
    )
)


def benchmark_args(
    num_steps: int,
    warmup_steps: int,
    mode: str,
    config_path: str = "small.toml",
    compile: str = "none",
    context_length: int = 512,
) -> list[str]:
    return [
        "python",
        "benchmark.py",
        f"--config-path={config_path}",
        "--device=gpu",
        f"--num-steps={num_steps}",
        f"--warmup-steps={warmup_steps}",
        f"--mode={mode}",
        f"--compile={compile}",
        f"--context-length={context_length}",
        "--memory-snapshot-dir=/workspace",
    ]


@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 30,
    volumes={"/workspace": volume},
)
def run_benchmark(
    num_steps: int = 10,
    warmup_steps: int = 5,
    mode: str = "f,b,o",
    config_path: str = "small.toml",
    compile: str = "none",
    context_length: int = 512,
):
    import subprocess

    subprocess.run(
        benchmark_args(num_steps, warmup_steps, mode, config_path, compile, context_length),
        cwd="/root/cs336/cs336_systems",
        check=True,
    )
    volume.commit()


@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 30,
    volumes={"/workspace": volume},
)
def profile(
    num_steps: int = 10,
    warmup_steps: int = 5,
    mode: str = "f,b,o",
    compile: str = "none",
    context_length: int = 512,
):
    import subprocess

    subprocess.run(
        [
            "nsys",
            "profile",
            "--force-overwrite=true",
            "--stats=true",
            "-o",
            "/workspace/benchmark",
            "--",
            *benchmark_args(
                num_steps,
                warmup_steps,
                mode,
                compile=compile,
                context_length=context_length,
            ),
        ],
        cwd="/root/cs336/cs336_systems",
        check=True,
    )
    volume.commit()


@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 60,
)
def run_attention_benchmark(
    num_steps: int = 100,
    warmup_steps: int = 5,
    compile: str = "none",
):
    import subprocess

    subprocess.run(
        [
            "python",
            "benchmark_attention.py",
            "--device=gpu",
            f"--num-steps={num_steps}",
            f"--warmup-steps={warmup_steps}",
            f"--compile={compile}",
        ],
        cwd="/root/cs336/cs336_systems",
        check=True,
    )


@app.local_entrypoint()
def attention(
    num_steps: int = 100,
    warmup_steps: int = 5,
    compile: str = "none",
    gpu: str = "L40S",
):
    run_attention_benchmark.with_options(gpu=gpu).remote(num_steps, warmup_steps, compile)


@app.local_entrypoint()
def benchmark(
    num_steps: int = 10,
    warmup_steps: int = 5,
    mode: str = "f,b,o",
    config_path: str = "small.toml",
    compile: str = "none",
    context_length: int = 512,
    gpu: str = "L40S",
):
    run_benchmark.with_options(gpu=gpu).remote(
        num_steps, warmup_steps, mode, config_path, compile, context_length
    )


@app.local_entrypoint()
def main(
    num_steps: int = 10,
    warmup_steps: int = 5,
    mode: str = "f,b,o",
    compile: str = "none",
    context_length: int = 512,
    gpu: str = "L40S",
):
    profile.with_options(gpu=gpu).remote(
        num_steps, warmup_steps, mode, compile, context_length
    )
