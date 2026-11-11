"""Compile and run vec_add.cu on a Modal GPU.

From this directory:
  uv run modal run modal_run.py
  uv run modal run modal_run.py --sweep-n
  uv run modal run modal_run.py --sweep-n --gpu L40S
"""

from pathlib import Path

import modal

GPU = ["T4", "A10", "L40S", "A100-40GB"]
SRC = Path(__file__).parent / "vec_add.cu"

# Sweeps stay in a range T4 can hold (3 arrays * 4 bytes * 2^28 ≈ 3 GiB).
SWEEP_N = [1 << 16, 1 << 18, 1 << 20, 1 << 22, 1 << 24, 1 << 26, 1 << 28]

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .entrypoint([])
    .add_local_file(str(SRC), remote_path="/src/vec_add.cu")
)

app = modal.App("vec-add", image=image)


def compile_vec_add() -> str:
    import subprocess

    binary = "/tmp/vec_add"
    subprocess.run(
        [
            "nvcc",
            "-O2",
            "-std=c++17",
            "-arch=native",
            "/src/vec_add.cu",
            "-o",
            binary,
        ],
        check=True,
    )
    return binary


def run_vec_add(binary: str, args: list[str]) -> str:
    import subprocess

    result = subprocess.run(
        [binary, *args], check=True, capture_output=True, text=True
    )
    if result.stderr:
        print(result.stderr, end="", flush=True)
    return result.stdout.strip()


@app.function(gpu=GPU, timeout=30 * 60)
def compile_and_run(sweep_n: bool = False):
    binary = compile_vec_add()
    if not sweep_n:
        rows = [run_vec_add(binary, ["--verify"])]
    else:
        rows = [run_vec_add(binary, [str(n)]) for n in SWEEP_N]

    print("COPY INTO GOOGLE SHEETS: select the lines below, paste into A1, then Data > Split text to columns > Comma", flush=True)
    print("gpu,n,n_threads,n_blocks,warmup,iters,time_ms,bandwidth_GBs,gflops_s", flush=True)
    for row in rows:
        print(row.replace("\t", ","), flush=True)


@app.local_entrypoint()
def main(sweep_n: bool = False, gpu: str = "T4"):
    compile_and_run.with_options(gpu=gpu).remote(sweep_n)
