"""Compile and run CUDA kernels on a Modal GPU.

From this directory:
  uv run modal run modal_run.py
  uv run modal run modal_run.py --sweep-n
  uv run modal run modal_run.py --sweep-n --gpu L40S
  uv run modal run modal_run.py --kernel mat_mul --gpu T4
  uv run modal run modal_run.py --sweep-n --kernel mat_mul --gpu L40S
"""

from pathlib import Path

import modal

GPU = ["T4", "A10", "L40S", "A100-40GB"]
ROOT = Path(__file__).parent

# Vector length. 3 arrays * 4 bytes * 2^28 ≈ 3 GiB, fits a T4.
SWEEP_N_VEC_ADD = [1 << 16, 1 << 18, 1 << 20, 1 << 22, 1 << 24, 1 << 26, 1 << 28]
# Matrix width. Stops at 4096 so naive O(n^3) stays practical (no --verify).
SWEEP_N_MAT_MUL = [256, 512, 1024, 2048, 4096]

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .entrypoint([])
    .add_local_dir(
        str(ROOT),
        remote_path="/src",
        ignore=[".venv/**", "__pycache__/**", "*.pyc", ".git/**", "*.csv"],
    )
)

app = modal.App("cuda-bench", image=image)


def compile_bench() -> str:
    import subprocess

    binary = "/tmp/bench"
    subprocess.run(
        [
            "nvcc",
            "-O2",
            "-std=c++17",
            "-arch=native",
            "-I/src",
            "/src/apps/bench.cu",
            "/src/kernels/vec_add.cu",
            "/src/kernels/mat_mul.cu",
            "-o",
            binary,
        ],
        check=True,
    )
    return binary


def run_bench(binary: str, args: list[str]) -> str:
    import subprocess

    result = subprocess.run([binary, *args], capture_output=True, text=True)
    if result.stderr:
        print(result.stderr, end="", flush=True)
    if result.returncode != 0:
        if result.stdout:
            print(result.stdout, end="", flush=True)
        result.check_returncode()
    return result.stdout.strip()


@app.function(gpu=GPU, timeout=30 * 60)
def compile_and_run(sweep_n: bool = False, kernel: str = "vec_add"):
    binary = compile_bench()
    if not sweep_n:
        rows = [run_bench(binary, ["--kernel", kernel, "--verify"])]
    else:
        sizes = SWEEP_N_MAT_MUL if kernel == "mat_mul" else SWEEP_N_VEC_ADD
        rows = [run_bench(binary, ["--kernel", kernel, str(n)]) for n in sizes]

    print("COPY INTO GOOGLE SHEETS: select the lines below, paste into A1, then Data > Split text to columns > Comma", flush=True)
    print("gpu,n,grid,block,warmup,iters,time_ms,bandwidth_GBs,gflops_s", flush=True)
    for row in rows:
        print(row.replace("\t", ","), flush=True)


@app.local_entrypoint()
def main(sweep_n: bool = False, gpu: str = "T4", kernel: str = "vec_add"):
    compile_and_run.with_options(gpu=gpu).remote(sweep_n, kernel)
