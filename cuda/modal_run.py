"""Compile and run CUDA kernels on a Modal GPU.

From this directory:
  just run-matmul 4096 T4 true 16
  just sweep-matmul n 256,512,1024 T4 false
  just sweep-matmul tile 4,8,16,32 T4 true 4096
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
            "/src/kernels/sum_red.cu",
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


def validate_options(kernel: str, sweep: str, tiled: bool, block_size: int, n: int):
    if kernel not in ("vec_add", "mat_mul", "sum_red"):
        raise ValueError("--kernel must be vec_add, mat_mul, or sum_red")
    if kernel == "sum_red" and sweep:
        raise ValueError("sum_red does not support sweeps")
    if sweep not in ("", "n", "tile"):
        raise ValueError("--sweep must be n or tile")
    if sweep != "n" and n <= 0:
        raise ValueError("--n must be positive")
    if tiled and kernel != "mat_mul":
        raise ValueError("--tiled requires --kernel mat_mul")
    if kernel == "mat_mul" and not 1 <= block_size <= 32:
        raise ValueError("--block-size must be in [1, 32]")
    if sweep == "tile":
        if kernel != "mat_mul" or not tiled:
            raise ValueError("--sweep tile requires --kernel mat_mul --tiled")
    elif tiled and block_size not in (4, 8, 16, 32):
        raise ValueError("--block-size must be 4, 8, 16, or 32")


def parse_sweep_values(sweep: str, values: str) -> list[int]:
    if not values:
        return []
    if not sweep:
        raise ValueError("--values requires --sweep")
    if "=" in values:
        axis, values = values.split("=", 1)
        if axis != sweep:
            raise ValueError("Values prefix must match --sweep")
    try:
        parsed = [int(value.strip()) for value in values.split(",")]
    except ValueError:
        raise ValueError("--values must contain comma-separated integers") from None
    if any(value <= 0 for value in parsed):
        raise ValueError("Sweep values must be positive")
    if sweep == "tile" and any(value not in (4, 8, 16, 32) for value in parsed):
        raise ValueError("Tile sweep values must be 4, 8, 16, or 32")
    return parsed


def validate_run_shape(kernel: str, sweep: str, m: int, k: int):
    if m != -1 or k != -1:
        if kernel != "mat_mul" or sweep:
            raise ValueError("--m and --k are only supported for single matmul runs; sweeps stay square")
        if (m != -1 and m <= 0) or (k != -1 and k <= 0):
            raise ValueError("--m and --k must be positive")


@app.function(gpu=GPU, timeout=30 * 60)
def compile_and_run(sweep: str = "", kernel: str = "vec_add", tiled: bool = False, block_size: int = 32, n: int = -1, values: str = "", m: int = -1, k: int = -1):
    if n == -1:
        n = 256 if kernel == "mat_mul" else 1 << 20
    validate_options(kernel, sweep, tiled, block_size, n)
    validate_run_shape(kernel, sweep, m, k)
    sweep_values = parse_sweep_values(sweep, values)
    binary = compile_bench()
    if sweep == "tile":
        configs = [(n, tile) for tile in (sweep_values or (4, 8, 16, 32))]
    elif sweep == "n":
        sizes = sweep_values or (SWEEP_N_MAT_MUL if kernel == "mat_mul" else SWEEP_N_VEC_ADD)
        configs = [(size, block_size) for size in sizes]
    else:
        configs = [(n, block_size)]

    rows = []
    for size, tile in configs:
        args = ["--kernel", kernel, str(size)]
        if kernel == "mat_mul":
            args += ["--block-size", str(tile)]
            if not sweep:
                args += ["--m", str(size if m == -1 else m), "--k", str(size if k == -1 else k)]
        if tiled:
            args += ["--tiled"]
        if not sweep:
            args += ["--verify"]
        rows.append(run_bench(binary, args))

    print("COPY INTO GOOGLE SHEETS: select the lines below, paste into A1, then Data > Split text to columns > Comma", flush=True)
    print("gpu,m,n,k,grid,block,tiled,block_size,avg_time_ms,gflops_s", flush=True)
    for row in rows:
        fields = row.split("\t")
        row_block_size = fields[5].split("x")[0]
        fields[6:6] = [str(tiled), row_block_size]
        print(",".join(fields), flush=True)


@app.local_entrypoint()
def main(sweep: str = "", gpu: str = "T4", kernel: str = "vec_add", tiled: bool = False, block_size: int = 32, n: int = -1, values: str = "", m: int = -1, k: int = -1):
    if n == -1:
        n = 256 if kernel == "mat_mul" else 1 << 20
    validate_options(kernel, sweep, tiled, block_size, n)
    validate_run_shape(kernel, sweep, m, k)
    parse_sweep_values(sweep, values)
    compile_and_run.with_options(gpu=gpu).remote(sweep, kernel, tiled, block_size, n, values, m, k)
