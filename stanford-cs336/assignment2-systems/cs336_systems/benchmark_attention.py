import argparse
import itertools
import timeit
from collections.abc import Callable

import numpy as np
import torch

from cs336_basics.model import scaled_dot_product_attention

BATCH_SIZE = 8
# D_MODELS = (16, 32, 64, 128)
# SEQUENCE_LENGTHS = (256, 1024, 4096, 8192, 16384)
D_MODELS = (16,32)
SEQUENCE_LENGTHS = (256,1024)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark scaled dot-product attention.")
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=5,
        help="Untimed forward and backward steps before measurement",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=100,
        help="Timed forward passes and timed backward passes",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=("cpu", "gpu"),
        help="Device to run on. gpu uses CUDA when available, otherwise MPS.",
    )
    parser.add_argument(
        "--compile",
        type=str,
        default="none",
        choices=("none", "graph"),
        help="none runs attention eagerly. graph wraps it with torch.compile.",
    )
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "gpu":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("GPU requested but neither CUDA nor MPS is available")
    raise ValueError(f"Device must be 'cpu' or 'gpu', got {device_name!r}")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def memory_mib(device: torch.device) -> float | None:
    if device.type == "cuda":
        return torch.cuda.memory_allocated() / (1024**2)
    if device.type == "mps":
        return torch.mps.current_allocated_memory() / (1024**2)
    return None


def clear_grads(*tensors: torch.Tensor) -> None:
    for tensor in tensors:
        if tensor.grad is not None:
            tensor.grad.zero_()


def summarize(times: list[float]) -> tuple[float, float]:
    samples = np.array(times)
    mean_ms = samples.mean() * 1000
    std_ms = samples.std(ddof=1) * 1000 if len(samples) > 1 else 0.0
    return mean_ms, std_ms


def benchmark_pair(
    d_model: int,
    seq_len: int,
    device: torch.device,
    warmup_steps: int,
    num_steps: int,
    attention: Callable,
) -> tuple[float, float, float, float, float | None]:
    q = torch.randn(BATCH_SIZE, seq_len, d_model, device=device, requires_grad=True)
    k = torch.randn(BATCH_SIZE, seq_len, d_model, device=device, requires_grad=True)
    v = torch.randn(BATCH_SIZE, seq_len, d_model, device=device, requires_grad=True)
    grad_out = torch.ones(BATCH_SIZE, seq_len, d_model, device=device)
    out = None

    try:
        for _ in range(warmup_steps):
            out = attention(q, k, v)
            synchronize(device)
            out.backward(grad_out)
            synchronize(device)
            clear_grads(q, k, v)

        forward_times = []
        for _ in range(num_steps):
            synchronize(device)
            start = timeit.default_timer()
            out = attention(q, k, v)
            synchronize(device)
            forward_times.append(timeit.default_timer() - start)

        before_backward_mib = memory_mib(device)

        backward_times = []
        for step in range(num_steps):
            if step > 0:
                out = attention(q, k, v)
            synchronize(device)
            start = timeit.default_timer()
            out.backward(grad_out)
            synchronize(device)
            backward_times.append(timeit.default_timer() - start)
            clear_grads(q, k, v)

        forward_mean, forward_std = summarize(forward_times)
        backward_mean, backward_std = summarize(backward_times)
        return forward_mean, forward_std, backward_mean, backward_std, before_backward_mib
    finally:
        del q, k, v, grad_out, out
        if device.type == "cuda":
            torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print(
        f"Device: {device}\n"
        f"Batch size: {BATCH_SIZE}\n"
        f"Warm-up steps: {args.warmup_steps}\n"
        f"Timed steps: {args.num_steps}\n"
        f"Compile: {args.compile}\n"
    )

    attention = scaled_dot_product_attention
    if args.compile == "graph":
        attention = torch.compile(attention)

    for d_model, seq_len in itertools.product(D_MODELS, SEQUENCE_LENGTHS):
        label = f"d_model={d_model} seq_len={seq_len}"
        try:
            forward_mean, forward_std, backward_mean, backward_std, memory = benchmark_pair(
                d_model, seq_len, device, args.warmup_steps, args.num_steps, attention
            )
        except torch.OutOfMemoryError:
            print(f"{label} out of memory")
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue

        memory_text = "n/a" if memory is None else f"{memory:.1f} MiB"
        print(
            f"{label}\n"
            f"  forward: {args.num_steps} steps, {forward_mean:.2f} ± {forward_std:.2f} ms/step\n"
            f"  backward: {args.num_steps} steps, {backward_mean:.2f} ± {backward_std:.2f} ms/step\n"
            f"  memory before backward: {memory_text}\n"
        )


if __name__ == "__main__":
    main()
