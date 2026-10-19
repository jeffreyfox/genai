import argparse
import timeit
import os
import torch
from pathlib import Path
from dataclasses import asdict
import numpy as np
from torch.nn import functional as F

from cs336_basics.optimizer import AdamW
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.config import load_config

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmarking."
    )

    parser.add_argument(
        "--config-path",
        type=str,
        required=True,
        help="Path of the config file",
    )

    parser.add_argument(
        "--warmup-steps",
        type=int,
        required=False,
        default=5,
        help="Number of untimed warm-up steps before measurement",
    )

    parser.add_argument(
        "--num-steps",
        type=int,
        required=False,
        default=None,
        help="Number of timed steps (overrides config value)",
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=False,
        default="f,b,o",
        help="Comma-separated phases to time: f/forward, b/backward, o/optimizer. Every step still runs all three.",
    )

    parser.add_argument(
        "--device",
        type=str,
        required=False,
        default="cpu",
        choices=("cpu", "gpu"),
        help="Device to run on. gpu uses CUDA when available, otherwise MPS.",
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

def get_learning_rate(optimizer: AdamW) -> float:
    return optimizer.param_groups[0]["alpha"]

def get_random_batch(
    vocab_size: int, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    y = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    return x, y

PHASES = ("forward", "backward", "optimizer")
PHASE_ALIASES = {
    "f": "forward",
    "forward": "forward",
    "b": "backward",
    "backward": "backward",
    "o": "optimizer",
    "optimizer": "optimizer",
}

def parse_modes(mode: str) -> set[str]:
    parts = [part.strip() for part in mode.replace(",", " ").split() if part.strip()]
    unknown = [part for part in parts if part not in PHASE_ALIASES]
    if not parts or unknown:
        raise ValueError(
            f"Mode must list one or more of f, b, o, got {mode!r}"
        )
    return {PHASE_ALIASES[part] for part in parts}

def run_step(
    model: BasicsTransformerLM,
    x: torch.Tensor,
    y: torch.Tensor,
    optimizer: AdamW,
    device: torch.device,
    timed: set[str],
) -> dict[str, float]:
    durations = {phase: 0.0 for phase in timed}

    def run(phase: str, fn) -> None:
        if phase not in timed:
            fn()
            synchronize(device)
            return
        synchronize(device)
        start = timeit.default_timer()
        fn()
        synchronize(device)
        durations[phase] += timeit.default_timer() - start

    def forward() -> None:
        nonlocal loss
        loss = F.cross_entropy(model(x).transpose(1, 2), y)

    loss = None
    run("forward", forward)
    model.zero_grad(set_to_none=True)
    run("backward", loss.backward)
    run("optimizer", optimizer.step)
    return durations

def main() -> None:
    args = parse_args()
    
    config_path = Path(args.config_path)

    device = resolve_device(args.device)
    timed = parse_modes(args.mode)
    warmup_steps = args.warmup_steps

    model_config, training_config = load_config(config_path)
    num_steps = (
        args.num_steps
        if args.num_steps is not None
        else training_config.num_steps
    )

    print(
        f"Device: {device}\n"
        f"Timed phases: {', '.join(phase for phase in PHASES if phase in timed)}\n"
        f"Warm-up steps: {warmup_steps}\n"
        f"Timed steps: {num_steps}\n"
        f"Vocab size: {model_config.vocab_size}\n"
        f"Batch size: {training_config.batch_size}\n"
        f"Context length: {training_config.context_length}\n"
        f"Learning rate: {training_config.learning_rate}\n"
    )

    model = BasicsTransformerLM(**asdict(model_config)).to(device)
    model.train()

    optimizer = AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        betas=training_config.betas,
        eps=training_config.eps,
        weight_decay=training_config.weight_decay,
    )

    x, y = get_random_batch(
        vocab_size=model_config.vocab_size,
        batch_size=training_config.batch_size,
        context_length=training_config.context_length,
        device=device,
    )

    print(f"Running {warmup_steps} warmup steps ...")
    for _ in range(warmup_steps):
        run_step(model, x, y, optimizer, device, timed=set())

    print(f"Running {num_steps} timed steps ...")
    samples = {phase: [] for phase in PHASES if phase in timed}
    for _ in range(num_steps):
        durations = run_step(model, x, y, optimizer, device, timed)
        for phase, duration in durations.items():
            samples[phase].append(duration)

    for phase in PHASES:
        if phase not in samples:
            continue
        times = np.array(samples[phase])
        mean_ms = times.mean() * 1000
        std_ms = times.std(ddof=1) * 1000 if num_steps > 1 else 0.0
        print(
            f"{phase}: {num_steps} steps, "
            f"{mean_ms:.1f} ± {std_ms:.1f} ms/step"
        )


if __name__ == "__main__":
    main()
