import argparse
import timeit
import os
import torch
from pathlib import Path
from dataclasses import asdict, replace
import numpy as np
from torch.nn import functional as F

from cs336_basics.optimizer import AdamW
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.config import load_config

SCRIPT_DIR = Path(__file__).resolve().parent

def resolve_script_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return SCRIPT_DIR / candidate

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmarking."
    )

    parser.add_argument(
        "--config-path",
        type=str,
        required=True,
        help="Config file. A relative path is resolved inside cs336_systems.",
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
        help="Comma-separated phases to run and time: f/forward, b/backward, o/optimizer. Backward requires forward.",
    )

    parser.add_argument(
        "--device",
        type=str,
        required=False,
        default="cpu",
        choices=("cpu", "gpu"),
        help="Device to run on. gpu uses CUDA when available, otherwise MPS.",
    )

    parser.add_argument(
        "--compile",
        type=str,
        required=False,
        default="none",
        choices=("none", "graph"),
        help="none runs the model eagerly. graph wraps it with torch.compile.",
    )

    parser.add_argument(
        "--context-length",
        type=int,
        required=False,
        default=None,
        help="Overrides the config context length for the model and the batch.",
    )

    parser.add_argument(
        "--memory-snapshot-dir",
        type=str,
        required=False,
        default=None,
        help="Directory for the CUDA memory snapshot. A relative path is inside cs336_systems. The filename is <config>-<stages>-<context_length>.pickle.",
    )

    return parser.parse_args()


def snapshot_filename(config_path: Path, mode: str, context_length: int) -> str:
    stages = mode.replace(",", "").replace(" ", "")
    return f"{config_path.stem}-{stages}-{context_length}.pickle"

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
    phases = {PHASE_ALIASES[part] for part in parts}
    if "backward" in phases and "forward" not in phases:
        raise ValueError(f"backward requires forward in the same step, got {mode!r}")
    return phases

def run_step(
    model: BasicsTransformerLM,
    x: torch.Tensor,
    y: torch.Tensor,
    optimizer: AdamW,
    device: torch.device,
    phases: set[str],
    *,
    time_it: bool,
) -> dict[str, float]:
    durations = {phase: 0.0 for phase in phases}

    def run(phase: str, fn) -> None:
        if not time_it:
            fn()
            synchronize(device)
            return
        synchronize(device)
        start = timeit.default_timer()
        fn()
        synchronize(device)
        durations[phase] += timeit.default_timer() - start

    if "forward" in phases:
        def forward() -> None:
            nonlocal loss
            loss = F.cross_entropy(model(x).transpose(1, 2), y)

        loss = None
        run("forward", forward)

    if "backward" in phases:
        model.zero_grad(set_to_none=True)
        run("backward", loss.backward)

    if "optimizer" in phases:
        run("optimizer", optimizer.step)

    return durations

def main() -> None:
    args = parse_args()
    
    config_path = resolve_script_path(args.config_path)

    device = resolve_device(args.device)
    timed = parse_modes(args.mode)
    warmup_steps = args.warmup_steps

    model_config, training_config = load_config(config_path)
    if args.context_length is not None:
        model_config = replace(model_config, context_length=args.context_length)
        training_config = replace(training_config, context_length=args.context_length)
    num_steps = (
        args.num_steps
        if args.num_steps is not None
        else training_config.num_steps
    )

    print(
        f"Device: {device}\n"
        f"Phases: {', '.join(phase for phase in PHASES if phase in timed)}\n"
        f"Warm-up steps: {warmup_steps}\n"
        f"Timed steps: {num_steps}\n"
        f"Vocab size: {model_config.vocab_size}\n"
        f"Batch size: {training_config.batch_size}\n"
        f"Context length: {training_config.context_length}\n"
        f"Learning rate: {training_config.learning_rate}\n"
        f"Compile: {args.compile}\n"
    )

    model = BasicsTransformerLM(**asdict(model_config)).to(device)
    model.train()
    if args.compile == "graph":
        model = torch.compile(model)

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
        run_step(model, x, y, optimizer, device, timed, time_it=False)

    if device.type == "cuda":
        torch.cuda.memory._record_memory_history(max_entries=1000000)

    print(f"Running {num_steps} timed steps ...")
    samples = {phase: [] for phase in PHASES if phase in timed}
    for _ in range(num_steps):
        durations = run_step(model, x, y, optimizer, device, timed, time_it=True)
        for phase, duration in durations.items():
            samples[phase].append(duration)

    if device.type == "cuda":
        snapshot_dir = (
            SCRIPT_DIR
            if args.memory_snapshot_dir is None
            else resolve_script_path(args.memory_snapshot_dir)
        )
        snapshot_path = snapshot_dir / snapshot_filename(
            config_path, args.mode, training_config.context_length
        )
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        torch.cuda.memory._dump_snapshot(str(snapshot_path))
        torch.cuda.memory._record_memory_history(enabled=None)
        print(f"Wrote CUDA memory snapshot to {snapshot_path}")

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
