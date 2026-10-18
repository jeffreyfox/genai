import argparse
import time
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
        "--print-interval",
        type=int,
        required=False,
        default=100,
        help="Interval to print training metrics to stdout",
    )

    parser.add_argument(
        "--num-steps",
        type=int,
        required=False,
        default=None,
        help="Number of training steps (overrides config value)",
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

def get_learning_rate(optimizer: AdamW) -> float:
    return optimizer.param_groups[0]["alpha"]

def get_random_batch(
    vocab_size: int, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    y = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    return x, y

def main() -> None:
    args = parse_args()
    
    config_path = Path(args.config_path)

    print_interval = args.print_interval
    device = resolve_device(args.device)

    model_config, training_config = load_config(config_path)
    num_steps = (
        args.num_steps
        if args.num_steps is not None
        else training_config.num_steps
    )

    print(
        f"Device: {device}\n"
        f"Steps: {num_steps}\n"
        f"Vocab size: {model_config.vocab_size}\n"
        f"Batch size: {training_config.batch_size}\n"
        f"Context length: {training_config.context_length}\n"
        f"Learning rate: {training_config.learning_rate}\n"
        f"Print interval: {print_interval}\n"
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

    train_start_time = time.time()

    for step in range(num_steps):
        x, y = get_random_batch(
            vocab_size=model_config.vocab_size,
            batch_size=training_config.batch_size,
            context_length=training_config.context_length,
            device=device,
        )

        logits = model(x)
        train_loss = F.cross_entropy(logits.transpose(1, 2), y)

        optimizer.zero_grad(set_to_none=True)
        train_loss.backward()
        optimizer.step()

        step_num = step + 1

        if step_num % print_interval == 0:
            print(
                f"Step {step_num}: "
                f"training loss = {train_loss.item():.3f}"
            )
            
    train_end_time = time.time()
    print(
        f"Trained for {num_steps} steps, "
        f"total training time: {train_end_time - train_start_time:.1f} s",
    )


if __name__ == "__main__":
    main()
