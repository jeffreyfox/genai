import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    context_length: int
    d_model: int
    d_ff: int
    rope_theta: float
    num_layers: int
    num_heads: int


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int
    context_length: int
    learning_rate: float
    num_steps: int
    weight_decay: float


def load_config(
    path: Path,
) -> tuple[ModelConfig, TrainingConfig]:
    with path.open("rb") as file:
        raw = tomllib.load(file)

    return (
        ModelConfig(**raw["model"]),
        TrainingConfig(**raw["training"]),
    )