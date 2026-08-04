import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from config import load_config
from model import TransformerLM, softmax
from optimizer import AdamW
from serialization import load_checkpoint
from tokenizer import get_tokenizer_from_vocab_merges_pickle_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a checkpoint and generate text from a prompt."
    )

    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to a checkpoint file saved by train_llm.py",
    )
    parser.add_argument(
        "--vocab-path",
        type=str,
        required=True,
        help="Path of the vocab file",
    )
    parser.add_argument(
        "--merges-path",
        type=str,
        required=True,
        help="Path of the merges file",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        required=True,
        help="Path of the model config file",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=False,
        default="Once upon a time",
        help="Text prompt to continue",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        required=False,
        default=128,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        required=False,
        default=1.0,
        help="Sampling temperature. Use 0 for greedy decoding.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        required=False,
        default=None,
        help="If set, sample only from the top-k logits",
    )
    parser.add_argument(
        "--device",
        type=str,
        required=False,
        default="cpu",
        help="Device to run inference on",
    )

    return parser.parse_args()


def load_model_from_checkpoint(
    checkpoint_path: Path,
    config_path: Path,
    device: str,
) -> tuple[TransformerLM, int]:
    model_config, training_config = load_config(config_path)
    model = TransformerLM(**asdict(model_config))
    optimizer = AdamW(model.parameters(), lr=training_config.learning_rate)

    iteration = load_checkpoint(checkpoint_path, model, optimizer)
    model.to(device)
    model.eval()

    return model, iteration


def sample_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int | None,
) -> int:
    if temperature == 0:
        return int(torch.argmax(logits).item())

    logits = logits / temperature

    if top_k is not None:
        top_k = min(top_k, logits.shape[-1])
        values, indices = torch.topk(logits, top_k)
        probs = softmax(values, dim=-1)
        choice = torch.multinomial(probs, num_samples=1)
        return int(indices[choice].item())

    probs = softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


@torch.no_grad()
def generate(
    model: TransformerLM,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    context_length: int,
    device: str,
    temperature: float = 1.0,
    top_k: int | None = None,
    eos_token: str = "<|endoftext|>",
) -> str:
    token_ids = tokenizer.encode(prompt)

    eos_ids = tokenizer.encode(eos_token)
    eos_id = eos_ids[0] if len(eos_ids) == 1 else None

    for _ in range(max_new_tokens):
        input_ids = token_ids[-context_length:]
        x = torch.tensor([input_ids], device=device)
        token_positions = torch.arange(len(input_ids), device=device)

        logits = model(x, token_positions)
        next_token = sample_next_token(logits[0, -1], temperature, top_k)
        token_ids.append(next_token)

        if eos_id is not None and next_token == eos_id:
            break

    return tokenizer.decode(token_ids)


def main() -> None:
    args = parse_args()

    checkpoint_path = Path(args.checkpoint_path)
    vocab_path = Path(args.vocab_path)
    merges_path = Path(args.merges_path)
    config_path = Path(args.config_path)

    if not checkpoint_path.exists():
        raise ValueError(f"Checkpoint file does not exist: {checkpoint_path}")

    device = args.device
    model_config, _ = load_config(config_path)

    tokenizer = get_tokenizer_from_vocab_merges_pickle_path(
        vocab_path=vocab_path,
        merges_path=merges_path,
    )
    model, iteration = load_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        device=device,
    )

    print(f"Loaded checkpoint from {checkpoint_path} (iteration {iteration})")
    print(f"Prompt: {args.prompt!r}")

    generated_text = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        context_length=model_config.context_length,
        device=device,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    print("\nGenerated text:\n")
    print(generated_text)


if __name__ == "__main__":
    main()
