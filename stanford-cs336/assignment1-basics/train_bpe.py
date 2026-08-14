import argparse
import pickle
import time
from pathlib import Path

from tokenizer import train_bpe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a BPE tokenizer."
    )

    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Path to the training text.",
    )

    parser.add_argument(
        "--vocab-size",
        type=int,
        required=True,
        help="Target vocabulary size.",
    )

    parser.add_argument(
        "--num-splits",
        type=int,
        required=False,
        default=1,
        help="Number of splits for the input training text",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        required=False,
        default=1,
        help="Number of parallel workers in the pre-tokenzation step.",
    )

    parser.add_argument(
        "--sequence-freq-input-path",
        type=str,
        required=False,
        default=None,
        help="Input path to load sequence freq file.",
    )

    parser.add_argument(
        "--sequence-freq-output-path",
        type=str,
        required=False,
        default=None,
        help="Output path to load sequence freq file.",
    )

    parser.add_argument(
        "--print-interval",
        type=int,
        required=False,
        default=1000,
        help="print interval for the merging process",
    )

    return parser.parse_args()

def save_vocab(
    vocab: dict[int, bytes],
    output_path: Path,
) -> None:
    with output_path.open("wb") as file:
        pickle.dump(
            vocab,
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

def save_merges(
    merges: list[tuple[bytes, bytes]],
    output_path: Path,
) -> None:
    with output_path.open("wb") as file:
        pickle.dump(
            merges,
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

def main() -> None:
    args = parse_args()

    input_path = args.input_path
    vocab_size = args.vocab_size
    special_tokens = ["<|endoftext|>"]
    num_splits = args.num_splits
    num_workers = args.num_workers
    sequence_freq_input_path = Path(args.sequence_freq_input_path) if args.sequence_freq_input_path else None
    sequence_freq_output_path = Path(args.sequence_freq_output_path) if args.sequence_freq_output_path else None
    print_interval = args.print_interval

    print(f"Training BPE on input: {input_path}, vocab_size: {vocab_size}, num_splits: {num_splits}, num_workers: {num_workers}, print_interval: {print_interval}")
    start = time.time()
    vocab, merges = train_bpe(
        input_path,
        vocab_size,
        special_tokens,
        num_splits,
        num_workers,
        sequence_freq_input_path,
        sequence_freq_output_path,
        print_interval,
    )
    end = time.time()
    print(f"Training took {end-start:.0f} seconds")
    print(f"Vocab size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")
    tag = ""
    if "TinyStoriesV2" in input_path:
        tag = "ts"
    elif "owt" in input_path:
        tag = "owt"
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = data_dir / f"vocab_{tag}.pkl"
    merges_path = data_dir / f"merges_{tag}.pkl"
    save_vocab(vocab, vocab_path)
    save_merges(merges, merges_path)
    print(f"Vocab and merges saved to {vocab_path} and {merges_path}")
    

if __name__ == "__main__":
    main()
