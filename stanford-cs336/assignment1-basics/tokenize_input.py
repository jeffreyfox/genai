import argparse
import pickle
import time
import torch
from pathlib import Path
from dataclasses import asdict
import numpy as np
from itertools import pairwise
from functools import partial
from multiprocessing import Pool
import shutil


from tokenizer import (
    BPETokenizer,
    get_tokenizer_from_vocab_merges_pickle_path,
)
from model import (
    TransformerLM,
    softmax,
)
from config import (
    load_config,
    ModelConfig,
    TrainingConfig,
)
from cs336_basics.pretokenization_example import find_chunk_boundaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tokenize an input training text into a binary file."
    )

    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Path of the input text file",
    )

    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path of the output text file",
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

    return parser.parse_args()


BUFFER_SIZE = 10_000  # tokens

def tokenize_to_bin(
    chunk: tuple[int, tuple[int, int]],
    input_path: Path,
    output_path: Path,
    tokenizer: BPETokenizer,
) -> None:
    start_time = time.time()
    chunk_id, (start, end) = chunk
    size_in_mb = (end - start) / (1024 * 1024)
    new_output_path = output_path.with_name(
        f"{output_path.stem}_{chunk_id}{output_path.suffix}"
    )
    print(f"Tokenizing chunk {chunk_id}, {size_in_mb:.0f} MB data.")
    with (
        input_path.open("rb") as input_file,
        new_output_path.open("wb") as output_file,
    ):
        start_time = time.time()
        input_file.seek(start)
        text = input_file.read(end - start).decode("utf-8", errors="ignore")
        token_ids = tokenizer.encode(text)
        np.asarray(token_ids, dtype=np.uint16).tofile(output_file)
        end_time = time.time()
        print(f"Finished processing chunk {chunk_id}, tokenized {len(token_ids)} tokens, {size_in_mb:.0f} MB data. Took {end_time-start_time:.1f} s")

def merge_bin_chunks(
    output_path: Path,
    num_chunks: int,
) -> None:
    with output_path.open("wb") as output_file:
        for chunk_id in range(num_chunks):
            chunk_path = output_path.with_name(
                f"{output_path.stem}_{chunk_id}{output_path.suffix}"
            )

            print(f"Appending {chunk_path} into {output_path}")

            with chunk_path.open("rb") as chunk_file:
                shutil.copyfileobj(
                    chunk_file,
                    output_file,
                    length=16 * 1024 * 1024,
                )

def tokenize_input(
    input_path: Path,
    output_path: Path,
    tokenizer: BPETokenizer,
    num_splits: int,
    num_workers: int,
) -> None:
    start = time.time()
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_splits, b"<|endoftext|>")
        boundaries = list(pairwise(boundaries)) # list of (start, end) tuples
    chunks = list(enumerate(boundaries))
    print(f"Generated {len(chunks)} pre-processing chunks")

    tokenize_to_bin_worker = partial(
        tokenize_to_bin,
        input_path=input_path,
        output_path=output_path,
        tokenizer=tokenizer,
    )

    print(f"Started preprocessing {len(chunks)} chunks with {num_workers} workers")
    # Count the token sequence frequency counter in parallel. After this
    # we can only run in serial.
    with Pool(num_workers) as pool:
        pool.map(tokenize_to_bin_worker, chunks)
        
    end = time.time()

def main() -> None:
    args = parse_args()
    
    vocab_path = Path(args.vocab_path)
    merges_path = Path(args.merges_path)
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    num_splits = args.num_splits
    num_workers = args.num_workers
    run_tag = input_path.stem

    tokenizer = get_tokenizer_from_vocab_merges_pickle_path(
        vocab_path=vocab_path,
        merges_path=merges_path,
    )

    tokenize_input(
        input_path=input_path,
        output_path=output_path,
        tokenizer=tokenizer,
        num_splits=num_splits,
        num_workers=num_workers,
    )
    
    merge_bin_chunks(
        output_path=Path(f"data/{run_tag}.bin"),
        num_chunks=num_splits,
    )

if __name__ == "__main__":
    main()
