import pickle
import re
import time
from collections import Counter
from collections.abc import Iterable, Iterator
from functools import partial
from itertools import pairwise
from multiprocessing import Pool
from pathlib import Path

import regex

from cs336_basics.pretokenization_example import find_chunk_boundaries

type Token = int
type TokenPair = tuple[Token, Token]
type TokenSequence = tuple[Token, ...]

PATTERN = regex.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

# int => bytes. Keep the first 256 characters's mapping the same as unicode chr/ord. And then append special token.
def init_vocab(special_tokens: list[str]) -> dict[int, bytes]:
    vocab: dict[int, bytes] = {}
    for i in range(256):
        vocab[i] = bytes([i]) # use the raw byte value, not encoded by utf-8
    for i, token in enumerate(special_tokens):
        vocab[i+256] = token.encode("utf-8")
    # print(f"{vocab=}")
    return vocab


def get_sequence_freq(
    chunk: tuple[int, tuple[int, int]],
    input_path: str,
    special_tokens_pattern: str,
) -> Counter[TokenSequence]:
    start_time = time.time()
    chunk_id, (start, end) = chunk
    size_in_mb = (end - start) / (1024 * 1024)
    with open(input_path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore")
    
        sequence_freq = Counter()
        # split by <|endoftext|>, we get individual text chunks
        pieces = [p for p in re.split(special_tokens_pattern, text) if p]
        for p in pieces:
            for match in regex.finditer(PATTERN, p):
                token_bytes = match.group(0).encode("utf-8") # bytes
                # Note here we assume that the mapping from unmerged bytes (0-255) maps to the integer in vocab
                # in the same way as ord/chr.
                sequence_freq[tuple(token_bytes)] += 1
    end_time = time.time()
    print(f"Finished pre-processing chunk {chunk_id}, {start=}, {end=}, size={size_in_mb:.1f} MB. Took {end_time-start_time:.1f} s.")
    return sequence_freq

# Reconstructs pair frequency dict from sequence frequency dict
def get_pair_freq(sequence_freq: Counter[TokenSequence]) -> Counter[TokenPair]:
    pair_freq = Counter()
    for seq, freq in sequence_freq.items():
        for ch1, ch2 in pairwise(seq):
            pair_freq[(ch1, ch2)] += freq
    return pair_freq

def get_pair_to_seqs(sequence_freq: Counter[TokenSequence]) -> dict[TokenPair, set[TokenSequence]]:
    result = {}
    for seq in sequence_freq:
        for ch1, ch2 in pairwise(seq):
            result.setdefault((ch1, ch2), set()).add(seq)
    return result

def merge(seq: TokenSequence, pair: TokenPair, new_idx: int) -> TokenSequence:
    new_seq = []
    i = 0
    while i < len(seq):
        if i < len(seq) - 1 and seq[i] == pair[0] and seq[i+1] == pair[1]:
            new_seq.append(new_idx)
            i += 2
        else:
            new_seq.append(seq[i])
            i += 1
    return tuple(new_seq)


# Updates the structures in place.
def update_sequence_freq(
    sequence_freq: Counter[TokenSequence],
    pair: TokenPair,
    new_idx: int,
    pair_to_sequences: dict[TokenPair, set[TokenSequence]],
    pair_freq: Counter[TokenPair],
) -> None:
    # take a snapshot before updating the dict in the loop
    old_sequences = list(pair_to_sequences.get(pair, set()))
    for old_seq in old_sequences:
        # stage 1: deletion.

        # delete from sequence_freq
        freq = sequence_freq.pop(old_seq)

        # delete from pair_to_sequences
        for old_pair in set(pairwise(old_seq)): # use set to avoid duplicate processing
            affected_set = pair_to_sequences.get(old_pair)
            if affected_set is None:
                raise KeyError(f"Pair {old_pair} missing while removing sequence {old_seq}")

            affected_set.remove(old_seq) #  # will raise error if old_seq does not exist
            if not affected_set:
                del pair_to_sequences[old_pair] # does not raise error

        # ADDED: delete from pair_seq
        for old_pair in pairwise(old_seq):
            pair_freq[old_pair] -= freq
            if pair_freq[old_pair] == 0:
                del pair_freq[old_pair]

        # stage 2: addition
        # add to sequence_freq
        new_seq = merge(old_seq, pair, new_idx)
        if new_seq in sequence_freq:
            raise KeyError(f"New seq {new_seq} already existed in sequence_freq. old_seq = {old_seq}")
        sequence_freq[new_seq] = freq

        # Add to pair_to_sequences
        for new_pair in set(pairwise(new_seq)): # use set to avoid duplicate processing
            pair_to_sequences.setdefault(new_pair, set()).add(new_seq)

        # ADDED: add to pair_freq
        for new_pair in pairwise(new_seq):
            # print(f"DEBUG: adding {new_pair}'s freq {freq} to dict")
            pair_freq[new_pair] += freq


def save_sequence_freq(
    token_freq: Counter[TokenSequence],
    output_path: Path,
) -> None:
    with output_path.open("wb") as file:
        pickle.dump(
            token_freq,
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

def load_sequence_freq(input_path: Path) -> Counter[TokenSequence]:
    with input_path.open("rb") as file:
        return pickle.load(file)

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    num_splits: int = 1,
    num_workers: int = 1,
    sequence_freq_input_path: Path | None = None,
    sequence_freq_output_path: Path | None = None,
    print_interval: int = 1000,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given a path to an input text file, trains a BPE tokenizer.

    Returns:
        vocab: dict[int, bytes] The tokenizer vocabulary, a mapping from int (token ID in the
            vocabulary) to bytes (token bytes).
        merges: list[tuple[bytes, bytes]] A list of BPE merges produced from training. Each list
            item is a tuple of bytes (<token1>, <token2>), representing that <token1> was merged with
            <token2>. The merges should be ordered by order of creation."""

    start = time.time()
    # Load from input
    if sequence_freq_input_path is not None:
        if not sequence_freq_input_path.exists():
            raise ValueError(f"Sequence frequency file {sequence_freq_input_path} does not exist!")
        sequence_freq = load_sequence_freq(sequence_freq_input_path)
        print(f"Loaded sequence frequency from {sequence_freq_input_path}")    
    else: # compute on the fly
        special_tokens_pattern = "|".join(map(re.escape, special_tokens))
        with open(input_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, num_splits, b"<|endoftext|>")
            boundaries = list(pairwise(boundaries)) # list of (start, end) tuples
        chunks = list(enumerate(boundaries))
        print(f"Generated {len(chunks)} pre-processing chunks")

        get_sequence_freq_worker = partial(
            get_sequence_freq,
            input_path=input_path,
            special_tokens_pattern=special_tokens_pattern,
        )

        print(f"Started preprocessing {len(chunks)} chunks with {num_workers} workers")
        # Count the token sequence frequency counter in parallel. After this
        # we can only run in serial.
        with Pool(num_workers) as pool:
            sequence_freq_local = pool.map(get_sequence_freq_worker, chunks)
        
        # Reduce the counter into a global counter
        sequence_freq = Counter()
        for local in sequence_freq_local:
            sequence_freq.update(local)
        
        # save the file to disk for future use
        if sequence_freq_output_path is not None:
            save_sequence_freq(sequence_freq, sequence_freq_output_path)
            print(f"Saved sequence frequency to {sequence_freq_output_path}")
   
    # initializes the mapping
    pair_to_seqs = get_pair_to_seqs(sequence_freq)
    pair_freq = get_pair_freq(sequence_freq)
    end = time.time()

    print(f"sequence_freq dict has {len(sequence_freq)} entries")
    print(f"pair_to_seqs dict has {len(pair_to_seqs)} entries")
    print(f"pair_freq dict has {len(pair_freq)} entries")
    print(f"Pre-tokenization took {end-start:.0f} seconds")

    start = end
    start2 = end

    vocab: dict[int, bytes] = init_vocab(special_tokens)
    merges: list[tuple[bytes, bytes]] = []

    num_merges = vocab_size - len(vocab)
    print(f"To perform {num_merges} merges")

    for i in range(num_merges):
        # pair_freq = get_pair_freq(sequence_freq)
        
        if not pair_freq:
            print("No more pairs to merge. Stop merging here")
            break

        # Find the next pair to merge: the one with the highest frequency. If tied
        # returns the one with higher lexigraphical order on the first element. If tied,
        # the second element. Note the second term  we can't use pair because that is
        # the integer index, not the string.
        pair_to_merge, _ = max(
            pair_freq.items(),
            key=lambda item: (
                item[1], # freq
                vocab[item[0][0]], # first element
                vocab[item[0][1]], # second element
            ),
        )

        idx = len(vocab) # idx of the merged byte sequence
        update_sequence_freq(sequence_freq, pair_to_merge, idx, pair_to_seqs, pair_freq)

        # update vocab and merges
        b1, b2 = vocab[pair_to_merge[0]], vocab[pair_to_merge[1]]
        vocab[idx] = b1 + b2
        merges.append((b1, b2))
        if i % print_interval == 0:
            end2 = time.time()
            print(f"Iteration {i}, time elapsed: {end2-start2:.0f} s, merging: {b1} + {b2}")
            start2 = end2

    end = time.time()
    print(f"Merging took {end-start:.0f} seconds")

    return vocab, merges


class BPETokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = vocab
        self.reverse_vocab: dict[bytes, int] = {v: k for k, v in vocab.items()}

        self.merges: dict[tuple[int, int], int] = {}
        self.merges_orig = merges
        for merge in merges:
            idx1 = self.reverse_vocab[merge[0]]
            idx2 = self.reverse_vocab[merge[1]]
            merged = merge[0] + merge[1]
            merged_idx = self.reverse_vocab[merged]
            self.merges[(idx1, idx2)] = merged_idx

        # original gpt2 pattern for splitting strings
        self.pat = regex.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

        self.special_tokens = None
        self.special_tokens_pattern = None
        if special_tokens:
            # put longer tokens upfront to handle overlapping special tokens
            self.special_tokens = sorted(special_tokens, key=lambda x: -len(x))
            self.special_tokens_pattern = "(" + "|".join(map(re.escape, self.special_tokens)) + ")"

    # For now we don't need the dict value. We can just return the set of keys
    # The value is for potential future use.
    def _get_stats(self, tokens: list[int]) -> dict[tuple[int, int], int]:
        stats: dict[tuple(int, int), int] = {}
        for ch1, ch2 in pairwise(tokens):
            stats[(ch1, ch2)] = stats.get((ch1, ch2), 0) + 1
        return stats

    # merge pair into idx for all occurances in tokens
    def _merge(self, tokens: list[int], pair: tuple[int, int], idx: int) -> list[int]:
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i+1] == pair[1]:
                new_tokens.append(idx)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens

    # Encodes a part of string split by the reg expression. It assumes that the string
    # is not a special token (handled elsewhere)
    def _encode_part(self, part: str) -> list[int]:
        tokens = list(part.encode("utf-8")) # list[int]
        # tiktok's first 256 mapping is not the same as unicode mapping, so we use reverse_vocab
        # to convert the unicode index to index in vocab.
        tokens = [self.reverse_vocab[bytes([value])] for value in tokens]

        while len(tokens) >= 2: # stops when we are left with only one token
            # find the next pair to merge
            stats = self._get_stats(tokens)
        
            # iterates over the keys of stats and find the one that has the smallest
            # merged index value in the merges dict.
            pair_to_merge = min(stats, key=lambda k: self.merges.get(k, float('inf')))

            # If no more keys to merge, pair_to_merge will default to the first elment of stats
            # and we can check if this is actually in self.merges. If not we can stop.
            if pair_to_merge not in self.merges:
                break
            
            merged_idx = self.merges[pair_to_merge]
            tokens = self._merge(tokens, pair_to_merge, merged_idx)
        
        return tokens

    # an alternative implementation where we apply the merges according to the merged file one by one.
    def _encode_part_2(self, part: str) -> list[int]:
        tokens = part.encode("utf-8") # bytes
        # tiktok's first 256 mapping is not the same as unicode mapping, so we use reverse_vocab
        # to convert the unicode index to index in vocab.
        tokens = [self.reverse_vocab[bytes([value])] for value in tokens]

        for merge in self.merges_orig:
            b1, b2 = merge
            idx1, idx2 = self.reverse_vocab[b1], self.reverse_vocab[b2]
            new_idx = self.reverse_vocab[b1 + b2]
            tokens = self._merge(tokens, (idx1, idx2), new_idx)

        return tokens

    def encode(self, s: str) -> list[int]:
        if self.special_tokens_pattern:
            pieces = [p for p in re.split(self.special_tokens_pattern, s) if p]
        else:
            pieces = [s]

        tokens: list[int] = []
        for p in pieces:
            if self.special_tokens and p in self.special_tokens:
                tokens.append(self.reverse_vocab[p.encode("utf-8")])
            else:
                for part in regex.findall(self.pat, p): # work on splitted parts one by one
                    tokens.extend(self._encode_part(part))
                    # tokens.extend(self._encode_part_2(part))

        return tokens

    def decode(self, indices: list[int]) -> str:
        bytes_list = [self.vocab[idx] for idx in indices]
        text = b"".join(bytes_list).decode("utf-8", errors="replace")
        return text

    def encode_iterable(self, texts: Iterable[str]) -> Iterator[int]:
        for text in texts:
            yield from self.encode(text)


def get_tokenizer_from_vocab_merges_pickle_path(
    vocab_path: Path,
    merges_path: Path,
) -> BPETokenizer:
    print(f"Creating tokenizer from vocab_path {vocab_path} and merges_path {merges_path}")

    with Path(vocab_path).open("rb") as f:
        vocab = pickle.load(f)

    with Path(merges_path).open("rb") as f:
        merges = pickle.load(f)

    special_tokens = ["<|endoftext|>"]
    return BPETokenizer(vocab, merges, special_tokens) 
