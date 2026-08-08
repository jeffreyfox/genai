### Generate token sequence frequency files
First generate the frequency ts pkl files so we don't need to regenerate it.
```
uv run train_bpe.py --input-path=data/TinyStoriesV2-GPT4-train.txt --vocab-size=10000 --num-workers=8 --num-splits=32
```

Optionally pass in the output path to write the sequence frequency file:
--sequence-freq-output-path=freq_ts.pkl

So next time we can skip the multi-processings stage which is the most time consuming part.

Benccmarking on M1 CPU:
```
sequence_freq dict has 59933 entries
pair_to_seqs dict has 2108 entries
pair_freq dict has 2108 entries
Pre-tokenization took 71 seconds
To perform 9743 merges
Iteration 0, time elapsed: 0 s, merging: b' ' + b't'
Iteration 1000, time elapsed: 3 s, merging: b' c' + b'ut'
Iteration 2000, time elapsed: 4 s, merging: b' v' + b'al'
Iteration 3000, time elapsed: 4 s, merging: b' qu' + b'ar'
Iteration 4000, time elapsed: 5 s, merging: b' te' + b'ase'
Iteration 5000, time elapsed: 5 s, merging: b' spoon' + b's'
Iteration 6000, time elapsed: 5 s, merging: b' tim' + b'er'
Iteration 7000, time elapsed: 5 s, merging: b' L' + b'izzie'
Iteration 8000, time elapsed: 6 s, merging: b' invit' + b'ations'
Iteration 9000, time elapsed: 6 s, merging: b' sc' + b'ulpture'
Merging took 47 seconds
Training took 118 seconds
Vocab size: 10000
Number of merges: 9743
Vocab and merges saved to vocab_ts.pkl and merges_ts.pkl
```

Benchmarking on M4 2024 CPU:
```
Finished pre-processing chunk 31, start=2158136601, end=2227753162, size=66.4 MB. Took 9.9 s.
Merging took 28 seconds
Training took 67 seconds
```

### Tokenize input text
Next is to tokenize the training input into binary token integers. We do this once before training the model
so that the model trainer can simply load the encoded token integers instead of raw texts.

Also do this for validation dataset.
```
uv run tokenize_input.py --input-path=data/TinyStoriesV2-GPT4-train.txt --output-path=data/TinyStoriesV2-GPT4-train.bin --vocab-path=vocab_ts.pkl --merges-path=merges_ts.pkl --num-splits=64 --num-workers=8

uv run tokenize_input.py --input-path=data/TinyStoriesV2-GPT4-valid.txt --output-path=data/TinyStoriesV2-GPT4-valid.bin --vocab-path=vocab_ts.pkl --merges-path=merges_ts.pkl --num-splits=1 --num-workers=1
```

Benchmarking on M1 CPU:
```
TODO
```

Benchmarking on M4 CPU:
```
tokenized 8457751 tokens, 33 MB data. Took 36.3 s
```

### Train model
Then start training the model
The training configs are stored in configs/train_llm_ts.toml
```
uv run train_llm.py --dataset-name=TinyStoriesV2-GPT4 --vocab-path=vocab_ts.pkl --merges-path=merges_ts.pkl --config-path=configs/train_llm_ts.toml --print-interval=100 --checkpoint-interval=100 --device=mps
```
Training on mps makes it 1.5x faster. Then on M4 is 2x faster on M1.

On M1 CPU:
tokens/s = 2251.5

On M1 GPU:
tokens/s = 3127.7

On M4 GPU:
```
Step 5000: training loss = 1.780, validation loss = 1.718, tokens processed = 819200, time elapsed = 104.3 s,  tokens/s = 7851.7
Trained for 5000 steps, processed 40960000 tokens, total training time: 9367.6 s
```

TODO:
- finish the memory/flops estimates in the assignments
- connect to wandb
- try the effect of torch.compile
- try training owt
- refactor code for better organization
- understand difference between encode_part and encode_part_2 and see if we can improve the latter
