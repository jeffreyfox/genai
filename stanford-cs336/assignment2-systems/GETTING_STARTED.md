# Benchmark

Times are milliseconds per step, mean ± sample standard deviation, over 10 timed steps. Phases are forward (`f`), backward (`b`), and optimizer (`o`). Warm-up steps run first and are not included in these numbers.

Shared model settings from `cs336_systems/training_config.toml`:

| Setting | Value |
| --- | --- |
| Vocab size | 10000 |
| Batch size | 32 |
| Context length | 256 |
| Learning rate | 0.001 |

## Commands

From `assignment2-systems`:

```sh
# Local machine. device is cpu or gpu (CUDA if available, otherwise MPS).
just benchmark gpu 10 5 f,b,o

# Modal L40S, timeit output only.
just benchmark-modal 10 5 f,b,o

# Modal L40S, Nsight Systems profile written to the cs336-data volume.
just profile-modal 10 5 f,b,o
```

The three numbers are timed steps, warm-up steps, and which phases to time.

## Modal L40S

`just benchmark-modal`, varying only the warm-up count:

| Warm-up steps | Forward | Backward | Optimizer |
| --- | --- | --- | --- |
| 0 | 57.5 ± 77.1 | 69.4 ± 45.0 | 3.9 ± 0.8 |
| 2 | 33.0 ± 0.0 | 55.0 ± 0.1 | 3.5 ± 0.0 |
| 5 | 33.2 ± 0.0 | 54.9 ± 0.1 | 6.6 ± 0.3 |

With no warm-up, forward and backward vary widely across the 10 measured steps. After 2 warm-up steps those times settle near 33 ms and 55 ms. A 5-step warm-up matches that.

## Mac M1 (MPS)

`just benchmark` on the local GPU:

| Forward | Backward | Optimizer |
| --- | --- | --- |
| 565.6 ± 1.4 | 1207.0 ± 27.5 | 64.5 ± 2.7 |

The same model is much slower here than on the L40S: about 0.57 s for the forward pass and 1.2 s for the backward pass.
