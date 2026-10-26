#!/usr/bin/env bash
# Run the Modal benchmark for each model size and stage, then download the snapshot.
# Snapshots land in this directory as <config>-<stages>.pickle, for example medium-fbo.pickle.
# A failed run (for example out of memory) is reported and the loop continues.

set -euo pipefail

cd "$(dirname "$0")"

# configs=(small medium large xl 10b)
configs=(small medium large)
# modes=("f,b,o" "f")
modes=("f,b")
num_steps=10
warmup_steps=5

for cfg in "${configs[@]}"; do
  for mode in "${modes[@]}"; do
    echo "===== ${cfg} ${mode} ====="
    if just benchmark-modal "$num_steps" "$warmup_steps" "$mode" "${cfg}.toml"; then
      just download-snapshot "${cfg}.toml" "$mode"
    else
      echo "FAILED ${cfg} ${mode}"
    fi
  done
done
