#!/usr/bin/env bash
# Change one setting at a time from the defaults and append one JSON line per run.
# Usage: bench/sweep.sh MODEL_DIR OUT.jsonl [SECONDS_PER_RUN]
set -euo pipefail
model=$(cd "$1" && pwd)
out=$2
secs=${3:-10}
cache=$(mktemp -d)
trap 'rm -rf "$cache"' EXIT

run() {
    local label=$1 clients=$2 texts=$3
    shift 3
    echo "run: $label" >&2
    env CYBERTRON_MODEL="$model" "$@" uv run python bench/bench.py \
        --seconds "$secs" --clients "$clients" --texts "$texts" --label "$label" >>"$out"
}

run "defaults" 4 8
for p in FP16 INT8; do run "precision=$p" 4 8 CYBERTRON_PRECISION=$p; done
run "precision=FP32 exec=f16" 4 8 CYBERTRON_OV_CONFIG='{"INFERENCE_PRECISION_HINT": "f16"}'
run "precision=FP16 exec=f32" 4 8 CYBERTRON_PRECISION=FP16 CYBERTRON_OV_CONFIG='{"INFERENCE_PRECISION_HINT": "f32"}'
for h in LATENCY THROUGHPUT; do
    for c in 1 4 16; do run "hint=$h clients=$c" "$c" 8 CYBERTRON_HINT=$h; done
done
for t in 1 2 4; do run "threads=$t" 4 8 CYBERTRON_THREADS=$t; done
for b in 8 16 32 64; do run "batch_size=$b texts=64" 4 64 CYBERTRON_BATCH_SIZE=$b; done
for m in 32 64 128 256; do run "max_len=$m" 4 8 CYBERTRON_MAX_LEN=$m; done
run "buckets=[max_len]" 4 8 CYBERTRON_BUCKETS='[]'
run "buckets=[8..128] fine" 4 8 CYBERTRON_BUCKETS='[8,16,24,32,48,64,96]'
run "cache=cold" 1 8 CYBERTRON_CACHE_DIR="$cache"
run "cache=warm" 1 8 CYBERTRON_CACHE_DIR="$cache"
