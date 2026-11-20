# Tuning

Each setting is an environment variable with the `CYBERTRON_` prefix. `GET /health` returns the values that OpenVINO resolved at startup: execution precision, hint, streams, threads, pool size, and buckets. Read them there, because OpenVINO picks its own defaults per CPU.

## Bench setup

All numbers in this file come from one machine:

| Item | Value |
| --- | --- |
| CPU | Apple M5, 10 cores, arm64 |
| Memory | 16 GB |
| OS | macOS (Darwin) |
| OpenVINO | 2025.4.1 |
| Python | 3.12 |

These are arm64 numbers. They do not predict x86 results. On this CPU the OpenVINO default execution precision is f16. On x86 the default depends on the instruction set (f32, or bf16 on CPUs with bf16 support).

`bench/bench.py` starts one server, sends `POST /v1/classify` from N clients for a fixed time, and prints one JSON line. Each request has 8 texts of 4 to 60 common English words. A client that gets `429` counts it, waits 50 ms, and sends a new request. `bench/sweep.sh` changes one setting at a time from the defaults:

```sh
bench/sweep.sh models/sst2 results.jsonl 15
uv run python bench/bench.py --table results.jsonl
```

The "texts/s" column counts texts in successful requests. Latency is per request. "Peak RSS" is the server process, sampled every 100 ms. "Startup" runs from process start to the first `/health` answer.

### distilbert-base-uncased-finetuned-sst-2-english, 15 s per run

| run | precision (exec) | hint | streams | threads | slots | clients | texts/s | p50 ms | p95 ms | p99 ms | peak RSS MB | 429s | startup s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| defaults | FP32 (f32) | THROUGHPUT | 7 | 10 | 7 | 4 | 85.9 | 430.9 | 472.6 | 510.9 | 1142 | 0 | 6.15 |
| precision=FP16 | FP16 (f16) | THROUGHPUT | 7 | 10 | 7 | 4 | 162.8 | 225.2 | 250.0 | 272.1 | 751 | 0 | 6.53 |
| precision=INT8 | INT8 (f32) | THROUGHPUT | 7 | 10 | 7 | 4 | 83.8 | 437.2 | 485.4 | 491.6 | 1298 | 0 | 7.22 |
| precision=FP32 exec=f16 | FP32 (f16) | THROUGHPUT | 7 | 10 | 7 | 4 | 161.5 | 212.5 | 254.3 | 280.4 | 752 | 0 | 6.36 |
| precision=FP16 exec=f32 | FP16 (f32) | THROUGHPUT | 7 | 10 | 7 | 4 | 84.9 | 321.6 | 483.2 | 508.3 | 1062 | 0 | 6.1 |
| hint=LATENCY clients=1 | FP32 (f32) | LATENCY | 1 | 10 | 1 | 1 | 95.3 | 80.5 | 102.4 | 130.0 | 880 | 0 | 5.2 |
| hint=LATENCY clients=4 | FP32 (f32) | LATENCY | 1 | 10 | 1 | 4 | 94.9 | 81.3 | 95.3 | 120.4 | 900 | 804 | 5.04 |
| hint=LATENCY clients=16 | FP32 (f32) | LATENCY | 1 | 10 | 1 | 16 | 91.8 | 83.8 | 101.3 | 133.8 | 879 | 3819 | 5.52 |
| hint=THROUGHPUT clients=1 | FP32 (f32) | THROUGHPUT | 7 | 10 | 7 | 1 | 27.1 | 356.6 | 372.4 | 466.9 | 1133 | 0 | 5.04 |
| hint=THROUGHPUT clients=4 | FP32 (f32) | THROUGHPUT | 7 | 10 | 7 | 4 | 87.9 | 428.6 | 460.7 | 490.2 | 1174 | 0 | 5.25 |
| hint=THROUGHPUT clients=16 | FP32 (f32) | THROUGHPUT | 7 | 10 | 7 | 16 | 109.3 | 478.8 | 686.9 | 882.8 | 1200 | 2157 | 4.58 |
| threads=1 | FP32 (f32) | THROUGHPUT | 1 | 1 | 1 | 4 | 22.2 | 357.9 | 365.3 | 427.8 | 881 | 843 | 5.04 |
| threads=2 | FP32 (f32) | THROUGHPUT | 2 | 2 | 2 | 4 | 42.2 | 375.0 | 406.3 | 503.5 | 930 | 560 | 5.48 |
| threads=4 | FP32 (f32) | THROUGHPUT | 4 | 4 | 4 | 4 | 73.8 | 427.2 | 456.7 | 520.2 | 1042 | 0 | 5.03 |
| batch_size=8 texts=64 | FP32 (f32) | THROUGHPUT | 7 | 10 | 7 | 4 | 108.3 | 2288.1 | 2550.6 | 2871.7 | 1232 | 0 | 5.26 |
| batch_size=16 texts=64 | FP32 (f32) | THROUGHPUT | 7 | 10 | 7 | 4 | 99.3 | 2383.9 | 2958.9 | 3081.8 | 1446 | 0 | 4.8 |
| batch_size=32 texts=64 | FP32 (f32) | THROUGHPUT | 7 | 10 | 7 | 4 | 88.0 | 2992.6 | 3190.3 | 3973.2 | 1557 | 0 | 5.51 |
| batch_size=64 texts=64 | FP32 (f32) | THROUGHPUT | 7 | 10 | 7 | 4 | 64.5 | 4326.8 | 5042.5 | 5042.5 | 1762 | 0 | 4.81 |
| max_len=32 | FP32 (f32) | THROUGHPUT | 7 | 10 | 7 | 4 | 170.5 | 213.1 | 244.6 | 276.7 | 1124 | 0 | 5.02 |
| max_len=64 | FP32 (f32) | THROUGHPUT | 7 | 10 | 7 | 4 | 87.3 | 432.8 | 467.2 | 557.9 | 1157 | 0 | 4.83 |
| max_len=128 | FP32 (f32) | THROUGHPUT | 7 | 10 | 7 | 4 | 88.9 | 400.4 | 453.8 | 518.8 | 1155 | 0 | 5.28 |
| max_len=256 | FP32 (f32) | THROUGHPUT | 7 | 10 | 7 | 4 | 89.5 | 416.1 | 448.9 | 477.9 | 1170 | 0 | 5.24 |
| buckets=[max_len] | FP32 (f32) | THROUGHPUT | 7 | 10 | 7 | 4 | 42.7 | 852.4 | 923.8 | 971.8 | 1394 | 0 | 5.06 |
| buckets=[8..128] fine | FP32 (f32) | THROUGHPUT | 7 | 10 | 7 | 4 | 93.6 | 320.5 | 444.9 | 495.0 | 1153 | 0 | 5.25 |
| cache=cold | FP32 (f32) | THROUGHPUT | 7 | 10 | 7 | 1 | 28.0 | 356.4 | 369.8 | 372.2 | 1162 | 0 | 5.5 |
| cache=warm | FP32 (f32) | THROUGHPUT | 7 | 10 | 7 | 1 | 28.2 | 356.5 | 369.1 | 412.3 | 1083 | 0 | 5.45 |

Run-to-run noise: the `defaults` and `hint=THROUGHPUT clients=4` rows use the same settings and differ by 2%.

### Tiny random BERT (2 layers, hidden size 32), 10 s per run

This model shows the cost outside the model: HTTP, JSON, tokenizing, and the thread hand-off. On this model the service handles about 2,700 texts/s with one client and 7,600 texts/s with four clients at the defaults. To get all rows, run `bench/sweep.sh` on a model that `tests/tiny.py` writes.

| run | clients | texts/s | p50 ms | p99 ms | peak RSS MB |
| --- | --- | --- | --- | --- | --- |
| defaults | 4 | 7649.1 | 3.8 | 8.8 | 486 |
| hint=LATENCY clients=1 | 1 | 3875.8 | 1.9 | 4.5 | 413 |
| hint=THROUGHPUT clients=1 | 1 | 2727.8 | 3.0 | 6.0 | 476 |
| hint=THROUGHPUT clients=16 | 16 | 8981.3 | 8.6 | 15.9 | 489 |
| batch_size=64 texts=64 | 4 | 12825.3 | 19.8 | 40.3 | 632 |

## Memory test

`tests/test_memory.py` (marked `slow`) loads a MiniLM-sized model. It runs every bucket and batch size on every infer request, then runs 500 batches of random size and length across all requests. On the bench machine, RSS went from 868 MB to 884 MB after warm-up. The test fails when RSS grows by 10% or more.

## Settings

### `CYBERTRON_MODEL`

Path to the model directory. There is no default. See the README for the accepted layouts.

### `CYBERTRON_LABELS`

A JSON list that replaces `config.id2label`, for example `["neg", "pos"]`. The length must match the head width.

### `CYBERTRON_HOOKS`

An importable module with `pre` and `post` functions. See the README. Hooks run on the request path, so their cost adds to every request.

### `CYBERTRON_PRECISION`

`FP32` (default), `FP16`, or `INT8`. This sets how the converted model stores its weights. `FP16` uses OpenVINO weight compression. `INT8` uses NNCF `INT8_ASYM` weight compression.

The execution precision is a separate value. `FP32` and `INT8` pin execution to f32. `FP16` keeps the OpenVINO default for the CPU, which is f16 on the bench machine. To set the execution precision yourself, use `CYBERTRON_OV_CONFIG='{"INFERENCE_PRECISION_HINT": "f16"}'`.

On the bench machine, execution precision set the speed and the memory, and weight storage did not:

- f16 execution ran about 1.9 times faster than f32 (163 against 86 texts/s) and used about 390 MB less peak RSS. This held for FP16 and for FP32 weights.
- `INT8` with f32 execution was as fast as `FP32` and used more peak RSS (1298 against 1142 MB).
- `INT8` with f16 execution returned NaN logits under concurrent load with OpenVINO 2025.4.1. The service then answers 500 for that request. This is why `INT8` pins f32.

To choose: check accuracy on your own data at f16 before you use it. Compare the `inference_precision` value in `/health` across CPUs, because the default changes with the CPU.

### `CYBERTRON_HINT`

`THROUGHPUT` (default) or `LATENCY`. OpenVINO derives streams, threads, and the optimal number of infer requests from it.

On the bench machine, `THROUGHPUT` made 7 streams and 7 infer requests. `LATENCY` made 1 stream with all 10 threads and 1 infer request.

- With one client, `LATENCY` had a p50 of 81 ms. `THROUGHPUT` had 357 ms, because one stream has only a part of the cores.
- `THROUGHPUT` needs enough clients to fill its streams. With 4 clients it did 88 texts/s, and with 16 clients 109 texts/s.
- `LATENCY` with automatic slots gives one slot. A second concurrent request gets `429`.

To choose: use `LATENCY` when one request at a time must be fast and clients retry on `429`. Use `THROUGHPUT` when many clients send at once.

### `CYBERTRON_STREAMS`

Sets `NUM_STREAMS`. Unset lets the hint decide. Each stream runs one inference at a time on its own share of the threads. More streams give more parallel requests and a higher latency for each. This setting was not swept on its own. The `threads` rows below changed streams with threads.

### `CYBERTRON_THREADS`

Sets `INFERENCE_NUM_THREADS`. Unset uses all cores. With `THROUGHPUT`, OpenVINO made one stream per thread at 1, 2, and 4 threads. Throughput on the bench machine was 22, 42, and 74 texts/s, and 86 texts/s with all 10 cores. Set it below the core count when other processes share the machine. In a container, compare `threads` in `/health` with the CPU limit. In the Docker check on the bench machine (OrbStack), OpenVINO reported 16 threads.

### `CYBERTRON_HYPER_THREADING`

Sets `ENABLE_HYPER_THREADING`. Unset keeps the OpenVINO default. It applies to CPUs with SMT. The bench machine has no SMT, so this was not measured.

### `CYBERTRON_CPU_PINNING`

Sets `ENABLE_CPU_PINNING`. Unset keeps the OpenVINO default. Pinning keeps stream threads on fixed cores. It was not measured on the bench machine, where OpenVINO reported it off for compiled models.

### `CYBERTRON_OV_CONFIG`

A JSON object of other OpenVINO CPU properties, for example `{"EXECUTION_MODE_HINT": "ACCURACY"}`. The service rejects keys that a typed setting owns: `PERFORMANCE_HINT`, `NUM_STREAMS`, `INFERENCE_NUM_THREADS`, `ENABLE_HYPER_THREADING`, `ENABLE_CPU_PINNING`, and `CACHE_DIR`. A key that the CPU plugin does not support stops startup with an error. `INFERENCE_PRECISION_HINT` here replaces the precision default.

### `CYBERTRON_SLOTS`

The size of the infer request pool, which is the number of requests that run at once. `0` (default) uses `OPTIMAL_NUMBER_OF_INFER_REQUESTS` from OpenVINO: 7 with `THROUGHPUT` and 1 with `LATENCY` on the bench machine. A request that finds no free slot gets `429` at once.

Each slot holds its own input and output buffers, so more slots use more memory. With more slots than streams, requests wait inside OpenVINO for a free stream. The worker thread limit is `slots + 8`.

### `CYBERTRON_BATCH_SIZE`

The largest number of texts in one inference. Default `32`. A request with more texts runs in several inferences on the same slot. The service sorts texts by length first, so each batch pads to the length of its own longest text.

With 64 texts per request on the bench machine, a smaller batch was faster: batch size 8 did 108 texts/s, and 64 did 65 texts/s. Peak RSS rose with batch size, from 1232 MB to 1762 MB. Smaller batches pad less and keep the activation memory small. Batch size also bounds the model's batch dimension, so the shape count is at most batch size times bucket count.

### `CYBERTRON_MAX_LEN`

The token limit per text. Default `128`. The service uses the smallest of this value, the tokenizer `model_max_length`, and the model position limit. Longer texts are cut.

Cost grows with the padded length. With texts of up to about 62 tokens, `max_len=32` doubled the speed (171 texts/s), because every batch padded to 32 tokens or fewer instead of 64. That changes the model input and can change the result. `max_len` of 64, 128, and 256 gave the same speed, since no text was longer than 64 tokens. Set it to cover your longest texts that matter.

### `CYBERTRON_BUCKETS`

Sequence lengths that each batch pads to. Default `[16, 32, 64, 128]`. The service drops values at or above `max_len` and always adds `max_len` as the top bucket.

Buckets bound the number of input shapes, which keeps the memory flat over time (see the memory test). With one bucket (`[]`, so every batch pads to `max_len`), throughput fell to 43 texts/s from 86. Finer buckets (`[8, 16, 24, 32, 48, 64, 96]`) gave 94 texts/s. Put bucket edges near the common text lengths. Each extra bucket adds shapes, and each shape uses some memory per slot.

### `CYBERTRON_CACHE_DIR`

A directory for compiled model blobs (`CACHE_DIR`). Unset turns the cache off. The service converts the model from PyTorch at each start, and the cache skips only the compile. On the bench machine, startup was 5.5 s cold and 5.45 s warm for DistilBERT, so the compile is a small part. Larger models or slower CPUs can gain more. In Docker, mount a volume at `/cache`.

### `CYBERTRON_MAX_TEXTS`

The largest number of texts in one request. Default `256`. More gets `413`. This bounds the time that one request holds a slot.

### `CYBERTRON_MAX_CHARS`

The longest text in characters. Default `4096`. Longer gets `413`. This bounds tokenizer time and memory.

### `CYBERTRON_API_KEY`

When set, `/v1/*` requires this value in the `X-API-Key` header. There is no default key. Health and metrics routes stay open.

### `CYBERTRON_RATE_LIMIT`

The number of `/v1/*` requests that one client address can send in one window. `0` (default) turns it off. Over the limit gets `429` with `Retry-After`. The table clears at the end of each window, so it holds at most one entry per address seen in that window.

### `CYBERTRON_RATE_WINDOW`

The window length in seconds for `CYBERTRON_RATE_LIMIT`. Default `60`.

### `CYBERTRON_GC_INTERVAL`

Seconds between calls to `gc.collect()` and, on glibc, `malloc_trim(0)`. Default `300`. `0` turns it off. This returns freed memory to the OS between bursts. Not measured on the bench machine, since `malloc_trim` exists only on Linux.

### `CYBERTRON_GC_THRESHOLDS`

The Python GC thresholds set after startup, as `[gen0, gen1, gen2]`. Default `[50000, 20, 20]`. At startup the service also freezes the objects that exist (the tokenizer and the compiled model), so the collector does not scan them again. Higher thresholds mean fewer collections on the request path. Not measured on the bench machine.
