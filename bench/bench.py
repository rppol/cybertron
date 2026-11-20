"""Start one server with the CYBERTRON_* settings in the environment, load it, print one JSON line.

    uv run python bench/bench.py --clients 4 --seconds 15 >> results.jsonl
    uv run python bench/bench.py --table results.jsonl
"""

import argparse
import asyncio
import json
import os
import platform
import random
import socket
import subprocess
import sys
import time

import httpx
import psutil

WORDS = (
    "the a of to and in is it you that he was for on are with as his they be at one have this from or had by "
    "word but what some we can out other were all there when up use your how said an each she which do their "
    "time if will way about many then them write would like so these her long make thing see him two has look "
    "more day could go come did number sound no most people my over know water than call first who may down"
).split()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clients", type=int, default=4)
    p.add_argument("--seconds", type=float, default=15)
    p.add_argument("--texts", type=int, default=8, help="texts per request")
    p.add_argument("--words", default="4-60", help="words per text, min-max")
    p.add_argument("--label", default="")
    p.add_argument("--table", help="print a Markdown table from a JSONL results file and exit")
    args = p.parse_args()
    if args.table:
        return table(args.table)

    port = _free_port()
    start = time.perf_counter()
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "cybertron.app:create_app", "--factory", "--port", str(port), "--log-level", "warning"],
        stdout=sys.stderr,
    )
    try:
        info = _wait(port, server)
        startup = time.perf_counter() - start
        result = asyncio.run(_load(port, server.pid, args))
    finally:
        server.terminate()
        server.wait()
    knobs = {k[len("CYBERTRON_"):].lower(): v for k, v in os.environ.items() if k.startswith("CYBERTRON_")}
    knobs.pop("model", None)
    print(json.dumps({"label": args.label, "knobs": knobs, "clients": args.clients, "texts_per_request": args.texts,
                      "startup_s": round(startup, 2), "info": info, "hardware": hardware(), **result}), flush=True)


def hardware():
    if sys.platform == "darwin":
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    else:
        cpu = next((line.split(":", 1)[1].strip() for line in open("/proc/cpuinfo") if line.startswith("model name")), "")
    return {"machine": platform.machine(), "system": platform.system(), "cpu": cpu,
            "cores": psutil.cpu_count(logical=False), "threads": psutil.cpu_count(),
            "memory_gb": round(psutil.virtual_memory().total / 2**30, 1)}


def table(path):
    rows = [json.loads(line) for line in open(path) if line.strip()]
    print("| run | precision (exec) | hint | streams | threads | slots | clients | texts/s | p50 ms | p95 ms | p99 ms | peak RSS MB | 429s | startup s |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in rows:
        i = r["info"]
        print(f"| {r['label']} | {i['precision']} ({i['inference_precision']}) | {i['hint']} | {i['streams']} | {i['threads']} "
              f"| {i['slots']} | {r['clients']} | {r['texts_per_s']} | {r['p50_ms']} | {r['p95_ms']} | {r['p99_ms']} "
              f"| {r['peak_rss_mb']} | {r['busy']} | {r['startup_s']} |")


async def _load(port, pid, args):
    rng = random.Random(0)
    lo, hi = map(int, args.words.split("-"))
    pool = [" ".join(rng.choices(WORDS, k=rng.randint(lo, hi))) for _ in range(512)]
    proc = psutil.Process(pid)
    latencies, busy, texts, peak = [], 0, 0, proc.memory_info().rss
    deadline = time.perf_counter() + args.seconds

    async def client(n):
        nonlocal busy, texts
        crng = random.Random(n)
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=60) as c:
            while time.perf_counter() < deadline:
                body = {"texts": crng.sample(pool, args.texts)}
                t = time.perf_counter()
                r = await c.post("/v1/classify", json=body)
                if r.status_code == 429:
                    busy += 1
                    await asyncio.sleep(0.05)
                    continue
                r.raise_for_status()
                latencies.append(time.perf_counter() - t)
                texts += args.texts

    async def watch():
        nonlocal peak
        while time.perf_counter() < deadline:
            peak = max(peak, proc.memory_info().rss)
            await asyncio.sleep(0.1)

    t0 = time.perf_counter()
    await asyncio.gather(watch(), *(client(n) for n in range(args.clients)))
    elapsed = time.perf_counter() - t0
    latencies.sort()

    def pct(q):
        return round(latencies[min(len(latencies) - 1, int(q * len(latencies)))] * 1000, 1) if latencies else None

    return {"texts_per_s": round(texts / elapsed, 1), "p50_ms": pct(0.5), "p95_ms": pct(0.95), "p99_ms": pct(0.99),
            "peak_rss_mb": round(peak / 2**20), "busy": busy, "requests": len(latencies)}


def _wait(port, server):
    while server.poll() is None:
        try:
            return httpx.get(f"http://127.0.0.1:{port}/health", timeout=1).json()["info"]
        except httpx.HTTPError:
            time.sleep(0.2)
    raise SystemExit(f"server exited with {server.returncode}")


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


if __name__ == "__main__":
    main()
