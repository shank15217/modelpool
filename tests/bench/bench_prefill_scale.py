#!/usr/bin/env python3
"""
Quick scaling test for prefill across concurrency levels.
Runs bench_prefill_rag.py at 1, 2, 4, 8 concurrent for a given context size.
"""

import asyncio
import json
import subprocess
import sys

ENDPOINT = sys.argv[1] if len(sys.argv) > 1 else "http://192.168.35.185:8000/v1"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "/run/host/AITOOLCHAIN/models/Qwen3.6-27B-FP8"
CTX = int(sys.argv[3]) if len(sys.argv) > 3 else 16384

CONCURRENCY_LEVELS = [1, 2, 4, 8]


async def run_level(n):
    cmd = [
        sys.executable,
        "tests/bench/bench_prefill_rag.py",
        "--endpoint", ENDPOINT,
        "--model", MODEL,
        "--concurrent", str(n),
        "--ctx", str(CTX),
        "--runs", "3",
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        print(f"ERROR at concurrency {n}: {stderr.decode()}", file=sys.stderr)
        return None
    return json.loads(stdout.decode())


async def main():
    print(f"Prefill scaling test @ {CTX} tokens", file=sys.stderr)
    print(f"Endpoint: {ENDPOINT}", file=sys.stderr)
    print(f"Model: {MODEL}\n", file=sys.stderr)

    results = {}
    for n in CONCURRENCY_LEVELS:
        print(f"Testing {n} concurrent...", file=sys.stderr)
        res = await run_level(n)
        if res:
            results[n] = res["summary"]["avg_aggregate_prefill_tps"]
            print(f"  -> {results[n]} t/s aggregate\n", file=sys.stderr)

    print("\n=== PREFILL SCALING ===", file=sys.stderr)
    for n, tps in results.items():
        print(f"{n:>2} agents: {tps:>8.1f} t/s aggregate", file=sys.stderr)

    print(json.dumps({"context": CTX, "results": results}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
