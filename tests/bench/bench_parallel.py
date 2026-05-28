#!/usr/bin/env python3
"""ModelPool benchmark: parallel slot performance.

Tests inference throughput with multiple concurrent requests to measure
how well a resource utilizes parallel slots.

Usage:
  python bench_parallel.py [--endpoint URL] [--max-slots N]

Output: JSON results to stdout, human-readable summary to stderr.
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


def chat(endpoint: str, messages: list, max_tokens: int = 256, label: str = "") -> dict:
    """Send a chat completion request and return timing + usage."""
    start = time.time()
    resp = requests.post(
        f"{endpoint}/v1/chat/completions",
        json={
            "model": "bench",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.3,
        },
        timeout=300,
    )
    elapsed = time.time() - start
    data = resp.json()
    usage = data.get("usage", {})
    pt = usage.get("prompt_tokens", 0)
    ct = usage.get("completion_tokens", 0)
    return {
        "label": label,
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "total_tokens": pt + ct,
        "time_s": round(elapsed, 2),
    }


# Diverse prompts to avoid KV cache sharing
PROMPTS = [
    {
        "role": "system",
        "content": "You are an expert Python developer.",
    },
    {
        "role": "system",
        "content": "You are a systems engineer.",
    },
    {
        "role": "system",
        "content": "You are a helpful assistant.",
    },
    {
        "role": "system",
        "content": "You are a cybersecurity analyst.",
    },
]

QUESTIONS = [
    "Write a complete Python implementation of an LRU cache with get, put, and eviction.",
    "Explain how OVS (Open vSwitch) works with OVN for software-defined networking.",
    "Write a detailed comparison of ZFS vs ext4 vs Btrfs. At least 400 words.",
    "Describe the security implications of running containers as root vs non-root.",
]

MAX_TOKENS = [512, 512, 512, 512]


def bench_single(endpoint: str) -> dict:
    """Baseline: single request at a time."""
    results = []
    for i in range(len(PROMPTS)):
        messages = [PROMPTS[i], {"role": "user", "content": QUESTIONS[i]}]
        r = chat(endpoint, messages, max_tokens=MAX_TOKENS[i], label=f"req_{i}")
        results.append(r)

    total_tok = sum(r["total_tokens"] for r in results)
    total_time = sum(r["time_s"] for r in results)
    return {
        "test": "sequential",
        "requests": len(results),
        "total_tokens": total_tok,
        "wall_time_s": round(total_time, 2),
        "throughput_tps": round(total_tok / total_time, 1) if total_time > 0 else 0,
        "per_request": results,
    }


def bench_concurrent(endpoint: str, n_concurrent: int) -> dict:
    """Run n_concurrent requests simultaneously."""
    start = time.time()
    results = []

    with ThreadPoolExecutor(max_workers=n_concurrent) as pool:
        futures = []
        for i in range(n_concurrent):
            messages = [PROMPTS[i % len(PROMPTS)], {"role": "user", "content": QUESTIONS[i % len(QUESTIONS)]}]
            futures.append(
                pool.submit(chat, endpoint, messages, max_tokens=MAX_TOKENS[i % len(MAX_TOKENS)], label=f"slot_{i}")
            )
        for f in as_completed(futures):
            results.append(f.result())

    wall_time = time.time() - start
    total_tok = sum(r["total_tokens"] for r in results)

    return {
        "test": f"concurrent_{n_concurrent}",
        "requests": n_concurrent,
        "total_tokens": total_tok,
        "wall_time_s": round(wall_time, 2),
        "throughput_tps": round(total_tok / wall_time, 1) if wall_time > 0 else 0,
        "per_request": results,
    }


def main():
    parser = argparse.ArgumentParser(description="ModelPool parallel slot benchmark")
    parser.add_argument(
        "--endpoint", "-e",
        default="http://localhost:8080",
        help="Inference endpoint URL",
    )
    parser.add_argument(
        "--resource", "-r",
        default=None,
        help="Resource name (for labeling)",
    )
    parser.add_argument(
        "--max-slots",
        type=int,
        default=3,
        help="Maximum concurrent requests to test (default: 3)",
    )
    args = parser.parse_args()

    label = args.resource or args.endpoint

    print(f"{'=' * 70}", file=sys.stderr)
    print(f"Parallel Benchmark: {label}", file=sys.stderr)
    print(f"Endpoint:           {args.endpoint}", file=sys.stderr)
    print(f"{'=' * 70}", file=sys.stderr)

    all_results = []

    # Baseline: sequential
    print(f"\n--- SEQUENTIAL (baseline) ---", file=sys.stderr)
    seq = bench_single(args.endpoint)
    for r in seq["per_request"]:
        print(
            f"  {r['label']}: {r['total_tokens']} tok in {r['time_s']}s",
            file=sys.stderr,
        )
    print(f"  Total: {seq['total_tokens']} tok in {seq['wall_time_s']}s ({seq['throughput_tps']} t/s)", file=sys.stderr)
    all_results.append(seq)

    # Concurrent: 2..max_slots
    for n in range(2, args.max_slots + 1):
        print(f"\n--- CONCURRENT x{n} ---", file=sys.stderr)
        conc = bench_concurrent(args.endpoint, n)
        for r in conc["per_request"]:
            print(
                f"  {r['label']}: {r['total_tokens']} tok in {r['time_s']}s",
                file=sys.stderr,
            )

        # Speedup vs sequential
        sequential_time = sum(r["time_s"] for r in seq["per_request"][:n])
        speedup = round(sequential_time / conc["wall_time_s"], 2) if conc["wall_time_s"] > 0 else 0
        conc["speedup_vs_sequential"] = speedup

        print(
            f"  Wall: {conc['wall_time_s']}s | Combined: {conc['throughput_tps']} t/s | Speedup: {speedup}x",
            file=sys.stderr,
        )
        all_results.append(conc)

    output = {
        "resource": label,
        "endpoint": args.endpoint,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results": all_results,
    }

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
