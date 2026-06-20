#!/usr/bin/env python3
"""
Prefill-focused benchmark for RAG / log analysis workloads (OpenShift Lightspeed style).

Characteristics of this workload:
- Large contexts (4K-32K tokens of logs/RAG documents)
- Heavy prompt processing (prefill dominates cost)
- Short generation (summaries, classifications, answers)
- Concurrent queries from multiple users
- Time-to-first-token (TTFT) is the user-visible latency

Usage:
  python bench_prefill_rag.py --endpoint http://... --concurrent 4 --ctx 16384

Output: JSON with prefill throughput, TTFT percentiles, concurrent scaling.
"""

import argparse
import asyncio
import json
import random
import sys
import time

import aiohttp

ENDPOINT = None
MODEL = None

# Realistic log line templates for RAG simulation
LOG_TEMPLATES = [
    "2026-06-17T{:02d}:{:02d}:{:02d}.{:03d}Z [INFO] pod/{}-{} starting container...",
    "2026-06-17T{:02d}:{:02d}:{:02d}.{:03d}Z [DEBUG] loading config from /etc/{}/config.yaml",
    "2026-06-17T{:02d}:{:02d}:{:02d}.{:03d}Z [WARN] connection to {}:{} timed out after {}ms",
    "2026-06-17T{:02d}:{:02d}:{:02d}.{:03d}Z [ERROR] failed to reconcile {}: {}",
    "2026-06-17T{:02d}:{:02d}:{:02d}.{:03d}Z [INFO] route {} -> {} created successfully",
]


def make_rag_context(target_tokens: int) -> str:
    """Generate realistic RAG-style context (log lines + document fragments)."""
    lines = []
    tokens = 0
    # ~25-30 tokens per log line on average
    target_lines = max(10, target_tokens // 28)

    for i in range(target_lines):
        template = random.choice(LOG_TEMPLATES)
        hour = random.randint(8, 18)
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        ms = random.randint(0, 999)
        line = template.format(
            hour, minute, second, ms,
            random.choice(["nginx", "redis", "postgres", "api", "worker"]),
            random.randint(1000, 9999),
            random.randint(1, 5000),
            random.choice(["timeout", "connection refused", "auth failed", "OOM"]),
            random.choice(["deployment", "statefulset", "daemonset"]),
        )
        lines.append(line)
        tokens += len(line.split())

        if tokens >= target_tokens:
            break

    # Add some RAG document flavor at the end
    doc_fragment = (
        "\n\nRelevant documentation excerpt:\n"
        "When troubleshooting pod startup failures, check:\n"
        "1. Image pull errors (check registry credentials and network policy)\n"
        "2. Resource limits (CPU/memory requests vs node capacity)\n"
        "3. Init container failures (look at previous container logs)\n"
        "4. Volume mount issues (check PersistentVolume and StorageClass)\n"
    )
    lines.append(doc_fragment)

    return "\n".join(lines)


async def prefill_request(
    session: aiohttp.ClientSession,
    context_size: int,
    max_gen_tokens: int = 5,
) -> dict:
    """
    Send a single request optimized for prefill measurement.
    Uses very small generation to isolate prefill time.
    """
    context = make_rag_context(context_size)
    start = time.perf_counter()

    try:
        async with session.post(
            f"{ENDPOINT}/chat/completions",
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": context},
                    {"role": "user", "content": "Summarize the key issues and recommended actions."},
                ],
                "max_tokens": max_gen_tokens,
                "temperature": 0.1,
            },
            timeout=aiohttp.ClientTimeout(total=300),
        ) as resp:
            data = await resp.json()
            elapsed = time.perf_counter() - start

            usage = data.get("usage", {})
            pt = usage.get("prompt_tokens", 0)
            ct = usage.get("completion_tokens", 0)

            # TTFT approximation: total time for tiny generation
            # (decode of 5 tokens is negligible compared to prefill of 8K-32K)
            prefill_tps = pt / elapsed if elapsed > 0 else 0

            return {
                "success": True,
                "prompt_tokens": pt,
                "completion_tokens": ct,
                "elapsed_s": round(elapsed, 3),
                "prefill_tps": round(prefill_tps, 1),
            }
    except Exception as e:
        elapsed = time.perf_counter() - start
        return {
            "success": False,
            "error": str(e),
            "elapsed_s": round(elapsed, 3),
        }


async def run_concurrent_prefill(
    num_requests: int,
    context_size: int,
    max_gen_tokens: int = 5,
) -> dict:
    """Fire N concurrent prefill-heavy requests."""
    async with aiohttp.ClientSession() as session:
        tasks = [
            prefill_request(session, context_size, max_gen_tokens)
            for _ in range(num_requests)
        ]
        results = await asyncio.gather(*tasks)

    successful = [r for r in results if r.get("success")]
    failed = len(results) - len(successful)

    if not successful:
        return {"error": "all requests failed", "failed": failed}

    total_prompt = sum(r["prompt_tokens"] for r in successful)
    total_time = max(r["elapsed_s"] for r in successful)  # wall clock
    aggregate_tps = total_prompt / total_time if total_time > 0 else 0

    # Per-request stats
    tps_values = [r["prefill_tps"] for r in successful]
    tps_values.sort()

    def pct(p):
        idx = int(len(tps_values) * p / 100)
        return round(tps_values[min(idx, len(tps_values) - 1)], 1)

    return {
        "requests": num_requests,
        "successful": len(successful),
        "failed": failed,
        "context_size": context_size,
        "wall_clock_s": round(total_time, 2),
        "total_prompt_tokens": total_prompt,
        "aggregate_prefill_tps": round(aggregate_tps, 1),
        "per_request": {
            "p50_tps": pct(50),
            "p95_tps": pct(95),
            "p99_tps": pct(99),
            "min_tps": round(min(tps_values), 1),
            "max_tps": round(max(tps_values), 1),
        },
    }


async def main():
    parser = argparse.ArgumentParser(description="Prefill benchmark for RAG workloads")
    parser.add_argument("--endpoint", "-e", required=True)
    parser.add_argument("--model", "-m", default="/run/host/AITOOLCHAIN/models/Qwen3.6-27B-FP8")
    parser.add_argument("--concurrent", "-c", type=int, default=4)
    parser.add_argument("--ctx", type=int, default=16384)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    global ENDPOINT, MODEL
    ENDPOINT = args.endpoint
    MODEL = args.model

    print(f"Prefill benchmark: {args.concurrent} concurrent, {args.ctx} token contexts", file=sys.stderr)
    print(f"Endpoint: {ENDPOINT}", file=sys.stderr)
    print(f"Model: {MODEL}\n", file=sys.stderr)

    all_runs = []
    for run in range(1, args.runs + 1):
        print(f"Run {run}/{args.runs}...", file=sys.stderr)
        result = await run_concurrent_prefill(args.concurrent, args.ctx)
        print(f"  aggregate: {result.get('aggregate_prefill_tps', 0)} t/s "
              f"(p50={result.get('per_request', {}).get('p50_tps', 0)})", file=sys.stderr)
        all_runs.append(result)

    # Summary
    agg_values = [r["aggregate_prefill_tps"] for r in all_runs if "aggregate_prefill_tps" in r]
    avg_agg = sum(agg_values) / len(agg_values) if agg_values else 0

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"SUMMARY: {args.concurrent} concurrent RAG queries @ {args.ctx} tokens", file=sys.stderr)
    print(f"  Average aggregate prefill: {avg_agg:.1f} t/s", file=sys.stderr)
    print(f"  Runs: {agg_values}", file=sys.stderr)
    print(f"{'='*60}\n", file=sys.stderr)

    print(json.dumps({
        "workload": "RAG / log analysis (OpenShift Lightspeed style)",
        "endpoint": ENDPOINT,
        "model": MODEL,
        "concurrent_requests": args.concurrent,
        "context_size": args.ctx,
        "runs": args.runs,
        "summary": {
            "avg_aggregate_prefill_tps": round(avg_agg, 1),
            "individual_runs": all_runs,
        },
    }, indent=2))


if __name__ == "__main__":
    asyncio.run(main())


def bench_single_prefill(endpoint: str, context_size: int, runs: int = 3):
    """Synchronous single-request prefill benchmark (for quick checks)."""
    import requests

    results = []
    for i in range(runs):
        context = make_rag_context(context_size)
        start = time.time()
        resp = requests.post(
            f"{endpoint}/chat/completions",
            json={
                "model": MODEL or "/run/host/AITOOLCHAIN/models/Qwen3.6-27B-FP8",
                "messages": [
                    {"role": "system", "content": context},
                    {"role": "user", "content": "Summarize key issues."},
                ],
                "max_tokens": 5,
                "temperature": 0.1,
            },
            timeout=300,
        )
        elapsed = time.time() - start
        data = resp.json()
        pt = data.get("usage", {}).get("prompt_tokens", 0)
        tps = pt / elapsed if elapsed > 0 else 0
        results.append({"run": i + 1, "prompt_tokens": pt, "time_s": round(elapsed, 3), "prefill_tps": round(tps, 1)})

    avg_tps = sum(r["prefill_tps"] for r in results) / len(results)
    print(f"Single-request prefill @ {context_size} tokens: {avg_tps:.1f} t/s avg ({runs} runs)", file=sys.stderr)
    return {"context_size": context_size, "avg_prefill_tps": round(avg_tps, 1), "runs": results}
