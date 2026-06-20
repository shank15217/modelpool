#!/usr/bin/env python3
"""
Isolate prefill vs decode slowdown after large context.

Tests generation speed at different context sizes to find the degradation curve.
Also tests with/without prefix caching by using unique vs repeated prompts.
"""

import json
import sys
import time
import requests

ENDPOINT = "http://192.168.35.185:8000/v1"
MODEL = "/run/host/AITOOLCHAIN/models/Qwen3.6-27B-FP8"


def make_context(target_tokens):
    """Generate realistic filler context."""
    chunk = (
        "System check: all nodes operational. CPU nominal. Memory stable. "
        "Network latency 2ms. Disk IO normal. OVN flows active. "
        "WireGuard peers connected. "
    )
    return chunk * max(1, target_tokens // 25)


def timed_request(messages, max_tokens, label):
    """Send a request and measure prefill + decode separately."""
    start = time.perf_counter()
    resp = requests.post(
        f"{ENDPOINT}/chat/completions",
        json={
            "model": MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.3,
            "stream": True,
        },
        stream=True,
        timeout=300,
    )

    ttft = None
    total_chunks = 0
    first_chunk_time = None

    for line in resp.iter_lines():
        if not line:
            continue
        line = line.decode("utf-8", errors="replace")
        if not line.startswith("data: "):
            continue
        data_str = line[6:]
        if data_str.strip() == "[DONE]":
            break

        now = time.perf_counter()
        if ttft is None:
            ttft = now - start
            first_chunk_time = now

        total_chunks += 1

    elapsed = time.perf_counter() - start
    decode_time = elapsed - (ttft or elapsed)

    # Get usage stats from the last chunk or final response
    usage = {}
    try:
        # Non-streaming request to get accurate token counts
        pass
    except:
        pass

    result = {
        "label": label,
        "ttft_s": round(ttft, 3) if ttft else None,
        "total_time_s": round(elapsed, 3),
        "decode_time_s": round(decode_time, 3) if decode_time > 0 else None,
        "content_tokens": total_chunks,
        "decode_tps": round(total_chunks / decode_time, 1) if decode_time and total_chunks > 0 and decode_time > 0 else None,
    }

    print(f"  {label:40s}: TTFT={result['ttft_s']}s  decode={result['decode_tps']} t/s  "
          f"({total_chunks} tok in {result['decode_time_s']}s)  total={result['total_time_s']}s",
          file=sys.stderr)
    return result


def main():
    print("=" * 80, file=sys.stderr)
    print("PREFILL vs DECODE ISOLATION BENCHMARK", file=sys.stderr)
    print(f"Endpoint: {ENDPOINT}", file=sys.stderr)
    print(f"Model: {MODEL}", file=sys.stderr)
    print("=" * 80, file=sys.stderr)

    # Warmup
    print("\nWarming up...", file=sys.stderr)
    timed_request(
        [{"role": "user", "content": "Hello"}],
        max_tokens=10,
        label="warmup",
    )

    results = []

    # === Test 1: Baseline generation (minimal context, 256 tokens) ===
    print("\n--- BASELINE: Minimal context generation ---", file=sys.stderr)
    for run in range(3):
        r = timed_request(
            [{"role": "user", "content": "Write a Python function to reverse a string. Include docstring."}],
            max_tokens=256,
            label=f"baseline_0K_run{run+1}",
        )
        results.append(r)

    # === Test 2: Generation at increasing context sizes ===
    print("\n--- DECODE SPEED vs CONTEXT SIZE (512 token generation) ---", file=sys.stderr)
    for ctx_size in [1000, 4000, 8000, 16000, 32000]:
        context = make_context(ctx_size)
        r = timed_request(
            [
                {"role": "system", "content": context},
                {"role": "user", "content": "Write a Python function to reverse a string. Include docstring."},
            ],
            max_tokens=512,
            label=f"decode_ctx_{ctx_size//1000}K",
        )
        results.append(r)

    # === Test 3: Same context, varying generation length ===
    print("\n--- GENERATION LENGTH IMPACT (8K context) ---", file=sys.stderr)
    context_8k = make_context(8000)
    for gen_len in [64, 128, 256, 512]:
        r = timed_request(
            [
                {"role": "system", "content": context_8k},
                {"role": "user", "content": "Write a Python function to reverse a string. Include docstring."},
            ],
            max_tokens=gen_len,
            label=f"gen_{gen_len}_at_8K",
        )
        results.append(r)

    # === Test 4: Prefix caching impact (same prompt repeated) ===
    print("\n--- PREFIX CACHE IMPACT (repeated prompt) ---", file=sys.stderr)
    context_8k_b = make_context(8000)
    prompt = "Write a Python function to reverse a string. Include docstring."

    # First request (cold cache)
    r = timed_request(
        [
            {"role": "system", "content": context_8k_b},
            {"role": "user", "content": prompt},
        ],
        max_tokens=256,
        label="prefix_cold",
    )
    results.append(r)

    # Second request (warm cache, same prefix)
    r = timed_request(
        [
            {"role": "system", "content": context_8k_b},
            {"role": "user", "content": prompt},
        ],
        max_tokens=256,
        label="prefix_warm",
    )
    results.append(r)

    # Third request (warm cache, different suffix)
    r = timed_request(
        [
            {"role": "system", "content": context_8k_b},
            {"role": "user", "content": "Write a Python function to check if a string is a palindrome. Include docstring."},
        ],
        max_tokens=256,
        label="prefix_warm_diff_suffix",
    )
    results.append(r)

    # === Summary ===
    print("\n" + "=" * 80, file=sys.stderr)
    print("SUMMARY: Decode speed vs context size", file=sys.stderr)
    print("=" * 80, file=sys.stderr)
    print(f"  {'Context':>8} | {'Decode t/s':>12} | {'TTFT':>8} | {'Total':>8}", file=sys.stderr)
    print(f"  {'-'*8} | {'-'*12} | {'-'*8} | {'-'*8}", file=sys.stderr)

    for r in results:
        if r["label"].startswith("decode_ctx_") or r["label"].startswith("baseline_"):
            ctx = r["label"].replace("decode_ctx_", "").replace("baseline_", "")
            tps = r.get("decode_tps", "?")
            ttft = r.get("ttft_s", "?")
            total = r.get("total_time_s", "?")
            print(f"  {ctx:>8} | {tps:>10.1f} | {ttft:>7.2f}s | {total:>7.2f}s", file=sys.stderr)

    print(f"\n{'='*80}", file=sys.stderr)
    print("SUMMARY: Prefix caching impact", file=sys.stderr)
    print(f"{'='*80}", file=sys.stderr)
    for r in results:
        if r["label"].startswith("prefix_"):
            tps = r.get("decode_tps", "?")
            ttft = r.get("ttft_s", "?")
            total = r.get("total_time_s", "?")
            print(f"  {r['label']:25s}: decode={tps} t/s, TTFT={ttft}s, total={total}s", file=sys.stderr)

    print(f"\n{'='*80}", file=sys.stderr)

    print(json.dumps({"results": results}, indent=2))


if __name__ == "__main__":
    main()
