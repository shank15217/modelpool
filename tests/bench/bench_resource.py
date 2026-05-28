#!/usr/bin/env python3
"""ModelPool benchmark: prompt eval + generation speed at multiple context sizes.

Usage:
  python bench_resource.py [--endpoint URL] [--resource NAME] [--max-ctx TOKENS]

Tests:
  1. Prompt eval at 1K, 4K, 8K, 16K, 32K context (short generation)
  2. Generation speed at small context (64, 256, 512 tokens)
  3. Real-world tasks (title gen, triage, summarize, code review)

Output: JSON results to stdout, human-readable summary to stderr.
"""

import argparse
import json
import sys
import time

import requests


def make_context(target_tokens: int) -> str:
    """Generate ~target_tokens of filler context by repeating a chunk."""
    chunk = (
        "System check: all nodes operational. CPU nominal. Memory stable. "
        "Network latency 2ms. Disk IO normal. OVN flows active. "
        "WireGuard peers connected. "
    )
    # ~25 tokens per chunk
    n_chunks = max(1, target_tokens // 25)
    return chunk * n_chunks


def chat(endpoint: str, messages: list, max_tokens: int = 256, temperature: float = 0.3) -> dict:
    """Send a chat completion request and return timing + usage."""
    start = time.time()
    resp = requests.post(
        f"{endpoint}/v1/chat/completions",
        json={
            "model": "bench",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        timeout=600,
    )
    elapsed = time.time() - start
    data = resp.json()
    usage = data.get("usage", {})
    pt = usage.get("prompt_tokens", 0)
    ct = usage.get("completion_tokens", 0)
    content = data["choices"][0]["message"]["content"][:100].replace("\n", " ")
    return {
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "total_tokens": pt + ct,
        "time_s": round(elapsed, 2),
        "preview": content,
    }


def bench_prompt_eval(endpoint: str, max_ctx: int) -> list:
    """Benchmark prompt processing at increasing context sizes."""
    results = []
    context_sizes = [1000, 4000, 8000, 16000, 32000]
    # Filter to max_ctx
    context_sizes = [c for c in context_sizes if c <= max_ctx]

    for target in context_sizes:
        context = make_context(target)
        messages = [
            {"role": "system", "content": context},
            {"role": "user", "content": "Summarize the above in one short sentence."},
        ]
        r = chat(endpoint, messages, max_tokens=20)
        prompt_tps = r["prompt_tokens"] / r["time_s"] if r["time_s"] > 0 else 0
        r["test"] = f"prompt_eval_{target // 1000}K"
        r["prompt_eval_tps"] = round(prompt_tps, 1)
        results.append(r)
        print(
            f"  {r['test']:12s}: {r['prompt_tokens']:>6,}+{r['completion_tokens']:>3} "
            f"in {r['time_s']:>6.1f}s | prompt eval: {prompt_tps:>7.1f} t/s",
            file=sys.stderr,
        )
    return results


def bench_generation(endpoint: str) -> list:
    """Benchmark generation speed at small context (prompt eval is minimal)."""
    results = []
    gen_sizes = [64, 256, 512]

    for max_out in gen_sizes:
        messages = [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {
                "role": "user",
                "content": "Write a Python implementation of a binary search tree "
                "with insert, search, and delete operations.",
            },
        ]
        r = chat(endpoint, messages, max_tokens=max_out)
        gen_tps = r["completion_tokens"] / r["time_s"] if r["time_s"] > 0 else 0
        r["test"] = f"generation_{max_out}"
        r["generation_tps"] = round(gen_tps, 1)
        results.append(r)
        print(
            f"  gen {max_out:>4} tok: {r['prompt_tokens']:>4}+{r['completion_tokens']:>4} "
            f"in {r['time_s']:>5.1f}s | gen: {gen_tps:>5.1f} t/s",
            file=sys.stderr,
        )
    return results


def bench_realworld(endpoint: str) -> list:
    """Benchmark real-world tasks."""
    tasks = [
        {
            "test": "title_gen_3K",
            "label": "Title generation (3K ctx)",
            "context": make_context(3000),
            "user": "Generate a short title (5-8 words) for this conversation.",
            "max_tokens": 20,
        },
        {
            "test": "triage_2K",
            "label": "Triage/classify (2K ctx)",
            "context": make_context(2000),
            "user": "Classify this text into one of: technical, social, administrative, alert.",
            "max_tokens": 10,
        },
        {
            "test": "summarize_8K",
            "label": "Summarize (8K ctx)",
            "context": make_context(8000),
            "user": "Summarize the key points from the above in 2-3 sentences.",
            "max_tokens": 100,
        },
        {
            "test": "code_review",
            "label": "Code review (~120 tok)",
            "context": "",
            "user": (
                "Review this code and identify bugs:\n\n"
                "def fibonacci(n):\n    if n <= 0:\n        return 1\n"
                "    return fibonacci(n-1) + fibonacci(n-2)\n\n"
                "def sort_list(items):\n    return sorted(items, reverse=False)\n\n"
                "class Cache:\n    def __init__(self):\n        self.cache = {}\n"
                "    def get(self, key):\n        return self.cache[key]\n"
                "    def set(self, key, value):\n        self.cache = {key: value}"
            ),
            "max_tokens": 200,
        },
    ]

    results = []
    for task in tasks:
        messages = [
            {"role": "system", "content": task["context"]},
            {"role": "user", "content": task["user"]},
        ]
        r = chat(endpoint, messages, max_tokens=task["max_tokens"])
        r["test"] = task["test"]
        overall = r["total_tokens"] / r["time_s"] if r["time_s"] > 0 else 0
        r["overall_tps"] = round(overall, 1)
        results.append(r)
        print(
            f"  {task['label']:28s}: {r['prompt_tokens']:>5,}+{r['completion_tokens']:>3} "
            f"in {r['time_s']:>5.1f}s ({overall:>5.1f} t/s)",
            file=sys.stderr,
        )
    return results


def main():
    parser = argparse.ArgumentParser(description="ModelPool resource benchmark")
    parser.add_argument(
        "--endpoint", "-e",
        default="http://localhost:8080",
        help="Inference endpoint URL (default: http://localhost:8080)",
    )
    parser.add_argument(
        "--resource", "-r",
        default=None,
        help="Resource name (for labeling results)",
    )
    parser.add_argument(
        "--max-ctx",
        type=int,
        default=32000,
        help="Maximum context size to test (default: 32000)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output only JSON results to stdout",
    )
    args = parser.parse_args()

    label = args.resource or args.endpoint

    if not args.json:
        print(f"{'=' * 70}", file=sys.stderr)
        print(f"Benchmark: {label}", file=sys.stderr)
        print(f"Endpoint:  {args.endpoint}", file=sys.stderr)
        print(f"{'=' * 70}", file=sys.stderr)

    all_results = []

    if not args.json:
        print(f"\n--- PROMPT EVAL ---", file=sys.stderr)
    all_results.extend(bench_prompt_eval(args.endpoint, args.max_ctx))

    if not args.json:
        print(f"\n--- GENERATION ---", file=sys.stderr)
    all_results.extend(bench_generation(args.endpoint))

    if not args.json:
        print(f"\n--- REAL-WORLD TASKS ---", file=sys.stderr)
    all_results.extend(bench_realworld(args.endpoint))

    # Compute summary
    prompt_tests = [r for r in all_results if r["test"].startswith("prompt_eval")]
    gen_tests = [r for r in all_results if r["test"].startswith("generation")]

    # Use mid-range prompt eval (8K or closest)
    pe_summary = None
    for size in [8000, 4000, 1000]:
        match = next((r for r in prompt_tests if f"_{size // 1000}K" in r["test"]), None)
        if match:
            pe_summary = match["prompt_eval_tps"]
            break

    # Use longest generation test for gen speed
    gen_summary = None
    if gen_tests:
        gen_summary = gen_tests[-1]["generation_tps"]

    output = {
        "resource": label,
        "endpoint": args.endpoint,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "summary": {
            "prompt_eval_tps": pe_summary,
            "generation_tps": gen_summary,
        },
        "results": all_results,
    }

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
