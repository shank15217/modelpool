#!/usr/bin/env python3
"""Concurrent multi-agent benchmark for vLLM."""

import asyncio
import json
import sys
import time

import aiohttp

ENDPOINT = "http://192.168.35.185:8000/v1"
MODEL = "/run/host/AITOOLCHAIN/models/Qwen3.6-27B-FP8"

TASKS = [
    {"name": "coding", "system": "You are a coding assistant.",
     "prompt": "Write a Python function that implements a thread-safe LRU cache.", "max_tokens": 300},
    {"name": "review", "system": "You are a code reviewer.",
     "prompt": "Review this code for bugs:\ndef fibonacci(n):\n    if n <= 0: return 1\n    return fibonacci(n-1) + fibonacci(n-2)", "max_tokens": 150},
    {"name": "summarize", "system": "You summarize logs.",
     "prompt": "Summarize the following log excerpt in 3 bullet points.", "max_tokens": 100},
    {"name": "triage", "system": "You classify issues.",
     "prompt": "Classify: 'App crashes on Android 14 when opening settings'.", "max_tokens": 50},
]

async def send(session, task, agent_id):
    start = time.time()
    try:
        async with session.post(f"{ENDPOINT}/chat/completions", json={
            "model": MODEL,
            "messages": [{"role": "system", "content": task["system"]},
                         {"role": "user", "content": task["prompt"]}],
            "max_tokens": task["max_tokens"],
            "temperature": 0.3,
        }, timeout=aiohttp.ClientTimeout(total=300)) as resp:
            data = await resp.json()
            elapsed = time.time() - start
            usage = data.get("usage", {})
            return {
                "agent": agent_id,
                "task": task["name"],
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "time_s": round(elapsed, 2),
            }
    except Exception as e:
        return {"agent": agent_id, "task": task["name"], "error": str(e)}

async def run(n):
    tasks = [TASKS[i % len(TASKS)] for i in range(n)]
    async with aiohttp.ClientSession() as session:
        coros = [send(session, tasks[i], i) for i in range(n)]
        results = await asyncio.gather(*coros)
    total_tok = sum(r.get("prompt_tokens", 0) + r.get("completion_tokens", 0) for r in results)
    wall = max(r.get("time_s", 0) for r in results)
    tput = total_tok / wall if wall > 0 else 0
    print(f"\n{n} agents: wall={wall:.1f}s, total_tok={total_tok}, throughput={tput:.1f} t/s", file=sys.stderr)
    return {"agents": n, "wall_clock_s": round(wall, 1), "total_tokens": total_tok, "throughput_tps": round(tput, 1)}

async def main():
    print("WARMUP...", file=sys.stderr)
    async with aiohttp.ClientSession() as session:
        for i in range(3):
            await send(session, TASKS[0], 0)
    print("Warmup done.\n", file=sys.stderr)

    out = []
    for n in [1, 2, 4, 8]:
        for _ in range(3):
            res = await run(n)
            out.append(res)
    print(json.dumps({"engine": "vLLM AITER latest", "results": out}, indent=2))

asyncio.run(main())
