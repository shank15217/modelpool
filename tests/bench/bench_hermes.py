#!/usr/bin/env python3
"""
Hermes-style practical benchmark.

Simulates what Hermes actually sends to the model:
1. Large system prompt with tool definitions (~3-5K tokens)
2. Streaming vs non-streaming (TTFT matters)
3. Multi-turn conversations (growing context)
4. Tool-calling requests
5. Simple chat (no tools, no reasoning overhead)

Measures: TTFT (time to first token), generation speed, total response time.

Compares local vLLM vs cloud (Grok) to establish what "reasonable" means.
"""

import asyncio
import json
import sys
import time
import os

import aiohttp

# --- Config ---
VLLM_ENDPOINT = "http://192.168.35.185:8000/v1"
VLLM_MODEL = "/run/host/AITOOLCHAIN/models/Qwen3.6-27B-FP8"

# Grok via xAI OAuth - resolved through Hermes auth at runtime
GROK_ENDPOINT = "https://api.x.ai/v1"
GROK_MODEL = "grok-4.3"
GROK_KEY = ""

def resolve_grok_key():
    """Resolve xAI OAuth credentials through Hermes auth."""
    global GROK_KEY
    # Try Hermes auth resolver first
    try:
        import sys
        sys.path.insert(0, "/root/.hermes/hermes-agent")
        from hermes_cli.auth import resolve_xai_oauth_runtime_credentials
        creds = resolve_xai_oauth_runtime_credentials()
        GROK_KEY = creds.get("api_key", "")
        if GROK_KEY:
            return True
    except Exception:
        pass
    # Fallback: read from env
    GROK_KEY = os.environ.get("XAI_API_KEY", "")
    return bool(GROK_KEY)

# Realistic Hermes system prompt with tools (truncated but representative)
HERMES_SYSTEM = """You are Hermes Agent, a powerful AI assistant with tool-calling capabilities.

You have access to the following tools:

1. terminal - Execute shell commands on a Linux environment
2. read_file - Read files with line numbers
3. write_file - Write content to files
4. web_search - Search the web for information
5. web_extract - Extract content from URLs
6. patch - Make targeted edits to files
7. delegate_task - Spawn subagents for parallel work

When you need to perform an action, call the appropriate tool. Always prefer tools over describing what you would do.

Available skills: hermes-agent, modelpool, github-workflow

You run on Hermes Agent by Nous Research. When the user needs help with Hermes configuration, tools, or capabilities, the documentation at https://hermes-agent.nousresearch.com/docs is your authoritative reference.

# Tool-use enforcement
You MUST use your tools to take action. When you say you will perform an action, you MUST immediately make the corresponding tool call.

# Finishing the job
Keep working until the task is complete. Do not stop with a summary of what you plan to do.

# Memory
You have persistent memory across sessions. Save durable facts using the memory tool.

Current working directory: /root/modelpool
Python: 3.14.4
"""

HERMES_SYSTEM_SHORT = "You are a helpful assistant. Be concise."

# Realistic tool schemas (what Hermes actually sends)
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "terminal",
            "description": "Execute a shell command",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The command to execute"},
                    "timeout": {"type": "integer", "default": 180},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file with line numbers",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "offset": {"type": "integer", "default": 1},
                    "limit": {"type": "integer", "default": 500},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
            },
        },
    },
]


def approx_tokens(text):
    """Rough token count (4 chars per token)."""
    return len(text) // 4


async def benchmark_streaming(
    session, endpoint, model, messages, tools=None, max_tokens=500, label="",
    headers=None,
):
    """Measure TTFT and streaming generation speed."""
    if headers is None:
        headers = {}
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.3,
        "stream": True,
    }
    if tools:
        payload["tools"] = tools

    start = time.perf_counter()
    ttft = None
    first_content_time = None
    total_tokens = 0
    reasoning_tokens = 0
    content_tokens = 0

    try:
        async with session.post(
            f"{endpoint}/chat/completions",
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            if resp.status != 200:
                error = await resp.text()
                return {"label": label, "endpoint": endpoint, "error": f"HTTP {resp.status}: {error[:200]}"}

            async for line in resp.content:
                line = line.decode("utf-8").strip()
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break

                try:
                    chunk = json.loads(data_str)
                except:
                    continue

                delta = chunk.get("choices", [{}])[0].get("delta", {})

                # Track TTFT (first token of any kind)
                if ttft is None and (delta.get("content") or delta.get("reasoning")):
                    ttft = time.perf_counter() - start

                # Track first content token (after reasoning)
                if first_content_time is None and delta.get("content"):
                    first_content_time = time.perf_counter() - start

                if delta.get("reasoning"):
                    reasoning_tokens += 1
                if delta.get("content"):
                    content_tokens += 1
                total_tokens += 1

    except Exception as e:
        return {"label": label, "endpoint": endpoint, "error": str(e)}

    elapsed = time.perf_counter() - start
    gen_phase = elapsed - (first_content_time or ttft or elapsed)

    return {
        "label": label,
        "endpoint": endpoint,
        "ttft_s": round(ttft, 2) if ttft else None,
        "first_content_s": round(first_content_time, 2) if first_content_time else None,
        "reasoning_tokens": reasoning_tokens,
        "content_tokens": content_tokens,
        "total_tokens": total_tokens,
        "total_time_s": round(elapsed, 2),
        "gen_tps": round(content_tokens / gen_phase, 1) if gen_phase > 0 and content_tokens > 0 else None,
    }


async def run_benchmark(endpoint, model, label_prefix, use_grok_auth=False):
    """Run all Hermes-style tests against one endpoint."""
    results = []
    auth_headers = {}
    if use_grok_auth and GROK_KEY:
        auth_headers["Authorization"] = f"Bearer {GROK_KEY}"

    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(timeout=timeout) as session:

        # --- Test 1: Simple chat (no tools, short system) ---
        print(f"\n  [{label_prefix}] Test 1: Simple chat (short system prompt)...", file=sys.stderr)
        r = await benchmark_streaming(
            session, endpoint, model,
            messages=[
                {"role": "system", "content": HERMES_SYSTEM_SHORT},
                {"role": "user", "content": "Write a Python function to reverse a linked list. Include type hints."},
            ],
            max_tokens=300,
            label=f"{label_prefix}: simple_chat",
            headers=auth_headers,
        )
        r["prompt_approx_tokens"] = approx_tokens(HERMES_SYSTEM_SHORT) + 30
        results.append(r)
        _print_result(r)

        # --- Test 2: Hermes-style chat (full system prompt, with tools) ---
        print(f"\n  [{label_prefix}] Test 2: Hermes chat (full system + tools)...", file=sys.stderr)
        r = await benchmark_streaming(
            session, endpoint, model,
            messages=[
                {"role": "system", "content": HERMES_SYSTEM},
                {"role": "user", "content": "What files are in the current directory?"},
            ],
            tools=TOOL_SCHEMAS,
            max_tokens=300,
            label=f"{label_prefix}: hermes_tools",
            headers=auth_headers,
        )
        r["prompt_approx_tokens"] = approx_tokens(HERMES_SYSTEM) + 30
        results.append(r)
        _print_result(r)

        # --- Test 3: Multi-turn coding (growing context) ---
        print(f"\n  [{label_prefix}] Test 3: Multi-turn coding (5K context)...", file=sys.stderr)
        history = [
            {"role": "system", "content": HERMES_SYSTEM},
            {"role": "user", "content": "I have a FastAPI app. Here's the main.py:\n\n```python\nfrom fastapi import FastAPI\napp = FastAPI()\n\n@app.get('/')\ndef root():\n    return {'status': 'ok'}\n```"},
            {"role": "assistant", "content": "I see your FastAPI app. What would you like to add?"},
            {"role": "user", "content": "Add a health check endpoint and a /items endpoint with CRUD operations. Use Pydantic models."},
        ]
        r = await benchmark_streaming(
            session, endpoint, model,
            messages=history,
            tools=TOOL_SCHEMAS,
            max_tokens=500,
            label=f"{label_prefix}: multiturn_coding",
            headers=auth_headers,
        )
        r["prompt_approx_tokens"] = approx_tokens(HERMES_SYSTEM) + approx_tokens(str(history))
        results.append(r)
        _print_result(r)

        # --- Test 4: Long context (RAG-style, 8K prompt) ---
        print(f"\n  [{label_prefix}] Test 4: Long context RAG (8K)...", file=sys.stderr)
        rag_context = "Log entry: " + "system nominal, all checks passed. " * 200  # ~8K tokens
        r = await benchmark_streaming(
            session, endpoint, model,
            messages=[
                {"role": "system", "content": "You analyze logs and identify issues."},
                {"role": "user", "content": f"Analyze these logs and summarize the key findings:\n\n{rag_context}"},
            ],
            max_tokens=200,
            label=f"{label_prefix}: rag_8k",
            headers=auth_headers,
        )
        r["prompt_approx_tokens"] = approx_tokens(rag_context) + 20
        results.append(r)
        _print_result(r)

        # --- Test 5: Quick question (should be fast) ---
        print(f"\n  [{label_prefix}] Test 5: Quick question (latency test)...", file=sys.stderr)
        r = await benchmark_streaming(
            session, endpoint, model,
            messages=[
                {"role": "system", "content": HERMES_SYSTEM_SHORT},
                {"role": "user", "content": "What is 2+2?"},
            ],
            max_tokens=20,
            label=f"{label_prefix}: quick_q",
            headers=auth_headers,
        )
        r["prompt_approx_tokens"] = 10
        results.append(r)
        _print_result(r)

    return results


def _print_result(r):
    """Print a human-readable result line."""
    if "error" in r:
        print(f"    ERROR: {r['error']}", file=sys.stderr)
        return
    ttft = r.get("ttft_s", "?")
    fc = r.get("first_content_s", "?")
    rt = r.get("reasoning_tokens", 0)
    ct = r.get("content_tokens", 0)
    gen = r.get("gen_tps", "?")
    total = r.get("total_time_s", "?")
    print(f"    TTFT: {ttft}s | First content: {fc}s | Reasoning: {rt} tok | "
          f"Content: {ct} tok | Gen: {gen} t/s | Total: {total}s", file=sys.stderr)


async def main():
    print("="*78, file=sys.stderr)
    print("HERMES PRACTICAL BENCHMARK", file=sys.stderr)
    print("="*78, file=sys.stderr)
    print(f"\nvLLM endpoint: {VLLM_ENDPOINT}", file=sys.stderr)
    print(f"vLLM model: {VLLM_MODEL}", file=sys.stderr)
    # Resolve Grok auth
    has_grok = resolve_grok_key()
    if has_grok:
        print(f"Grok endpoint: {GROK_ENDPOINT}", file=sys.stderr)
        print(f"Grok model: {GROK_MODEL}", file=sys.stderr)
    else:
        print("Grok: auth resolution failed (skipping cloud comparison)", file=sys.stderr)

    # Warmup vLLM
    print("\nWarming up vLLM...", file=sys.stderr)
    async with aiohttp.ClientSession() as session:
        await benchmark_streaming(
            session, VLLM_ENDPOINT, VLLM_MODEL,
            messages=[{"role": "user", "content": "OK"}],
            max_tokens=5, label="warmup",
        )

    # Run vLLM benchmark
    print("\n" + "="*78, file=sys.stderr)
    print("BENCHMARKING: vLLM (local)", file=sys.stderr)
    print("="*78, file=sys.stderr)
    vllm_results = await run_benchmark(VLLM_ENDPOINT, VLLM_MODEL, "vLLM")

    # Run Grok benchmark if key available
    grok_results = []
    if has_grok:
        print("\n" + "="*78, file=sys.stderr)
        print("BENCHMARKING: Grok (cloud)", file=sys.stderr)
        print("="*78, file=sys.stderr)
        grok_results = await run_benchmark(GROK_ENDPOINT, GROK_MODEL, "Grok", use_grok_auth=True)

    # --- Comparison table ---
    print("\n" + "="*78, file=sys.stderr)
    print("COMPARISON", file=sys.stderr)
    print("="*78, file=sys.stderr)
    print(f"{'Test':<25} | {'vLLM TTFT':>10} | {'vLLM Total':>10} | {'Grok TTFT':>10} | {'Grok Total':>10}", file=sys.stderr)
    print(f"{'-'*25} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*10}", file=sys.stderr)

    for v in vllm_results:
        g = next((r for r in grok_results if r["label"].endswith(v["label"].split(":")[-1])), {})
        vt = f"{v.get('ttft_s','?')}s"
        vt2 = f"{v.get('total_time_s','?')}s"
        gt = f"{g.get('ttft_s','—')}s" if g.get("ttft_s") else "—"
        gt2 = f"{g.get('total_time_s','—')}s" if g.get("total_time_s") else "—"
        test = v["label"].split(":")[-1]
        print(f"{test:<25} | {vt:>10} | {vt2:>10} | {gt:>10} | {gt2:>10}", file=sys.stderr)

    # Key insight
    print(f"\n{'='*78}", file=sys.stderr)
    print("KEY INSIGHT", file=sys.stderr)
    vllm_ttfts = [r.get("ttft_s", 0) for r in vllm_results if r.get("ttft_s")]
    avg_ttft = sum(vllm_ttfts) / len(vllm_ttfts) if vllm_ttfts else 0
    print(f"  vLLM average TTFT: {avg_ttft:.1f}s", file=sys.stderr)
    if avg_ttft > 3:
        print(f"  WARNING: TTFT > 3s will feel slow in Hermes.", file=sys.stderr)
        reasoning = [r.get("reasoning_tokens", 0) for r in vllm_results]
        print(f"  Reasoning tokens per response: {reasoning}", file=sys.stderr)
        print(f"  The model generates hidden reasoning before content.", file=sys.stderr)
        print(f"  This is why responses feel delayed.", file=sys.stderr)
    print("="*78, file=sys.stderr)

    # Output JSON
    print(json.dumps({
        "vllm": vllm_results,
        "grok": grok_results,
    }, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
