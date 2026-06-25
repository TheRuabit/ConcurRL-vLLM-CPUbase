"""
03_concurrency_driver.py - High-Concurrency Async Profiler Client

Issues burst requests concurrently via aiohttp while capturing 5 latency metrics:
  1. Client Serialization Time
  2. Server HTTP/Scheduler Overhead
  3. GPU Prefill Time (TTFT)
  4. GPU Decode Time
  5. Response Parsing Time

Sequentially sweeps through concurrency levels [32, 128, 256, 512, 1024].
Uses realistic 32k token context to simulate RL workload conditions.
Outputs raw performance arrays to ./result/03_concurrency_driver.json
"""

import argparse
import asyncio
import json
import os
import sys
import time

import aiohttp

CONCURRENCY_TIERS = [32, 128, 256, 512, 1024]

# Context text generator with caching
_CONTEXT_CACHE: dict[int, str] = {}

def get_context_text(target_tokens: int) -> str:
    """Generate realistic context text of approximately target_tokens length."""
    if target_tokens in _CONTEXT_CACHE:
        return _CONTEXT_CACHE[target_tokens]
    seed = (
        "System: You are a helpful AI assistant with tool access.\n\n"
        "User: Analyze this codebase for performance issues.\n\n"
        + ("def process(items):\n    return [transform(x) for x in items]\n\n" * 200)
        + "Assistant: The key bottleneck is the sequential processing loop. " * 200
    )
    chars_needed = target_tokens * 4
    if chars_needed <= len(seed):
        text = seed[:chars_needed]
    else:
        text = (seed * ((chars_needed // len(seed)) + 1))[:chars_needed]
    _CONTEXT_CACHE[target_tokens] = text
    return text


DEFAULT_CONTEXT_TOKENS = 32000
DEFAULT_PROMPT = get_context_text(DEFAULT_CONTEXT_TOKENS)


async def single_request(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict,
    request_id: int,
) -> dict:
    record = {"request_id": request_id}

    t0 = time.perf_counter()
    body = json.dumps(payload, ensure_ascii=False)
    record["client_serialization_ms"] = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    first_token_time = None
    token_times = []

    async with session.post(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        timeout=aiohttp.ClientTimeout(total=600),
    ) as resp:
        t_header = time.perf_counter()
        record["server_overhead_ms"] = (t_header - t1) * 1000

        if resp.status != 200:
            err = await resp.text()
            record["error"] = f"HTTP {resp.status}: {err[:300]}"
            record["total_ms"] = (time.perf_counter() - t1) * 1000
            return record

        full_text = ""
        async for line in resp.content:
            decoded = line.decode("utf-8", errors="ignore").strip()
            if not decoded or not decoded.startswith("data: "):
                continue
            data_str = decoded[len("data: "):]
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                token = delta.get("content", "")
                if token:
                    t_token = time.perf_counter()
                    if first_token_time is None:
                        first_token_time = t_token
                    token_times.append(t_token)
                    full_text += token
            except json.JSONDecodeError:
                continue

    t_done = time.perf_counter()

    if first_token_time is not None:
        record["gpu_prefill_ms"] = (first_token_time - t1) * 1000
    else:
        record["gpu_prefill_ms"] = None

    if len(token_times) >= 2:
        intervals = [
            (token_times[i + 1] - token_times[i]) * 1000
            for i in range(len(token_times) - 1)
        ]
        record["gpu_decode_ms"] = sum(intervals) / len(intervals)
    else:
        record["gpu_decode_ms"] = None

    record["response_parsing_ms"] = (t_done - t_header) * 1000
    record["total_ms"] = (t_done - t1) * 1000
    record["n_tokens"] = len(token_times)

    return record


async def run_tier(
    base_url: str,
    concurrency: int,
    context_tokens: int,
    max_tokens: int,
    n_samples: int,
) -> list[dict]:
    url = f"{base_url}/v1/chat/completions"
    context_text = get_context_text(context_tokens)
    payload = {
        "model": "default",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": context_text},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True,
    }

    all_results = []
    connector = aiohttp.TCPConnector(limit=concurrency + 64)
    async with aiohttp.ClientSession(connector=connector) as session:
        for batch_idx in range(n_samples):
            tasks = [
                single_request(session, url, payload, i)
                for i in range(concurrency)
            ]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in batch_results:
                if isinstance(r, dict):
                    all_results.append(r)
                else:
                    all_results.append({"error": str(r)})
            print(f"    Batch {batch_idx+1}/{n_samples} done ({concurrency} reqs)")
    return all_results


async def main():
    parser = argparse.ArgumentParser(description="ConcurRL Concurrency Driver")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--context-tokens", type=int, default=DEFAULT_CONTEXT_TOKENS,
                        help="Target context length in tokens (default: 32000)")
    parser.add_argument("--max-tokens", type=int, default=128,
                        help="Max output tokens per request")
    parser.add_argument("--samples", type=int, default=3, help="Batches per tier")
    parser.add_argument("--tiers", nargs="+", type=int, default=CONCURRENCY_TIERS)
    parser.add_argument("--output", default="./result/03_concurrency_driver.json")
    args = parser.parse_args()

    print("=" * 60)
    print("  ConcurRL Concurrency Driver")
    print("=" * 60)
    print(f"  Base URL:       {args.base_url}")
    print(f"  Context Tokens: {args.context_tokens:,}")
    print(f"  Max Output:     {args.max_tokens}")
    print(f"  Tiers:          {args.tiers}")
    print(f"  Samples:        {args.samples} batches per tier")
    print(f"  Output:         {args.output}")
    print("=" * 60)

    all_data = {
        "metadata": {
            "tiers": args.tiers,
            "samples": args.samples,
            "context_tokens": args.context_tokens,
            "max_tokens": args.max_tokens,
        },
        "results": {},
    }

    for tier in args.tiers:
        print(f"\n[*] Running tier: {tier} concurrent requests (ctx={args.context_tokens:,})...")
        t0 = time.time()
        results = await run_tier(
            args.base_url, tier, args.context_tokens, args.max_tokens, args.samples
        )
        elapsed = time.time() - t0
        all_data["results"][str(tier)] = results
        valid = [r for r in results if "error" not in r]
        print(f"    Completed: {len(valid)}/{len(results)} successful in {elapsed:.1f}s")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_data, f, indent=2)

    print(f"\n[+] Results written to {args.output}")
    print("[*] Done.")


if __name__ == "__main__":
    asyncio.run(main())
