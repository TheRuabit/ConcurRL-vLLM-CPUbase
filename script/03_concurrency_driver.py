#!/usr/bin/env python3
"""
High-Concurrency Async Profiler Client
=======================================
Issues burst requests concurrently via aiohttp while capturing 5 latency
metrics per request. Sequentially steps through concurrency tiers
[32, 128, 256, 512, 1024], dumping raw performance arrays to JSON.

Metric Hook Mapping (aligned with reference/04_server_decomposition.py):
  t_serialize        — client JSON serialization time
  t_first_byte       — POST → first SSE byte  (HTTP overhead on localhost)
  t_server_prefill   — POST → first content token (TTFT from client perspective)
  t_prefill          — TTFT − first_byte       (pure GPU attention over input)
  t_decode           — first → last content token interval
  t_response_parse   — first byte → end of SSE stream

Default: input 32k tokens → output 64 tokens

Prerequisites:
  vLLM server running (use 02_launch_vllm.py or manual launch)

Usage:
    python script/03_concurrency_driver.py --url http://localhost:8000
    python script/03_concurrency_driver.py --scenarios 32 128 256
    python script/03_concurrency_driver.py --num-batches 5 --warmup-batches 2

Output:
    result/03_concurrency_driver.json
"""

import argparse
import asyncio
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Parse CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="High-concurrency async profiler — sweeps 32 to 1024"
)
parser.add_argument("--url", default="http://localhost:8000",
                    help="vLLM server URL")
parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B",
                    help="Model name for API requests")
parser.add_argument("--output", default=None,
                    help="Output JSON path")
parser.add_argument("--input-tokens", type=int, default=32000,
                    help="Simulated input token count")
parser.add_argument("--output-tokens", type=int, default=64,
                    help="Max output tokens per request")
parser.add_argument("--scenarios", nargs="+", type=int,
                    default=[32, 128, 256, 512, 1024],
                    help="Concurrency levels to test")
parser.add_argument("--num-batches", type=int, default=3,
                    help="Number of measurement batches per concurrency level")
parser.add_argument("--warmup-batches", type=int, default=1,
                    help="Warmup batches before measurement")
parser.add_argument("--request-timeout", type=int, default=600,
                    help="Per-request timeout in seconds")
args = parser.parse_args()

if args.output:
    out_path = Path(args.output)
else:
    out_path = Path(__file__).resolve().parents[1] / "result" / "03_concurrency_driver.json"
out_path.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Imports (after CLI so --help is fast)
# ---------------------------------------------------------------------------
import aiohttp

# ---------------------------------------------------------------------------
# Context text generator (matching reference pattern)
# ---------------------------------------------------------------------------
_CONTEXT_CACHE: dict[int, str] = {}

def get_context_text(target_tokens: int) -> str:
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


def build_payload(context_text: str, model: str, max_tokens: int) -> dict:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": context_text},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }


# ---------------------------------------------------------------------------
# Statistics helper
# ---------------------------------------------------------------------------
def compute_stats(values: list[float], ndigits: int = 20) -> dict:
    if not values:
        return {"mean": 0, "p50": 0, "p95": 0, "p99": 0, "min": 0, "max": 0}
    n = len(values)
    s = sorted(values)
    return {
        "mean": round(sum(values) / n, ndigits),
        "p50": round(s[n // 2], ndigits),
        "p95": round(s[int(n * 0.95)], ndigits),
        "p99": round(s[min(int(n * 0.99), n - 1)], ndigits),
        "min": round(s[0], ndigits),
        "max": round(s[-1], ndigits),
    }


# ---------------------------------------------------------------------------
# Single request profiler
# ---------------------------------------------------------------------------
@dataclass
class RequestTrace:
    concurrency_level: int = 0
    batch: int = 0
    idx: int = 0
    input_tokens: int = 0
    t_serialize_ms: float = 0.0
    t_first_byte_ms: float = 0.0
    t_server_prefill_ms: float = 0.0
    t_prefill_ms: float = 0.0
    t_decode_ms: float = 0.0
    t_response_parse_ms: float = 0.0
    t_e2e_ms: float = 0.0
    num_output_tokens: int = 0
    success: bool = True
    error: str = ""


async def trace_one_request(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict,
    concurrency_level: int,
    batch: int,
    idx: int,
    input_tokens: int,
    semaphore: asyncio.Semaphore,
    timeout: int,
) -> RequestTrace:
    t = RequestTrace(
        concurrency_level=concurrency_level,
        batch=batch,
        idx=idx,
        input_tokens=input_tokens,
    )
    e2e_start = time.perf_counter()

    t0 = time.perf_counter()
    body = json.dumps(payload, ensure_ascii=False)
    t.t_serialize_ms = (time.perf_counter() - t0) * 1000

    t_post = time.perf_counter()

    async with semaphore:
        try:
            t_first_byte = None

            async with session.post(
                f"{url}/v1/chat/completions",
                data=body,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                t_first_byte = time.perf_counter()
                t.t_first_byte_ms = (t_first_byte - t_post) * 1000

                if resp.status != 200:
                    err = await resp.text()
                    t.success = False
                    t.error = f"HTTP {resp.status}: {err[:300]}"
                    t.t_e2e_ms = (time.perf_counter() - e2e_start) * 1000
                    return t

                first_byte = False
                first_token = False
                t_first_token = None
                t_last_token = None
                parse_start = None
                token_count = 0

                async for line in resp.content:
                    if not first_byte:
                        first_byte = True
                        parse_start = time.perf_counter()

                    line_str = line.decode("utf-8").strip()
                    if line_str.startswith("data: ") and line_str != "data: [DONE]":
                        try:
                            chunk = json.loads(line_str[6:])
                            choices = chunk.get("choices", [])
                            if choices:
                                content = choices[0].get("delta", {}).get("content", "")
                                if content:
                                    if not first_token:
                                        t_first_token = time.perf_counter()
                                        t.t_server_prefill_ms = (t_first_token - t_post) * 1000
                                        first_token = True
                                    t_last_token = time.perf_counter()
                                    token_count += 1
                        except json.JSONDecodeError:
                            pass

                if first_token and t_first_byte:
                    t.t_prefill_ms = max(0, t.t_server_prefill_ms - t.t_first_byte_ms)
                if parse_start:
                    t.t_response_parse_ms = (time.perf_counter() - parse_start) * 1000
                if first_token and t_last_token:
                    t.t_decode_ms = (t_last_token - t_first_token) * 1000
                t.num_output_tokens = token_count

        except asyncio.TimeoutError:
            t.success = False
            t.error = "Timeout"
        except Exception as e:
            t.success = False
            t.error = str(e)[:300]

    t.t_e2e_ms = (time.perf_counter() - e2e_start) * 1000
    return t


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------
async def run_batch(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    concurrency_level: int,
    batch: int,
    input_tokens: int,
    output_tokens: int,
    timeout: int,
) -> list[RequestTrace]:
    text = get_context_text(input_tokens)
    sem = asyncio.Semaphore(concurrency_level)
    tasks = [
        trace_one_request(
            session, url,
            build_payload(text, model, output_tokens),
            concurrency_level, batch, i, input_tokens, sem, timeout,
        )
        for i in range(concurrency_level)
    ]
    print(f"    Batch {batch}: {concurrency_level} req, "
          f"input={input_tokens:,}...", end=" ", flush=True)
    t0 = time.perf_counter()
    results = await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - t0
    ok = sum(1 for r in results if r.success)
    print(f"{elapsed:.1f}s, {ok}/{len(results)} OK")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main():
    url = args.url.rstrip("/")
    print(f"[03_driver] Concurrency sweep — async profiler")
    print(f"[03_driver] Server:       {url}")
    print(f"[03_driver] Model:        {args.model}")
    print(f"[03_driver] Input tokens: {args.input_tokens:,}")
    print(f"[03_driver] Output tokens: {args.output_tokens}")
    print(f"[03_driver] Scenarios:    {args.scenarios}")
    print(f"[03_driver] Batches:      {args.num_batches} (+{args.warmup_batches} warmup)")
    print()

    all_traces: list[RequestTrace] = []
    scenario_summaries: dict[str, dict] = {}

    async with aiohttp.ClientSession() as session:
        for concurrency in args.scenarios:
            print(f"{'─'*60}")
            print(f"  Concurrency Level: {concurrency}")
            print(f"{'─'*60}")

            # Warmup
            for w in range(args.warmup_batches):
                await run_batch(
                    session, url, args.model,
                    concurrency, -1,  # warmup batch marker
                    args.input_tokens, args.output_tokens,
                    args.request_timeout,
                )

            # Measurement
            scenario_traces: list[RequestTrace] = []
            for b in range(args.num_batches):
                batch_results = await run_batch(
                    session, url, args.model,
                    concurrency, b,
                    args.input_tokens, args.output_tokens,
                    args.request_timeout,
                )
                scenario_traces.extend(batch_results)

            all_traces.extend(scenario_traces)

            # Per-scenario summary
            successful = [r for r in scenario_traces if r.success]
            if successful:
                summary = {
                    "concurrency": concurrency,
                    "num_requests": len(scenario_traces),
                    "num_successful": len(successful),
                    "t_serialize_ms": compute_stats([r.t_serialize_ms for r in successful]),
                    "t_first_byte_ms": compute_stats([r.t_first_byte_ms for r in successful]),
                    "t_server_prefill_ms": compute_stats([r.t_server_prefill_ms for r in successful]),
                    "t_prefill_ms": compute_stats([r.t_prefill_ms for r in successful]),
                    "t_decode_ms": compute_stats([r.t_decode_ms for r in successful]),
                    "t_response_parse_ms": compute_stats([r.t_response_parse_ms for r in successful]),
                    "t_e2e_ms": compute_stats([r.t_e2e_ms for r in successful]),
                }
            else:
                summary = {
                    "concurrency": concurrency,
                    "num_requests": len(scenario_traces),
                    "num_successful": 0,
                    "error": "All requests failed",
                }
            scenario_summaries[str(concurrency)] = summary

            print(f"  → P95 E2E: {summary.get('t_e2e_ms', {}).get('p95', 'N/A')}ms\n")

    # -------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------
    raw_traces = []
    for t in all_traces:
        raw_traces.append({
            "concurrency_level": t.concurrency_level,
            "batch": t.batch,
            "idx": t.idx,
            "input_tokens": t.input_tokens,
            "t_serialize_ms": round(t.t_serialize_ms, 20),
            "t_first_byte_ms": round(t.t_first_byte_ms, 20),
            "t_server_prefill_ms": round(t.t_server_prefill_ms, 20),
            "t_prefill_ms": round(t.t_prefill_ms, 20),
            "t_decode_ms": round(t.t_decode_ms, 20),
            "t_response_parse_ms": round(t.t_response_parse_ms, 20),
            "t_e2e_ms": round(t.t_e2e_ms, 20),
            "num_output_tokens": t.num_output_tokens,
            "success": t.success,
            "error": t.error,
        })

    output = {
        "benchmark": "concurrency_driver_sweep",
        "server_url": url,
        "model": args.model,
        "input_tokens": args.input_tokens,
        "output_tokens": args.output_tokens,
        "num_batches": args.num_batches,
        "warmup_batches": args.warmup_batches,
        "scenarios": args.scenarios,
        "summaries": scenario_summaries,
        "raw_traces": raw_traces,
    }

    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\n[03_driver] Results saved to {out_path}")

    # -------------------------------------------------------------------
    # Summary table
    # -------------------------------------------------------------------
    print("\n" + "=" * 130)
    print("CONCURRENCY SWEEP — Latency Breakdown (P95, ms)")
    print("=" * 130)
    header = (f"{'Conc':>6s} {'Requests':>8s} {'OK':>6s} "
              f"{'Serialize':>10s} {'1stByte':>10s} {'TTFT':>10s} "
              f"{'Prefill':>10s} {'Decode':>10s} {'Parse':>10s} {'E2E':>10s}")
    print(header)
    print("-" * 130)

    for concurrency in args.scenarios:
        s = scenario_summaries.get(str(concurrency), {})
        if "error" in s:
            print(f"{concurrency:>6d} {s['num_requests']:>8d} {'FAIL':>6s}")
        else:
            n = s["num_successful"]
            print(f"{concurrency:>6d} {s['num_requests']:>8d} {n:>6d} "
                  f"{s['t_serialize_ms']['p95']:>8.2f}ms "
                  f"{s['t_first_byte_ms']['p95']:>8.2f}ms "
                  f"{s['t_server_prefill_ms']['p95']:>8.2f}ms "
                  f"{s['t_prefill_ms']['p95']:>8.2f}ms "
                  f"{s['t_decode_ms']['p95']:>8.2f}ms "
                  f"{s['t_response_parse_ms']['p95']:>8.2f}ms "
                  f"{s['t_e2e_ms']['p95']:>8.2f}ms")
    print("=" * 130)


if __name__ == "__main__":
    asyncio.run(main())
