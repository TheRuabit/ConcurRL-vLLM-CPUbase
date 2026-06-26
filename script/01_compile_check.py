#!/usr/bin/env python3
"""
System Compilation Check — Mock Server & Async Client
======================================================
Runs a lightweight FastAPI mock server in a background thread alongside an
async aiohttp client. Simulates realistic prefill/decode streaming intervals
to verify the 5-metric timing hooks work correctly BEFORE downloading any
large model weights.

Metrics validated:
  t_serialize        — client JSON serialization
  t_http_overhead    — POST → first SSE byte (server scheduling simulation)
  t_server_prefill   — POST → first content token (TTFT)
  t_decode           — first → last content token
  t_response_parse   — SSE chunk parsing after last token

Usage:
    python script/01_compile_check.py
    python script/01_compile_check.py --concurrency 8 --input-tokens 500

Output:
    result/01_compile_check.json
"""

import argparse
import asyncio
import json
import os
import random
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Parse CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Compile check — mock server timing validation")
parser.add_argument("--host", default="127.0.0.1", help="Mock server host")
parser.add_argument("--port", type=int, default=8000, help="Mock server port")
parser.add_argument("--concurrency", type=int, default=4,
                    help="Number of concurrent requests per test")
parser.add_argument("--input-tokens", type=int, default=100,
                    help="Simulated input token count (small for fast check)")
parser.add_argument("--output-tokens", type=int, default=16,
                    help="Simulated output token count")
parser.add_argument("--output", default=None, help="Output JSON path")
args = parser.parse_args()

if args.output:
    out_path = Path(args.output)
else:
    out_path = Path(__file__).resolve().parents[1] / "result" / "01_compile_check.json"
out_path.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Imports (after CLI so --help is fast)
# ---------------------------------------------------------------------------
import aiohttp

# ---------------------------------------------------------------------------
# Mock vLLM Streaming Server (background thread)
# ---------------------------------------------------------------------------
MOCK_PREFILL_MS_PER_TOKEN = 0.05   # 0.05ms per input token (fast mock)
MOCK_DECODE_MS_PER_TOKEN = 2.0     # 2ms per output token (fast mock)

def run_mock_server(host: str, port: int):
    """Launch a minimal FastAPI mock server in the current thread."""
    try:
        from fastapi import FastAPI, Request
        from fastapi.responses import StreamingResponse, JSONResponse
        import uvicorn
    except ImportError:
        print("[mock] ERROR: fastapi/uvicorn not installed. Run: pip install fastapi uvicorn")
        sys.exit(1)

    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        messages = body.get("messages", [])
        max_tokens = body.get("max_tokens", 16)
        stream = body.get("stream", False)

        total_input = sum(len(m.get("content", "")) // 4 for m in messages)
        input_tokens = max(total_input, 10)
        output_tokens = min(max_tokens, 32)

        prefill_delay = input_tokens * MOCK_PREFILL_MS_PER_TOKEN / 1000.0
        decode_delay = MOCK_DECODE_MS_PER_TOKEN / 1000.0

        await asyncio.sleep(prefill_delay)

        if stream:
            async def generate():
                for i in range(output_tokens):
                    chunk = {
                        "id": f"mock-{int(time.time()*1000)}",
                        "object": "chat.completion.chunk",
                        "choices": [{
                            "index": 0,
                            "delta": {"content": f"token_{i} "},
                            "finish_reason": None,
                        }],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                    if i < output_tokens - 1:
                        await asyncio.sleep(decode_delay)
                yield "data: [DONE]\n\n"
            return StreamingResponse(generate(), media_type="text/event-stream")
        else:
            await asyncio.sleep(decode_delay * output_tokens)
            return JSONResponse({
                "id": "mock-0",
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "mock " * output_tokens},
                    "finish_reason": "stop",
                }],
            })

    uvicorn.run(app, host=host, port=port, log_level="warning")


# ---------------------------------------------------------------------------
# Timing dataclass
# ---------------------------------------------------------------------------
@dataclass
class TimingResult:
    idx: int
    t_serialize_ms: float = 0.0
    t_http_overhead_ms: float = 0.0
    t_server_prefill_ms: float = 0.0   # TTFT
    t_decode_ms: float = 0.0
    t_response_parse_ms: float = 0.0
    t_e2e_ms: float = 0.0
    num_output_tokens: int = 0
    success: bool = True
    error: str = ""


# ---------------------------------------------------------------------------
# Stats helper (matching reference format)
# ---------------------------------------------------------------------------
def compute_stats(values: list[float], ndigits: int = 4) -> dict:
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
# Payload builder
# ---------------------------------------------------------------------------
def build_payload(input_tokens: int, output_tokens: int) -> dict:
    context = "test " * (input_tokens // 2)
    return {
        "model": "mock-model",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": context},
        ],
        "max_tokens": output_tokens,
        "temperature": 0.0,
        "stream": True,
    }


# ---------------------------------------------------------------------------
# Single async request profiler
# ---------------------------------------------------------------------------
async def profile_one_request(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict,
    idx: int,
    semaphore: asyncio.Semaphore,
) -> TimingResult:
    t = TimingResult(idx=idx)
    e2e_start = time.perf_counter()

    t0 = time.perf_counter()
    body = json.dumps(payload, ensure_ascii=False)
    t.t_serialize_ms = (time.perf_counter() - t0) * 1000

    t_post = time.perf_counter()

    async with semaphore:
        try:
            async with session.post(
                f"{url}/v1/chat/completions",
                data=body,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                t_first_byte = time.perf_counter()
                t.t_http_overhead_ms = (t_first_byte - t_post) * 1000

                if resp.status != 200:
                    err = await resp.text()
                    t.success = False
                    t.error = f"HTTP {resp.status}: {err[:300]}"
                    t.t_e2e_ms = (time.perf_counter() - e2e_start) * 1000
                    return t

                first_token = False
                t_first_token = None
                t_last_token = None
                parse_start = None
                token_count = 0

                async for line in resp.content:
                    if parse_start is None:
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
# Main
# ---------------------------------------------------------------------------
async def main():
    url = f"http://{args.host}:{args.port}"
    print(f"[01_check] Compile check — mock server timing validation")
    print(f"[01_check] Server: {url}")
    print(f"[01_check] Concurrency: {args.concurrency}")
    print(f"[01_check] Simulated input: {args.input_tokens} tokens, output: {args.output_tokens} tokens\n")

    # Start mock server in background thread
    server_thread = threading.Thread(
        target=run_mock_server,
        args=(args.host, args.port),
        daemon=True,
    )
    server_thread.start()
    print(f"[01_check] Mock server starting on {url}...")
    await asyncio.sleep(2.0)

    # Health check
    async with aiohttp.ClientSession() as session:
        for attempt in range(10):
            try:
                async with session.get(f"{url}/health",
                                       timeout=aiohttp.ClientTimeout(total=2)) as resp:
                    if resp.status == 200:
                        print(f"[01_check] Mock server ready (attempt {attempt+1})")
                        break
            except Exception:
                pass
            await asyncio.sleep(0.5)
        else:
            print("[01_check] ERROR: Mock server failed to start")
            return

        # Run 3 test batches
        all_results: list[TimingResult] = []
        payload = build_payload(args.input_tokens, args.output_tokens)
        sem = asyncio.Semaphore(args.concurrency)

        for batch in range(3):
            print(f"  Batch {batch}: {args.concurrency} concurrent requests...", end=" ", flush=True)
            t0 = time.perf_counter()
            tasks = [
                profile_one_request(session, url, payload, i, sem)
                for i in range(args.concurrency)
            ]
            results = await asyncio.gather(*tasks)
            elapsed = time.perf_counter() - t0
            ok = sum(1 for r in results if r.success)
            print(f"{elapsed:.2f}s, {ok}/{len(results)} OK")
            all_results.extend(results)

    # Verify results
    successful = [r for r in all_results if r.success]
    failed = [r for r in all_results if not r.success]
    print(f"\n[01_check] Total: {len(all_results)} requests, "
          f"{len(successful)} OK, {len(failed)} failed")

    if not successful:
        print("[01_check] FAIL: No successful requests")
        return

    # Compute stats for each metric
    metrics = {
        "t_serialize_ms": compute_stats([r.t_serialize_ms for r in successful]),
        "t_http_overhead_ms": compute_stats([r.t_http_overhead_ms for r in successful]),
        "t_server_prefill_ms": compute_stats([r.t_server_prefill_ms for r in successful]),
        "t_decode_ms": compute_stats([r.t_decode_ms for r in successful]),
        "t_response_parse_ms": compute_stats([r.t_response_parse_ms for r in successful]),
        "t_e2e_ms": compute_stats([r.t_e2e_ms for r in successful]),
    }

    # Validation checks
    issues = []
    m = metrics
    if m["t_serialize_ms"]["mean"] < 0:
        issues.append("Negative serialize time")
    if m["t_http_overhead_ms"]["mean"] < 0:
        issues.append("Negative HTTP overhead")
    if m["t_server_prefill_ms"]["mean"] < 0:
        issues.append("Negative prefill time")
    if m["t_decode_ms"]["mean"] < 0:
        issues.append("Negative decode time")
    if m["t_response_parse_ms"]["mean"] < 0:
        issues.append("Negative parse time")
    if m["t_server_prefill_ms"]["mean"] > m["t_e2e_ms"]["mean"]:
        issues.append("Prefill > E2E (timing logic error)")

    status = "PASS" if not issues else "FAIL"
    if issues:
        print(f"\n[01_check] Validation issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print(f"\n[01_check] All 5 timing metrics validated successfully")

    # Save output
    raw_traces = []
    for r in all_results:
        raw_traces.append({
            "idx": r.idx,
            "t_serialize_ms": round(r.t_serialize_ms, 4),
            "t_http_overhead_ms": round(r.t_http_overhead_ms, 4),
            "t_server_prefill_ms": round(r.t_server_prefill_ms, 4),
            "t_decode_ms": round(r.t_decode_ms, 4),
            "t_response_parse_ms": round(r.t_response_parse_ms, 4),
            "t_e2e_ms": round(r.t_e2e_ms, 4),
            "num_output_tokens": r.num_output_tokens,
            "success": r.success,
            "error": r.error,
        })

    output = {
        "benchmark": "compile_check_mock_server",
        "server_url": url,
        "concurrency": args.concurrency,
        "input_tokens": args.input_tokens,
        "output_tokens": args.output_tokens,
        "status": status,
        "issues": issues,
        "metrics": metrics,
        "raw_traces": raw_traces,
    }

    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"[01_check] Results saved to {out_path}")

    # Print summary table
    print("\n" + "=" * 85)
    print("COMPILE CHECK — Timing Metrics (ms)")
    print("=" * 85)
    print(f"{'Metric':<25s} {'Mean':>10s} {'P50':>10s} {'P95':>10s} {'P99':>10s} {'Min':>10s} {'Max':>10s}")
    print("-" * 85)
    for name, stats in metrics.items():
        print(f"{name:<25s} {stats['mean']:>10.4f} {stats['p50']:>10.4f} "
              f"{stats['p95']:>10.4f} {stats['p99']:>10.4f} "
              f"{stats['min']:>10.4f} {stats['max']:>10.4f}")
    print("=" * 85)
    print(f"Status: {status}")


if __name__ == "__main__":
    asyncio.run(main())
