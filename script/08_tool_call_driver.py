#!/usr/bin/env python3
"""
Phase 2e: Multi-Turn Tool-Calling Driver
=========================================
Simulates agentic RL workloads where rollouts interleave generation with
external tool invocations. Each rollout runs N turns, where each turn
appends a synthetic tool result to the conversation history and re-sends
the full context to vLLM.

This stresses:
  - Prefix caching (common system prefix should hit cache)
  - KV cache capacity (growing histories across concurrent rollouts)
  - Scheduler prefill/decode interleaving (mixed request phases)

Reuses from Phase 1/2d:
  - aiohttp TraceConfig for 15 client-side timing metrics
  - VllmMetricsScraper for vLLM Prometheus histograms
  - MetricsPoller for running/waiting/kv_cache gauge time-series

Usage:
    python script/08_tool_call_driver.py --url http://localhost:8000
    python script/08_tool_call_driver.py --concurrency 32 --num-turns 4
    python script/08_tool_call_driver.py --num-turns 8 --tool-latency-ms 50

Output:
    result/08_tool_call_driver.json
"""

import argparse
import asyncio
import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Phase 2e: multi-turn tool-calling benchmark driver"
)
parser.add_argument("--url", default="http://localhost:8000",
                    help="vLLM server URL")
parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B",
                    help="Model name for API requests")
parser.add_argument("--concurrency", type=int, default=16,
                    help="Number of parallel rollouts")
parser.add_argument("--num-turns", type=int, default=4,
                    help="Tool-calling turns per rollout")
parser.add_argument("--input-tokens", type=int, default=2000,
                    help="Approximate initial prompt size in tokens")
parser.add_argument("--max-output-tokens", type=int, default=512,
                    help="Max tokens per turn response (used when --output-length-dist is not set)")
parser.add_argument("--output-length-dist", default=None,
                    help=("Variable-length output distribution. Format: "
                          "'name1:tokens1:weight1,name2:tokens2:weight2,...' "
                          "Example: 'short:64:0.3,medium:512:0.5,long:2048:0.2'"))
parser.add_argument("--tool-latency-ms", type=int, default=100,
                    help="Simulated tool execution latency (ms)")
parser.add_argument("--tool-latency-jitter-ms", type=int, default=0,
                    help=("Random jitter added to tool latency (uniform ±jitter). "
                          "Set >0 to simulate async real-tool behavior. "
                          "Example: --tool-latency-ms 500 --tool-latency-jitter-ms 400 → [100,900]ms"))
parser.add_argument("--tool-response-tokens", type=int, default=500,
                    help="Simulated tool result size in tokens")
parser.add_argument("--num-batches", type=int, default=3,
                    help="Measurement batches")
parser.add_argument("--warmup-batches", type=int, default=1,
                    help="Warmup batches before measurement")
parser.add_argument("--temperature", type=float, default=0.7,
                    help="Sampling temperature")
parser.add_argument("--request-timeout", type=int, default=600,
                    help="Per-request timeout in seconds")
parser.add_argument("--output", default=None,
                    help="Output JSON path")
args = parser.parse_args()

PROJECT_DIR = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Output length distribution parsing
# ---------------------------------------------------------------------------
def parse_output_length_dist(spec: str) -> list[dict]:
    """Parse 'name1:tokens1:weight1,name2:tokens2:weight2,...' into list of
    dicts with keys 'name', 'max_tokens', 'weight', and normalized 'cdf'."""
    entries = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        fields = part.split(":")
        if len(fields) != 3:
            raise ValueError(f"Invalid output-length-dist entry: '{part}' (expected name:tokens:weight)")
        name = fields[0].strip()
        max_tokens = int(fields[1].strip())
        weight = float(fields[2].strip())
        entries.append({"name": name, "max_tokens": max_tokens, "weight": weight})
    if not entries:
        raise ValueError("Empty output-length-dist")
    total_weight = sum(e["weight"] for e in entries)
    cdf = 0.0
    for e in entries:
        cdf += e["weight"] / total_weight
        e["cdf"] = cdf
    entries[-1]["cdf"] = 1.0  # ensure exact 1.0
    return entries


def sample_output_length(dist: list[dict], rng) -> dict:
    """Sample one entry from the distribution using the given random generator."""
    r = rng.random()
    for entry in dist:
        if r < entry["cdf"]:
            return entry
    return dist[-1]


OUTPUT_LENGTH_DIST = None
if args.output_length_dist:
    OUTPUT_LENGTH_DIST = parse_output_length_dist(args.output_length_dist)
    print(f"[config] Output length distribution:")
    for e in OUTPUT_LENGTH_DIST:
        print(f"  {e['name']}: {e['max_tokens']} tokens, weight={e['weight']:.2f}")

# ---------------------------------------------------------------------------
# Reused from 03/05d: aiohttp timing hooks
# ---------------------------------------------------------------------------
def _get_http_trace(ctx):
    trace_request_ctx = getattr(ctx, "trace_request_ctx", None)
    if isinstance(trace_request_ctx, dict):
        return trace_request_ctx.get("http_trace")
    return None


def build_http_trace_config() -> "aiohttp.TraceConfig":
    import aiohttp
    trace_config = aiohttp.TraceConfig()

    async def on_connection_queued_start(session, ctx, params):
        trace = _get_http_trace(ctx)
        if trace is not None:
            trace["queued_start"] = time.perf_counter()

    async def on_connection_queued_end(session, ctx, params):
        trace = _get_http_trace(ctx)
        if trace is not None:
            trace["queued_end"] = time.perf_counter()

    async def on_dns_resolvehost_start(session, ctx, params):
        trace = _get_http_trace(ctx)
        if trace is not None:
            trace["dns_start"] = time.perf_counter()

    async def on_dns_resolvehost_end(session, ctx, params):
        trace = _get_http_trace(ctx)
        if trace is not None:
            trace["dns_end"] = time.perf_counter()

    async def on_connection_create_start(session, ctx, params):
        trace = _get_http_trace(ctx)
        if trace is not None:
            trace["connect_start"] = time.perf_counter()

    async def on_connection_create_end(session, ctx, params):
        trace = _get_http_trace(ctx)
        if trace is not None:
            trace["connect_end"] = time.perf_counter()

    async def on_request_headers_sent(session, ctx, params):
        trace = _get_http_trace(ctx)
        if trace is not None:
            trace["headers_sent"] = time.perf_counter()

    async def on_request_chunk_sent(session, ctx, params):
        trace = _get_http_trace(ctx)
        if trace is not None:
            now = time.perf_counter()
            trace.setdefault("first_chunk_sent", now)
            trace["last_chunk_sent"] = now

    trace_config.on_connection_queued_start.append(on_connection_queued_start)
    trace_config.on_connection_queued_end.append(on_connection_queued_end)
    trace_config.on_dns_resolvehost_start.append(on_dns_resolvehost_start)
    trace_config.on_dns_resolvehost_end.append(on_dns_resolvehost_end)
    trace_config.on_connection_create_start.append(on_connection_create_start)
    trace_config.on_connection_create_end.append(on_connection_create_end)
    trace_config.on_request_headers_sent.append(on_request_headers_sent)
    trace_config.on_request_chunk_sent.append(on_request_chunk_sent)
    return trace_config


# ---------------------------------------------------------------------------
# Per-turn trace dataclass
# ---------------------------------------------------------------------------
@dataclass
class TurnTrace:
    rollout_idx: int = 0
    turn_idx: int = 0
    cumulative_input_tokens: int = 0
    # 15 timing metrics
    t_serialize_ms: float = 0.0
    t_sem_wait_ms: float = 0.0
    t_http_connect_ms: float = 0.0
    t_http_conn_queued_ms: float = 0.0
    t_http_dns_ms: float = 0.0
    t_http_tcp_connect_ms: float = 0.0
    t_http_request_send_ms: float = 0.0
    t_http_response_headers_wait_ms: float = 0.0
    t_response_header_to_first_sse_byte_ms: float = 0.0
    t_first_sse_byte_to_first_token_ms: float = 0.0
    t_first_byte_ms: float = 0.0
    t_server_prefill_ms: float = 0.0
    t_prefill_ms: float = 0.0
    t_decode_ms: float = 0.0
    t_response_parse_ms: float = 0.0
    t_e2e_ms: float = 0.0
    # Output
    num_output_tokens: int = 0
    output_text: str = ""
    vllm_request_id: str = ""
    # Status
    success: bool = True
    error: str = ""


# ---------------------------------------------------------------------------
# Reused from 03/05d: Single request profiler
# ---------------------------------------------------------------------------
async def trace_one_turn(
    session: "aiohttp.ClientSession",
    url: str,
    payload: dict,
    semaphore: asyncio.Semaphore,
    timeout: int,
    rollout_idx: int,
    turn_idx: int,
) -> TurnTrace:
    import aiohttp
    t = TurnTrace(rollout_idx=rollout_idx, turn_idx=turn_idx)
    e2e_start = time.perf_counter()

    t0 = time.perf_counter()
    body = json.dumps(payload, ensure_ascii=False)
    t.t_serialize_ms = (time.perf_counter() - t0) * 1000

    t_pre_sem = time.perf_counter()
    async with semaphore:
        t_post_sem = time.perf_counter()
        t.t_sem_wait_ms = (t_post_sem - t_pre_sem) * 1000

        t_post = time.perf_counter()
        http_trace = {"request_start": t_post}
        try:
            async with session.post(
                f"{url}/v1/chat/completions",
                data=body,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=timeout),
                trace_request_ctx={"http_trace": http_trace},
            ) as resp:
                t_first_byte = time.perf_counter()
                t.t_http_connect_ms = (t_first_byte - t_post) * 1000
                t.t_http_conn_queued_ms = max(
                    0.0,
                    (http_trace.get("queued_end", 0.0) -
                     http_trace.get("queued_start", 0.0)) * 1000,
                )
                t.t_http_dns_ms = max(
                    0.0,
                    (http_trace.get("dns_end", 0.0) -
                     http_trace.get("dns_start", 0.0)) * 1000,
                )
                t.t_http_tcp_connect_ms = max(
                    0.0,
                    (http_trace.get("connect_end", 0.0) -
                     http_trace.get("connect_start", 0.0)) * 1000,
                )
                t.t_http_request_send_ms = max(
                    0.0,
                    (http_trace.get("last_chunk_sent", 0.0) -
                     http_trace.get("headers_sent", 0.0)) * 1000,
                )
                wait_start = (
                    http_trace.get("last_chunk_sent")
                    or http_trace.get("headers_sent")
                    or t_post
                )
                t.t_http_response_headers_wait_ms = max(
                    0.0, (t_first_byte - wait_start) * 1000
                )
                t.t_first_byte_ms = t.t_sem_wait_ms + t.t_http_connect_ms

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
                t_first_sse_byte = None
                parse_start = None
                token_count = 0
                output_parts = []

                async for line in resp.content:
                    if not first_byte:
                        first_byte = True
                        t_first_sse_byte = time.perf_counter()
                        parse_start = t_first_sse_byte
                        t.t_response_header_to_first_sse_byte_ms = max(
                            0.0, (t_first_sse_byte - t_first_byte) * 1000
                        )

                    line_str = line.decode("utf-8").strip()
                    if line_str.startswith("data: ") and line_str != "data: [DONE]":
                        try:
                            chunk = json.loads(line_str[6:])
                            if not t.vllm_request_id and chunk.get("id"):
                                t.vllm_request_id = str(chunk["id"])
                            choices = chunk.get("choices", [])
                            if choices:
                                content = choices[0].get("delta", {}).get("content", "")
                                if content:
                                    if not first_token:
                                        t_first_token = time.perf_counter()
                                        t.t_server_prefill_ms = (t_first_token - t_post) * 1000
                                        if t_first_sse_byte is not None:
                                            t.t_first_sse_byte_to_first_token_ms = max(
                                                0.0,
                                                (t_first_token - t_first_sse_byte) * 1000,
                                            )
                                        first_token = True
                                    t_last_token = time.perf_counter()
                                    token_count += 1
                                    output_parts.append(content)
                        except json.JSONDecodeError:
                            pass

                if first_token and t_first_byte:
                    t.t_prefill_ms = max(0, t.t_server_prefill_ms - t.t_first_byte_ms)
                if parse_start:
                    t.t_response_parse_ms = (time.perf_counter() - parse_start) * 1000
                if first_token and t_last_token:
                    t.t_decode_ms = (t_last_token - t_first_token) * 1000
                t.num_output_tokens = token_count
                t.output_text = "".join(output_parts)

        except asyncio.TimeoutError:
            t.success = False
            t.error = "Timeout"
        except Exception as e:
            t.success = False
            t.error = str(e)[:300]

    t.t_e2e_ms = (time.perf_counter() - e2e_start) * 1000
    return t


# ---------------------------------------------------------------------------
# Reused from 03/05d: VllmMetricsScraper
# ---------------------------------------------------------------------------
HISTOGRAM_METRICS = [
    ("vllm_queue_time",        "vllm:request_queue_time_seconds"),
    ("vllm_prefill_time",      "vllm:request_prefill_time_seconds"),
    ("vllm_decode_time",       "vllm:request_decode_time_seconds"),
    ("vllm_ttft",              "vllm:time_to_first_token_seconds"),
    ("vllm_e2e",               "vllm:e2e_request_latency_seconds"),
    ("vllm_inference_time",    "vllm:request_inference_time_seconds"),
    ("vllm_itl",               "vllm:inter_token_latency_seconds"),
    ("vllm_time_per_out_tok",  "vllm:request_time_per_output_token_seconds"),
    ("vllm_kv_computed_tokens","vllm:request_prefill_kv_computed_tokens"),
    ("http_request_duration",  "http_request_duration_highr_seconds"),
]


def percentile_from_buckets(buckets, total_count, pct):
    if total_count <= 0 or not buckets:
        return 0.0
    target = pct / 100.0 * total_count
    prev_bound, prev_count = 0.0, 0
    for bound, count in buckets:
        if count >= target:
            if count == prev_count:
                return bound
            frac = (target - prev_count) / (count - prev_count)
            return prev_bound + frac * (bound - prev_bound)
        prev_bound, prev_count = bound, count
    return buckets[-1][0] if buckets else 0.0


class VllmMetricsScraper:
    def __init__(self, base_url: str):
        self.metrics_url = f"{base_url.rstrip('/')}/metrics"

    async def scrape(self, session) -> dict:
        import aiohttp
        try:
            async with session.get(
                self.metrics_url,
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                if resp.status != 200:
                    return {}
                text = await resp.text()
                return self._parse_prometheus(text)
        except Exception:
            return {}

    @staticmethod
    def _parse_prometheus(text: str) -> dict:
        result = {}
        gauge_patterns = {
            "vllm_requests_running": r'vllm:num_requests_running\{[^}]*\}\s+([\d.eE+\-]+)',
            "vllm_requests_waiting": r'vllm:num_requests_waiting\{[^}]*\}\s+([\d.eE+\-]+)',
            "vllm_kv_cache_usage":   r'vllm:kv_cache_usage_perc\{[^}]*\}\s+([\d.eE+\-]+)',
        }
        counter_patterns = {
            "vllm_num_preemptions_total": r'vllm:num_preemptions_total\{[^}]*\}\s+([\d.eE+\-]+)',
        }
        for key, pattern in gauge_patterns.items():
            m = re.search(pattern, text)
            if m:
                result[key] = float(m.group(1))
        for key, pattern in counter_patterns.items():
            m = re.search(pattern, text)
            if m:
                result[key] = float(m.group(1))

        for prefix, metric_name in HISTOGRAM_METRICS:
            escaped = re.escape(metric_name)
            bucket_re = re.compile(
                rf'^{escaped}_bucket\{{[^}}]*le="([^"]+)"\}}\s+([\d.eE+\-]+)$',
                re.MULTILINE,
            )
            sum_re = re.compile(rf'^{escaped}_sum\{{[^}}]*\}}\s+([\d.eE+\-]+)$', re.MULTILINE)
            count_re = re.compile(rf'^{escaped}_count\{{[^}}]*\}}\s+([\d.eE+\-]+)$', re.MULTILINE)

            buckets = []
            for m in bucket_re.finditer(text):
                le, val = m.group(1), float(m.group(2))
                if le != "+Inf":
                    buckets.append((float(le), int(val)))
            buckets.sort(key=lambda x: x[0])

            m_sum = sum_re.search(text)
            m_count = count_re.search(text)

            if m_sum:
                result[f"{prefix}_sum"] = float(m_sum.group(1))
            if m_count:
                result[f"{prefix}_count"] = float(m_count.group(1))
            if buckets:
                result[f"{prefix}_buckets"] = buckets

            total = result.get(f"{prefix}_count", 0)
            if buckets and total > 0:
                result[f"{prefix}_p50"] = percentile_from_buckets(buckets, total, 50)
                result[f"{prefix}_p95"] = percentile_from_buckets(buckets, total, 95)
                result[f"{prefix}_p99"] = percentile_from_buckets(buckets, total, 99)
            if f"{prefix}_sum" in result and total > 0:
                result[f"{prefix}_avg"] = result[f"{prefix}_sum"] / total

        return result


def _subtract_buckets(b_buckets, a_buckets):
    b_map = {bound: cnt for bound, cnt in b_buckets}
    a_map = {bound: cnt for bound, cnt in a_buckets}
    all_bounds = sorted(set(b_map.keys()) | set(a_map.keys()))
    result = []
    for bound in all_bounds:
        delta = a_map.get(bound, 0) - b_map.get(bound, 0)
        result.append((bound, max(0, delta)))
    cum = 0
    cumulative = []
    for bound, cnt in result:
        cum += cnt
        cumulative.append((bound, cum))
    return cumulative


def compute_metrics_delta(before: dict, after: dict) -> dict:
    delta = {}
    for prefix, _metric_name in HISTOGRAM_METRICS:
        sum_b = before.get(f"{prefix}_sum", 0)
        sum_a = after.get(f"{prefix}_sum", 0)
        cnt_b = before.get(f"{prefix}_count", 0)
        cnt_a = after.get(f"{prefix}_count", 0)
        d_sum = sum_a - sum_b
        d_cnt = cnt_a - cnt_b
        if d_cnt > 0:
            delta[f"{prefix}_avg"] = d_sum / d_cnt
            delta[f"{prefix}_count"] = d_cnt
        b_bkts = before.get(f"{prefix}_buckets", [])
        a_bkts = after.get(f"{prefix}_buckets", [])
        if b_bkts and a_bkts:
            dk = _subtract_buckets(b_bkts, a_bkts)
            total = dk[-1][1] if dk else 0
            if total > 0:
                delta[f"{prefix}_p50"] = percentile_from_buckets(dk, total, 50)
                delta[f"{prefix}_p95"] = percentile_from_buckets(dk, total, 95)
                delta[f"{prefix}_p99"] = percentile_from_buckets(dk, total, 99)
    for key in ["vllm_requests_running", "vllm_requests_waiting", "vllm_kv_cache_usage"]:
        if key in after:
            delta[key] = after[key]
    if "vllm_num_preemptions_total" in after and "vllm_num_preemptions_total" in before:
        delta["vllm_num_preemptions"] = after["vllm_num_preemptions_total"] - before["vllm_num_preemptions_total"]
    elif "vllm_num_preemptions_total" in after:
        delta["vllm_num_preemptions"] = after["vllm_num_preemptions_total"]
    return delta


# ---------------------------------------------------------------------------
# Reused from 03/05d: MetricsPoller
# ---------------------------------------------------------------------------
class MetricsPoller:
    def __init__(self, base_url: str, interval_ms: int = 20):
        self.metrics_url = f"{base_url.rstrip('/')}/metrics"
        self.interval = interval_ms / 1000.0
        self._task: Optional[asyncio.Task] = None
        self._samples: list[dict] = []
        self._t0: float = 0

    async def _poll_loop(self):
        opener = __import__('urllib.request', fromlist=['build_opener', 'ProxyHandler'])
        proxy_handler = opener.ProxyHandler({})
        http_opener = opener.build_opener(proxy_handler)
        while True:
            t = time.perf_counter() - self._t0
            try:
                req = opener.Request(self.metrics_url)
                resp = http_opener.open(req, timeout=2)
                text = resp.read().decode()
                sample = {"t": round(t * 1000, 1)}
                for key, pattern in [
                    ("running", r'vllm:num_requests_running\{[^}]*\}\s+([\d.eE+\-]+)'),
                    ("waiting", r'vllm:num_requests_waiting\{[^}]*\}\s+([\d.eE+\-]+)'),
                    ("kv_cache", r'vllm:kv_cache_usage_perc\{[^}]*\}\s+([\d.eE+\-]+)'),
                    ("preemptions", r'vllm:num_preemptions_total\{[^}]*\}\s+([\d.eE+\-]+)'),
                ]:
                    m = re.search(pattern, text)
                    sample[key] = float(m.group(1)) if m else 0.0
                self._samples.append(sample)
            except Exception:
                pass
            await asyncio.sleep(self.interval)

    def start(self):
        self._samples = []
        self._t0 = time.perf_counter()
        self._task = asyncio.ensure_future(self._poll_loop())

    async def stop(self) -> list[dict]:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        return self._samples


# ---------------------------------------------------------------------------
# Tool simulation
# ---------------------------------------------------------------------------
TOOL_SYSTEM_PROMPT = """\
You are a helpful assistant that solves problems step by step. \
You have access to two tools:

1. calculator — for arithmetic computations. Usage: <tool_call>{"name": "calculator", "expression": "2+2"}</tool_call>
2. code_executor — for running code snippets. Usage: <tool_call>{"name": "code_executor", "code": "print(42)"}</tool_call>

When you need to use a tool, output a <tool_call>...</tool_call> block. \
When you have enough information, give your final answer in \\boxed{}."""

TOOL_RESULT_TEMPLATES = {
    "calculator": "Result: {result}",
    "code_executor": "Output:\n{result}",
}

# Pre-built tool results (synthetic, no real execution)
_CALC_RESULTS = [
    "42", "128", "256", "1024", "3.14159", "2.71828", "100", "64",
    "0.5", "144", "1000", "7.389", "1.414", "16", "81", "3600",
]

_CODE_RESULTS = [
    "Computed successfully.\nResult: 42",
    "Execution complete.\nOutput: [1, 2, 3, 4, 5]",
    "Done. Sum = 15",
    "Iteration complete. Count = 100",
    "Processed 50 items successfully.",
    "Array sorted. First element: 0",
    "Matrix multiplication complete. Shape: (10, 10)",
    "Optimization converged after 25 iterations.",
]


def generate_tool_result(tool_name: str, turn_idx: int) -> str:
    """Generate a synthetic tool result."""
    template = TOOL_RESULT_TEMPLATES.get(tool_name, "Result: {result}")
    if tool_name == "calculator":
        result = _CALC_RESULTS[turn_idx % len(_CALC_RESULTS)]
    else:
        result = _CODE_RESULTS[turn_idx % len(_CODE_RESULTS)]
    return template.format(result=result)


def estimate_tokens(text: str) -> int:
    """Rough token count estimate (1 token ≈ 4 chars for English)."""
    return max(1, len(text) // 4)


def build_initial_messages(system_prompt: str, user_prompt: str) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_turn_payload(
    messages: list[dict],
    model: str,
    max_tokens: int,
    temperature: float,
) -> dict:
    return {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }


def parse_tool_call(text: str) -> Optional[tuple[str, str]]:
    """Extract tool name and raw input from <tool_call>...</tool_call> tags."""
    m = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(1))
        name = obj.get("name", "")
        if name == "calculator":
            return ("calculator", obj.get("expression", ""))
        elif name == "code_executor":
            return ("code_executor", obj.get("code", ""))
        return (name, json.dumps(obj))
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Multi-turn rollout
# ---------------------------------------------------------------------------
async def run_one_rollout(
    session: "aiohttp.ClientSession",
    url: str,
    model: str,
    semaphore: asyncio.Semaphore,
    rollout_idx: int,
    num_turns: int,
    max_output_tokens: int,
    temperature: float,
    tool_latency_ms: int,
    tool_response_tokens: int,
    request_timeout: int,
    scraper: VllmMetricsScraper,
) -> dict:
    """Run one multi-turn tool-calling rollout. Returns per-turn traces."""
    import aiohttp

    # Build initial prompt (synthetic math problem)
    user_prompt = (
        f"Solve the following problem step by step. Use the calculator and "
        f"code_executor tools as needed. Problem #{rollout_idx}: "
        f"Compute the sum of the first {100 + rollout_idx * 10} natural numbers "
        f"and verify using a different method."
    )
    messages = build_initial_messages(TOOL_SYSTEM_PROMPT, user_prompt)

    # Estimate initial tokens
    cumulative_tokens = estimate_tokens(TOOL_SYSTEM_PROMPT) + estimate_tokens(user_prompt)

    turn_traces = []
    tool_name_for_next = None

    for turn_idx in range(num_turns):
        payload = build_turn_payload(messages, model, max_output_tokens, temperature)

        # Scrape vLLM metrics before this turn
        metrics_before = await scraper.scrape(session)

        trace = await trace_one_turn(
            session=session,
            url=url,
            payload=payload,
            semaphore=semaphore,
            timeout=request_timeout,
            rollout_idx=rollout_idx,
            turn_idx=turn_idx,
        )

        # Scrape vLLM metrics after this turn
        metrics_after = await scraper.scrape(session)
        vllm_delta = compute_metrics_delta(metrics_before, metrics_after)

        trace.cumulative_input_tokens = cumulative_tokens
        turn_record = asdict(trace)
        turn_record["vllm_metrics_delta"] = vllm_delta

        turn_traces.append(turn_record)

        if not trace.success:
            break

        # Parse tool call from model output
        tool_call = parse_tool_call(trace.output_text)

        # Append assistant message
        messages.append({"role": "assistant", "content": trace.output_text})

        if turn_idx < num_turns - 1:
            # Simulate tool execution with optional jitter
            jitter = args.tool_latency_jitter_ms
            actual_latency = tool_latency_ms
            if jitter > 0:
                import random as _rand
                actual_latency = max(10, tool_latency_ms + _rand.randint(-jitter, jitter))

            if tool_call:
                tool_name, tool_input = tool_call
                await asyncio.sleep(actual_latency / 1000.0)
                tool_result = generate_tool_result(tool_name, turn_idx)
            else:
                tool_name = "calculator" if turn_idx % 2 == 0 else "code_executor"
                await asyncio.sleep(actual_latency / 1000.0)
                tool_result = generate_tool_result(tool_name, turn_idx)

            tool_msg = f"Tool '{tool_name}' result:\n{tool_result}"
            messages.append({"role": "user", "content": tool_msg})
            cumulative_tokens += estimate_tokens(trace.output_text) + estimate_tokens(tool_msg)

    # Compute rollout-level summary
    successful_turns = [t for t in turn_traces if t["success"]]
    rollout_e2e_ms = sum(t["t_e2e_ms"] for t in turn_traces)

    # Scheduled → first output: time from first turn start to first token
    # (includes sem_wait + HTTP connect + GPU prefill for turn 0)
    scheduled_to_first_output_ms = turn_traces[0]["t_server_prefill_ms"] if turn_traces else 0

    # First output → completion: decode time across all turns + tool latency + inter-turn overhead
    first_output_to_completion_ms = max(0, rollout_e2e_ms - scheduled_to_first_output_ms)

    return {
        "rollout_idx": rollout_idx,
        "num_turns": len(turn_traces),
        "num_successful_turns": len(successful_turns),
        "rollout_e2e_ms": round(rollout_e2e_ms, 2),
        "scheduled_to_first_output_ms": round(scheduled_to_first_output_ms, 2),
        "first_output_to_completion_ms": round(first_output_to_completion_ms, 2),
        "turn_traces": turn_traces,
    }


# ---------------------------------------------------------------------------
# Stats helper
# ---------------------------------------------------------------------------
def compute_stats(values: list[float], ndigits: int = 4) -> dict:
    if not values:
        return {"mean": 0, "p50": 0, "p95": 0, "p99": 0, "min": 0, "max": 0, "count": 0}
    n = len(values)
    s = sorted(values)
    return {
        "mean": round(sum(values) / n, ndigits),
        "p50": round(s[n // 2], ndigits),
        "p95": round(s[int(n * 0.95)], ndigits),
        "p99": round(s[min(int(n * 0.99), n - 1)], ndigits),
        "min": round(s[0], ndigits),
        "max": round(s[-1], ndigits),
        "count": n,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main():
    import aiohttp

    print("=" * 60)
    print(" Phase 2e: Multi-Turn Tool-Calling Driver")
    print("=" * 60)
    print(f"  URL:              {args.url}")
    print(f"  Model:            {args.model}")
    print(f"  Concurrency:      {args.concurrency}")
    print(f"  Num turns:        {args.num_turns}")
    print(f"  Input tokens:     ~{args.input_tokens}")
    print(f"  Max output tokens:{args.max_output_tokens}")
    print(f"  Tool latency:     {args.tool_latency_ms}ms")
    print(f"  Tool resp tokens: {args.tool_response_tokens}")
    print(f"  Batches:          {args.num_batches} (+{args.warmup_batches} warmup)")
    print("=" * 60)

    scraper = VllmMetricsScraper(args.url)
    poller = MetricsPoller(args.url, interval_ms=20)
    semaphore = asyncio.Semaphore(args.concurrency)

    connector = aiohttp.TCPConnector(limit=0, limit_per_host=0)
    trace_config = build_http_trace_config()

    output_path = Path(args.output) if args.output else PROJECT_DIR / "result" / "08_tool_call_driver.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_results = {
        "config": {
            "url": args.url,
            "model": args.model,
            "concurrency": args.concurrency,
            "num_turns": args.num_turns,
            "input_tokens": args.input_tokens,
            "max_output_tokens": args.max_output_tokens,
            "output_length_dist": args.output_length_dist,
            "tool_latency_ms": args.tool_latency_ms,
            "tool_response_tokens": args.tool_response_tokens,
            "num_batches": args.num_batches,
            "warmup_batches": args.warmup_batches,
            "temperature": args.temperature,
        },
        "batches": [],
        "summary": {},
    }

    async with aiohttp.ClientSession(
        connector=connector,
        trace_configs=[trace_config],
    ) as session:
        # Check server health
        try:
            async with session.get(
                f"{args.url}/health",
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    print(f"[ERROR] Server health check failed: HTTP {resp.status}")
                    return
            print("[OK] Server health check passed.")
        except Exception as e:
            print(f"[ERROR] Cannot reach server at {args.url}: {e}")
            return

        total_batches = args.warmup_batches + args.num_batches

        for batch_idx in range(total_batches):
            is_warmup = batch_idx < args.warmup_batches
            label = f"{'warmup' if is_warmup else 'batch'} {batch_idx}"
            print(f"\n--- {label} (concurrency={args.concurrency}, turns={args.num_turns}) ---")

            # Start metrics poller
            poller.start()
            metrics_before = await scraper.scrape(session)

            t_batch_start = time.perf_counter()

            # Assign output lengths from distribution (or fixed)
            rng = __import__("random").Random(batch_idx * 1000 + 42)
            if OUTPUT_LENGTH_DIST:
                rollout_assignments = [
                    sample_output_length(OUTPUT_LENGTH_DIST, rng)
                    for _ in range(args.concurrency)
                ]
            else:
                rollout_assignments = [
                    {"name": "fixed", "max_tokens": args.max_output_tokens, "weight": 1.0, "cdf": 1.0}
                ] * args.concurrency

            # Wrap each rollout to record wall-clock completion offset and length category
            async def _tracked_rollout(idx):
                assigned = rollout_assignments[idx]
                result = await run_one_rollout(
                    session=session,
                    url=args.url,
                    model=args.model,
                    semaphore=semaphore,
                    rollout_idx=idx,
                    num_turns=args.num_turns,
                    max_output_tokens=assigned["max_tokens"],
                    temperature=args.temperature,
                    tool_latency_ms=args.tool_latency_ms,
                    tool_response_tokens=args.tool_response_tokens,
                    request_timeout=args.request_timeout,
                    scraper=scraper,
                )
                result["t_completed_offset_ms"] = round(
                    (time.perf_counter() - t_batch_start) * 1000, 2
                )
                result["output_length_category"] = assigned["name"]
                result["max_output_tokens_assigned"] = assigned["max_tokens"]
                return result

            # Launch all rollouts concurrently
            tasks = [_tracked_rollout(i) for i in range(args.concurrency)]
            rollout_results = await asyncio.gather(*tasks, return_exceptions=True)

            t_batch_ms = (time.perf_counter() - t_batch_start) * 1000
            poll_samples = await poller.stop()
            metrics_after = await scraper.scrape(session)
            vllm_delta = compute_metrics_delta(metrics_before, metrics_after)

            # Process results
            successful = []
            failed = []
            for i, r in enumerate(rollout_results):
                if isinstance(r, Exception):
                    failed.append({"rollout_idx": i, "error": str(r)[:300]})
                elif isinstance(r, dict):
                    if r["num_successful_turns"] == r["num_turns"]:
                        successful.append(r)
                    else:
                        failed.append(r)
                else:
                    failed.append({"rollout_idx": i, "error": "unknown"})

            # Collect per-turn stats
            all_turn_traces = []
            for r in successful:
                all_turn_traces.extend(r["turn_traces"])

            per_turn_ttft = defaultdict(list)
            per_turn_e2e = defaultdict(list)
            per_turn_decode = defaultdict(list)
            per_turn_output_tokens = defaultdict(list)
            for t in all_turn_traces:
                if t["success"]:
                    per_turn_ttft[t["turn_idx"]].append(t["t_server_prefill_ms"])
                    per_turn_e2e[t["turn_idx"]].append(t["t_e2e_ms"])
                    per_turn_decode[t["turn_idx"]].append(t["t_decode_ms"])
                    per_turn_output_tokens[t["turn_idx"]].append(t["num_output_tokens"])

            rollout_e2e_values = [r["rollout_e2e_ms"] for r in successful]
            total_output_tokens = sum(
                t["num_output_tokens"] for t in all_turn_traces if t["success"]
            )

            # Group-ready metrics: time from batch start until each rollout completes
            completion_offsets = sorted(
                r["t_completed_offset_ms"] for r in successful
            )
            group_ready_stats = {}
            if completion_offsets:
                n = len(completion_offsets)
                group_ready_stats = {
                    "p50": round(completion_offsets[int(n * 0.50)], 2),
                    "p95": round(completion_offsets[min(int(n * 0.95), n - 1)], 2),
                    "p99": round(completion_offsets[min(int(n * 0.99), n - 1)], 2),
                    "first": round(completion_offsets[0], 2),
                    "last": round(completion_offsets[-1], 2),
                }

            # Scheduled → first output and first output → completion breakdown
            sched_to_first = [r["scheduled_to_first_output_ms"] for r in successful]
            first_to_done = [r["first_output_to_completion_ms"] for r in successful]

            # Per output-length-category stats
            per_category = {}
            categories = set(r.get("output_length_category", "fixed") for r in successful)
            for cat in sorted(categories):
                cat_rollouts = [r for r in successful if r.get("output_length_category") == cat]
                cat_e2e = [r["rollout_e2e_ms"] for r in cat_rollouts]
                cat_turns = []
                for r in cat_rollouts:
                    cat_turns.extend(r["turn_traces"])
                cat_ttft = [t["t_server_prefill_ms"] for t in cat_turns if t.get("success")]
                cat_decode = [t["t_decode_ms"] for t in cat_turns if t.get("success")]
                cat_output_tokens = [t["num_output_tokens"] for t in cat_turns if t.get("success")]
                per_category[cat] = {
                    "count": len(cat_rollouts),
                    "max_tokens_assigned": cat_rollouts[0].get("max_output_tokens_assigned", 0) if cat_rollouts else 0,
                    "rollout_e2e_stats": compute_stats(cat_e2e),
                    "ttft_stats": compute_stats(cat_ttft),
                    "decode_stats": compute_stats(cat_decode),
                    "output_tokens_stats": compute_stats(cat_output_tokens),
                }

            batch_summary = {
                "batch": batch_idx,
                "is_warmup": is_warmup,
                "concurrency": args.concurrency,
                "num_turns": args.num_turns,
                "t_batch_ms": round(t_batch_ms, 2),
                "num_successful": len(successful),
                "num_failed": len(failed),
                "rollout_e2e_stats": compute_stats(rollout_e2e_values),
                "group_completion": group_ready_stats,
                "scheduled_to_first_output_stats": compute_stats(sched_to_first),
                "first_output_to_completion_stats": compute_stats(first_to_done),
                "total_output_tokens": total_output_tokens,
                "output_token_throughput": round(
                    total_output_tokens / (t_batch_ms / 1000) if t_batch_ms > 0 else 0, 2
                ),
                "per_turn_ttft": {
                    str(k): compute_stats(v) for k, v in sorted(per_turn_ttft.items())
                },
                "per_turn_e2e": {
                    str(k): compute_stats(v) for k, v in sorted(per_turn_e2e.items())
                },
                "per_turn_decode": {
                    str(k): compute_stats(v) for k, v in sorted(per_turn_decode.items())
                },
                "per_turn_output_tokens": {
                    str(k): compute_stats(v) for k, v in sorted(per_turn_output_tokens.items())
                },
                "per_category": per_category,
                "vllm_metrics_delta": vllm_delta,
                "poll_samples_count": len(poll_samples),
            }

            if not is_warmup:
                all_results["batches"].append({
                    "summary": batch_summary,
                    "rollout_results": successful + [
                        r for r in failed if isinstance(r, dict)
                    ],
                    "poll_samples": poll_samples,
                })

            # Print summary
            print(f"  Rollouts: {len(successful)}/{args.concurrency} successful")
            print(f"  Batch E2E: {t_batch_ms:.0f}ms")
            if rollout_e2e_values:
                print(f"  Rollout E2E: P50={compute_stats(rollout_e2e_values)['p50']:.0f}ms "
                      f"P95={compute_stats(rollout_e2e_values)['p95']:.0f}ms")
            if group_ready_stats:
                print(f"  Group-ready: P50={group_ready_stats['p50']:.0f}ms "
                      f"P95={group_ready_stats['p95']:.0f}ms "
                      f"(first={group_ready_stats['first']:.0f}ms "
                      f"last={group_ready_stats['last']:.0f}ms)")
            if sched_to_first:
                stf = compute_stats(sched_to_first)
                ftc = compute_stats(first_to_done)
                print(f"  Scheduled→1st output: P50={stf['p50']:.0f}ms | "
                      f"1st output→completion: P50={ftc['p50']:.0f}ms")
            print(f"  Throughput: {batch_summary['output_token_throughput']:.0f} tok/s")
            for turn_idx in sorted(per_turn_ttft.keys()):
                ttft_stats = compute_stats(per_turn_ttft[turn_idx])
                print(f"  Turn {turn_idx} TTFT: P50={ttft_stats['p50']:.0f}ms "
                      f"P95={ttft_stats['p95']:.0f}ms")
            if len(categories) > 1:
                print(f"  Per-category:")
                for cat in sorted(per_category.keys()):
                    cs = per_category[cat]
                    print(f"    {cat}: n={cs['count']}, max_tok={cs['max_tokens_assigned']}, "
                          f"E2E P50={cs['rollout_e2e_stats']['p50']:.0f}ms, "
                          f"TTFT P50={cs['ttft_stats']['p50']:.0f}ms")

    # Compute aggregate summary across measurement batches
    measurement_batches = [b for b in all_results["batches"] if not b["summary"]["is_warmup"]]
    if measurement_batches:
        all_rollout_e2e = []
        all_throughputs = []
        all_group_ready_p95 = []
        all_sched_to_first = []
        all_first_to_done = []
        all_per_turn_ttft = defaultdict(list)
        all_per_turn_e2e = defaultdict(list)
        for b in measurement_batches:
            s = b["summary"]
            all_rollout_e2e.append(s["rollout_e2e_stats"]["mean"])
            all_throughputs.append(s.get("output_token_throughput", 0))
            gc = s.get("group_completion", {})
            if gc.get("p95"):
                all_group_ready_p95.append(gc["p95"])
            stf = s.get("scheduled_to_first_output_stats", {})
            if stf.get("mean"):
                all_sched_to_first.append(stf["mean"])
            ftc = s.get("first_output_to_completion_stats", {})
            if ftc.get("mean"):
                all_first_to_done.append(ftc["mean"])
            for turn_key, stats in s["per_turn_ttft"].items():
                all_per_turn_ttft[turn_key].append(stats["mean"])
            for turn_key, stats in s["per_turn_e2e"].items():
                all_per_turn_e2e[turn_key].append(stats["mean"])

        all_results["summary"] = {
            "concurrency": args.concurrency,
            "num_turns": args.num_turns,
            "rollout_e2e_mean_across_batches": compute_stats(all_rollout_e2e),
            "output_token_throughput": compute_stats(all_throughputs),
            "group_completion_p95": compute_stats(all_group_ready_p95),
            "scheduled_to_first_output_mean": compute_stats(all_sched_to_first),
            "first_output_to_completion_mean": compute_stats(all_first_to_done),
            "per_turn_ttft_mean_across_batches": {
                k: compute_stats(v) for k, v in sorted(all_per_turn_ttft.items())
            },
            "per_turn_e2e_mean_across_batches": {
                k: compute_stats(v) for k, v in sorted(all_per_turn_e2e.items())
            },
        }

    # Write output
    output_path.write_text(json.dumps(all_results, indent=2, default=str), encoding="utf-8")
    print(f"\n[OK] Results written to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
