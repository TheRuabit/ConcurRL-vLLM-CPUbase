#!/usr/bin/env python3
"""
Phase 2d: HTTP-based GRPO Rollout Driver
==========================================
A self-contained GRPO rollout + reward loop that talks to vLLM via HTTP
(OpenAI-compatible API), capturing the FULL per-request timing decomposition
from 03_concurrency_driver.py plus rollout-level E2E and gauge waveform plots.

Reuses from Phase 1:
  - aiohttp TraceConfig for 15 client-side timing metrics
  - VllmMetricsScraper for vLLM Prometheus histograms
  - MetricsPoller for running/waiting/kv_cache gauge time-series
  - JaegerTraceFetcher for OTel spans

Reuses from Phase 2:
  - math reward function (06_math_reward.py)
  - Per-step rollout decomposition (07b patterns)

Additional outputs:
  - Per-request 15-metric timing (same as 03)
  - Rollout-step E2E (wall-clock for entire rollout batch)
  - Gauge waveform PNGs (same as 03b)

Usage:
    python script/05d_grpo_http_driver.py --url http://localhost:8000
    python script/05d_grpo_http_driver.py --num-steps 5 --rollout-n 8
    python script/05d_grpo_http_driver.py --data-path data/dapo-math-17k.parquet

Output:
    result/05d_grpo_http_driver.json
    result/plots/05d_gauge_<metric>_step<N>.png
    result/plots/05d_gauge_<metric>_overview.png
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Parse CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="HTTP-based GRPO rollout driver with full timing instrumentation"
)
parser.add_argument("--url", default="http://localhost:8000",
                    help="vLLM server URL")
parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B",
                    help="Model name for API requests")
parser.add_argument("--output", default=None,
                    help="Output JSON path")
parser.add_argument("--data-path", default=None,
                    help="Path to training data parquet (default: data/dapo-math-17k.parquet)")
parser.add_argument("--max-samples", type=int, default=200,
                    help="Max samples to load from dataset")
parser.add_argument("--num-steps", type=int, default=3,
                    help="Number of GRPO rollout steps")
parser.add_argument("--rollout-n", type=int, default=8,
                    help="Number of rollouts per prompt (GRPO group size)")
parser.add_argument("--train-batch-size", type=int, default=8,
                    help="Prompts per rollout step")
parser.add_argument("--max-prompt-length", type=int, default=8192,
                    help="Max prompt token length")
parser.add_argument("--max-response-length", type=int, default=64,
                    help="Max response token length")
parser.add_argument("--temperature", type=float, default=0.7,
                    help="Sampling temperature")
parser.add_argument("--top-p", type=float, default=1.0,
                    help="Top-p sampling")
parser.add_argument("--request-timeout", type=int, default=300,
                    help="Per-request timeout in seconds")
parser.add_argument("--jaeger-url", default="http://localhost:16686",
                    help="Jaeger query API URL for OTel trace fetching")
parser.add_argument("--poll-interval", type=int, default=20,
                    help="Metrics poll interval in ms (0=disabled)")
parser.add_argument("--reward-func", default=None,
                    help="Path to custom reward function module")
parser.add_argument("--connector-limit", type=int, default=0,
                    help="aiohttp TCPConnector total connection limit (0=unlimited)")
parser.add_argument("--connector-limit-per-host", type=int, default=0,
                    help="aiohttp TCPConnector per-host connection limit (0=unlimited)")
parser.add_argument("--no-plots", action="store_true", default=False,
                    help="Skip gauge plot generation")
parser.add_argument("--scheduler-trace-path", default=None,
                    help="Optional vLLM scheduler JSONL trace path")
parser.add_argument("--scheduler-trace-window-padding-ms", type=int, default=200,
                    help="Padding around each rollout step when correlating scheduler trace events")
args = parser.parse_args()

project_dir = Path(__file__).resolve().parents[1]
if args.output:
    out_path = Path(args.output)
else:
    out_path = project_dir / "result" / "05d_grpo_http_driver.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
plots_dir = project_dir / "result" / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)

if args.data_path is None:
    args.data_path = str(project_dir / "data" / "dapo-math-17k.parquet")

# ---------------------------------------------------------------------------
# Imports (after CLI so --help is fast)
# ---------------------------------------------------------------------------
import aiohttp

# ---------------------------------------------------------------------------
# Load reward function
# ---------------------------------------------------------------------------
sys.path.insert(0, str(project_dir))
if args.reward_func:
    reward_path = Path(args.reward_func)
    if reward_path.exists():
        import importlib.util
        spec = importlib.util.spec_from_file_location("reward_module", str(reward_path))
        reward_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reward_mod)
        compute_score = reward_mod.compute_score
    else:
        print(f"[05d] WARNING: reward function not found at {reward_path}, using default")
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location(
            "reward_default",
            str(project_dir / "script" / "06_math_reward.py"),
        )
        _mod = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        compute_score = _mod.compute_score
else:
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location(
        "reward_default",
        str(project_dir / "script" / "06_math_reward.py"),
    )
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    compute_score = _mod.compute_score


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
def load_dataset(path: str, max_samples: int = 200) -> list[dict]:
    """Load math dataset from parquet. Returns list of {prompt, answer}."""
    try:
        import pyarrow.parquet as pq
        table = pq.read_table(path)
        df = table.to_pandas()
    except Exception as e:
        print(f"[05d] WARNING: Cannot load {path}: {e}")
        print(f"[05d] Using synthetic math problems instead")
        return _synthetic_dataset(max_samples)

    samples = []
    for _, row in df.iterrows():
        prompt = row.get("prompt", row.get("problem", row.get("question", "")))
        answer = row.get("answer", row.get("solution", row.get("ground_truth", "")))
        if prompt and answer:
            # Handle prompt formats
            if isinstance(prompt, list):
                # Chat format: extract last user message
                for msg in reversed(prompt):
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        prompt = msg.get("content", str(msg))
                        break
                else:
                    prompt = str(prompt[-1]) if prompt else ""
            elif isinstance(prompt, dict):
                prompt = prompt.get("content", str(prompt))
            samples.append({
                "prompt": str(prompt),
                "answer": str(answer),
                "data_source": "math_dapo",
            })
            if len(samples) >= max_samples:
                break

    if not samples:
        print(f"[05d] WARNING: No valid samples in {path}, using synthetic")
        return _synthetic_dataset(max_samples)

    return samples


def _synthetic_dataset(n: int) -> list[dict]:
    """Generate synthetic math problems for testing without real data."""
    import random
    random.seed(42)
    samples = []
    for i in range(n):
        a, b = random.randint(1, 100), random.randint(1, 100)
        op = random.choice(["+", "-", "*"])
        if op == "+":
            answer = str(a + b)
        elif op == "-":
            answer = str(a - b)
        else:
            answer = str(a * b)
        prompt = (
            f"Solve the following math problem step by step. "
            f"Put your final answer in \\boxed{{}}.\n\n"
            f"What is {a} {op} {b}?"
        )
        samples.append({"prompt": prompt, "answer": answer, "data_source": "synthetic"})
    return samples


# ---------------------------------------------------------------------------
# Reused from 03: HTTP trace config + timing infrastructure
# ---------------------------------------------------------------------------
def _get_http_trace(ctx):
    trace_request_ctx = getattr(ctx, "trace_request_ctx", None)
    if isinstance(trace_request_ctx, dict):
        return trace_request_ctx.get("http_trace")
    return None


def build_http_trace_config() -> aiohttp.TraceConfig:
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
# Reused from 03: Per-request trace dataclass (15 metrics)
# ---------------------------------------------------------------------------
@dataclass
class RequestTrace:
    step: int = 0
    prompt_idx: int = 0
    rollout_idx: int = 0
    prompt_tokens: int = 0
    # 15 timing metrics from 03
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
# Reused from 03: Single request profiler
# ---------------------------------------------------------------------------
async def trace_one_request(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict,
    semaphore: asyncio.Semaphore,
    timeout: int,
    step: int,
    prompt_idx: int,
    rollout_idx: int,
) -> RequestTrace:
    t = RequestTrace(step=step, prompt_idx=prompt_idx, rollout_idx=rollout_idx)
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
# Reused from 03: VllmMetricsScraper
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

    async def scrape(self, session: aiohttp.ClientSession) -> dict:
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
            "vllm_waiting_capacity": r'vllm:num_requests_waiting_by_reason\{[^}]*reason="capacity"[^}]*\}\s+([\d.eE+\-]+)',
            "vllm_waiting_deferred": r'vllm:num_requests_waiting_by_reason\{[^}]*reason="deferred"[^}]*\}\s+([\d.eE+\-]+)',
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
    for key in ["vllm_requests_running", "vllm_requests_waiting", "vllm_kv_cache_usage",
                "vllm_waiting_capacity", "vllm_waiting_deferred"]:
        if key in after:
            delta[key] = after[key]
    if "vllm_num_preemptions_total" in after and "vllm_num_preemptions_total" in before:
        delta["vllm_num_preemptions"] = after["vllm_num_preemptions_total"] - before["vllm_num_preemptions_total"]
    elif "vllm_num_preemptions_total" in after:
        delta["vllm_num_preemptions"] = after["vllm_num_preemptions_total"]
    return delta


# ---------------------------------------------------------------------------
# Reused from 03: MetricsPoller (gauge time-series)
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

    @staticmethod
    def render_ascii(samples: list[dict], key: str, width: int = 70, height: int = 15,
                     title: str = "") -> str:
        if not samples:
            return f"  (no data for {key})"
        values = [s.get(key, 0) for s in samples]
        times = [s.get("t", 0) for s in samples]
        max_val = max(values) if values else 1
        if max_val == 0:
            max_val = 1
        t_min, t_max = times[0], times[-1]
        t_range = t_max - t_min if t_max > t_min else 1
        bins: list[list[float]] = [[] for _ in range(width)]
        for t, v in zip(times, values):
            idx = int((t - t_min) / t_range * (width - 1))
            idx = max(0, min(width - 1, idx))
            bins[idx].append(v)
        bin_vals = [sum(b) / len(b) if b else 0 for b in bins]
        lines = []
        if title:
            lines.append(f"  {title}")
        lines.append(f"  {key} (max={max_val:.0f})")
        for row in range(height, -1, -1):
            threshold = max_val * row / height
            line = "  |"
            for bv in bin_vals:
                line += "#" if bv >= threshold else " "
            line += f" {threshold:>6.0f}"
            lines.append(line)
        lines.append("  +" + "-" * width)
        lines.append(f"  {t_min:>8.0f}ms{' ' * (width - 14)}{t_max:>8.0f}ms")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Reused from 03: Jaeger OTel trace fetcher
# ---------------------------------------------------------------------------
class JaegerTraceFetcher:
    def __init__(self, base_url: str = "http://localhost:16686"):
        self.api_url = f"{base_url.rstrip('/')}/api/traces"
        self.service_name: Optional[str] = None

    async def _discover_service(self, session: aiohttp.ClientSession) -> str:
        if self.service_name:
            return self.service_name
        try:
            url = self.api_url.replace("/traces", "/services")
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                data = await resp.json()
                services = data.get("data", [])
                for s in services:
                    if "vllm" in s.lower():
                        self.service_name = s
                        return s
                for s in services:
                    if s == "unknown_service":
                        self.service_name = s
                        return s
                for s in services:
                    if "jaeger" not in s.lower():
                        self.service_name = s
                        return s
                self.service_name = services[0] if services else "unknown_service"
                return self.service_name
        except Exception:
            self.service_name = "unknown_service"
            return self.service_name

    async def fetch_spans(self, session, start_us, end_us, limit=20000):
        service = await self._discover_service(session)
        try:
            params = {
                "service": service, "start": str(start_us),
                "end": str(end_us), "limit": str(limit),
            }
            async with session.get(self.api_url, params=params,
                                   timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
                spans = []
                for trace in data.get("data", []):
                    for span in trace.get("spans", []):
                        spans.append(span)
                return spans
        except Exception:
            return []

    @staticmethod
    def extract_gen_ai_latencies(spans):
        result = {}
        for span in spans:
            op = span.get("operationName", "")
            if "request" not in op.lower():
                continue
            tags = {t["key"]: t.get("value", 0) for t in span.get("tags", [])}
            dur_us = span.get("duration", 0)
            if dur_us <= 0:
                continue
            known_tags = [
                "gen_ai.latency.time_in_queue",
                "gen_ai.latency.time_in_model_prefill",
                "gen_ai.latency.time_in_model_decode",
                "gen_ai.latency.time_in_model_inference",
                "gen_ai.latency.time_to_first_token",
            ]
            span_total = dur_us / 1e6
            accounted = 0.0
            for tag_name in known_tags:
                val = tags.get(tag_name, 0)
                if isinstance(val, str):
                    try:
                        val = float(val)
                    except (ValueError, TypeError):
                        val = 0.0
                if val > 0:
                    result.setdefault(tag_name, []).append(val)
                    accounted += val
            tok_overhead = span_total - accounted
            if tok_overhead > 0:
                result.setdefault("tokenization_overhead", []).append(tok_overhead)
            result.setdefault("e2e", []).append(span_total)
        return result

    @staticmethod
    def compute_percentiles(values):
        if not values:
            return {}
        values.sort()
        n = len(values)
        def _pct(p):
            idx = (p / 100.0) * (n - 1)
            lo = int(idx)
            hi = min(lo + 1, n - 1)
            frac = idx - lo
            return values[lo] + frac * (values[hi] - values[lo])
        return {
            "count": n, "p50": _pct(50), "p95": _pct(95), "p99": _pct(99),
            "mean": sum(values) / n, "min": values[0], "max": values[-1],
        }


# ---------------------------------------------------------------------------
# Statistics helper
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


def load_scheduler_trace_events(
    path: Path,
    start_us: int,
    end_us: int,
    padding_ms: int,
) -> tuple[list[dict], int]:
    if path is None or not path.exists():
        return [], 0

    start_s = (start_us / 1e6) - (padding_ms / 1000.0)
    end_s = (end_us / 1e6) + (padding_ms / 1000.0)
    events = []
    malformed = 0

    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            ts = event.get("ts")
            if isinstance(ts, (int, float)) and start_s <= ts <= end_s:
                events.append(event)
    except OSError:
        return [], malformed

    return events, malformed


def _count_items(events: list[dict], key: str) -> int:
    total = 0
    for event in events:
        value = event.get(key, [])
        if isinstance(value, list):
            total += len(value)
        elif isinstance(value, dict):
            total += len(value)
    return total


def summarize_scheduler_trace_events(events: list[dict]) -> dict:
    schedule_ticks = [e for e in events if e.get("event") == "schedule_tick"]
    lookups = [e for e in events if e.get("event") == "kv_cache_lookup"]
    allocations = [e for e in events if e.get("event") == "kv_cache_allocate"]

    scheduled_tokens = [
        e.get("total_num_scheduled_tokens", 0)
        for e in schedule_ticks
        if isinstance(e.get("total_num_scheduled_tokens", 0), (int, float))
    ]
    free_blocks = [
        e.get("free_blocks_after")
        for e in allocations
        if isinstance(e.get("free_blocks_after"), (int, float))
    ]
    usages = [
        e.get("kv_cache_usage_after")
        for e in allocations
        if isinstance(e.get("kv_cache_usage_after"), (int, float))
    ]

    success_allocs = [e for e in allocations if e.get("allocated") is True]
    failed_allocs = [e for e in allocations if e.get("allocated") is False]

    return {
        "num_schedule_ticks": len(schedule_ticks),
        "scheduled_tokens_total": sum(scheduled_tokens),
        "scheduled_tokens_mean_per_tick": (
            round(sum(scheduled_tokens) / len(scheduled_tokens), 4)
            if scheduled_tokens else 0
        ),
        "scheduled_new_req_count": _count_items(schedule_ticks, "scheduled_new_req_ids"),
        "scheduled_cached_req_count": _count_items(schedule_ticks, "scheduled_cached_req_ids"),
        "scheduled_resumed_req_count": _count_items(schedule_ticks, "scheduled_resumed_req_ids"),
        "preempted_req_count": _count_items(schedule_ticks, "preempted_req_ids"),
        "finished_req_count": _count_items(schedule_ticks, "finished_req_ids"),
        "kv_lookup_count": len(lookups),
        "kv_local_cached_tokens_total": sum(
            e.get("local_cached_tokens", 0)
            for e in lookups
            if isinstance(e.get("local_cached_tokens", 0), (int, float))
        ),
        "kv_allocate_attempt_count": len(allocations),
        "kv_allocate_success_count": len(success_allocs),
        "kv_allocate_failure_count": len(failed_allocs),
        "kv_blocks_allocated_total": sum(
            e.get("allocated_block_count", 0)
            for e in success_allocs
            if isinstance(e.get("allocated_block_count", 0), (int, float))
        ),
        "kv_free_blocks_min": min(free_blocks) if free_blocks else 0,
        "kv_free_blocks_max": max(free_blocks) if free_blocks else 0,
        "kv_usage_max": max(usages) if usages else 0,
    }


FIRST_TOKEN_STAGE_EVENTS = {
    "openai_chat_request_start",
    "openai_chat_render_done",
    "openai_chat_generate_start",
    "async_llm_add_request_start",
    "async_llm_input_processed",
    "async_llm_engine_enqueue_start",
    "async_llm_engine_enqueue_done",
    "engine_core_preprocess_done",
    "engine_core_add_request",
    "scheduler_first_seen",
    "scheduler_first_scheduled",
    "async_llm_first_output",
    "openai_chat_first_result",
    "openai_chat_first_chunk_yield",
}


FIRST_TOKEN_STAGE_PAIRS = {
    "server_frontend_render_ms": (
        "openai_chat_request_start", "openai_chat_render_done",
    ),
    "server_generate_to_add_ms": (
        "openai_chat_generate_start", "async_llm_add_request_start",
    ),
    "server_input_process_ms": (
        "async_llm_add_request_start", "async_llm_input_processed",
    ),
    "server_engine_enqueue_ms": (
        "async_llm_engine_enqueue_start", "async_llm_engine_enqueue_done",
    ),
    "server_core_preprocess_ms": (
        "async_llm_engine_enqueue_done", "engine_core_preprocess_done",
    ),
    "server_scheduler_wait_ms": (
        "engine_core_add_request", "scheduler_first_scheduled",
    ),
    "server_first_schedule_to_output_ms": (
        "scheduler_first_scheduled", "async_llm_first_output",
    ),
    "server_output_to_stream_result_ms": (
        "async_llm_first_output", "openai_chat_first_result",
    ),
    "server_stream_result_to_first_chunk_ms": (
        "openai_chat_first_result", "openai_chat_first_chunk_yield",
    ),
}


def derive_first_token_stages(
    traces: list[RequestTrace],
    events: list[dict],
) -> tuple[list[dict], dict]:
    by_req: dict[str, dict[str, float]] = {}
    for event in events:
        req_id = event.get("req_id")
        event_name = event.get("event")
        perf_ts = event.get("perf_ts")
        if (
            not isinstance(req_id, str)
            or event_name not in FIRST_TOKEN_STAGE_EVENTS
            or not isinstance(perf_ts, (int, float))
        ):
            continue
        by_req.setdefault(req_id, {}).setdefault(event_name, perf_ts)

    rows = []
    for trace in traces:
        req_id = trace.vllm_request_id
        if not req_id:
            continue
        event_times = dict(by_req.get(req_id, {}))
        for candidate_req_id, candidate_events in by_req.items():
            if candidate_req_id.startswith(f"{req_id}-"):
                for event_name, perf_ts in candidate_events.items():
                    event_times.setdefault(event_name, perf_ts)
        if not event_times:
            continue
        row = {
            "req_id": req_id,
            "prompt_idx": trace.prompt_idx,
            "rollout_idx": trace.rollout_idx,
            "client_headers_wait_ms": round(
                trace.t_http_response_headers_wait_ms, 4
            ),
            "client_header_to_first_sse_ms": round(
                trace.t_response_header_to_first_sse_byte_ms, 4
            ),
            "client_server_prefill_ms": round(trace.t_server_prefill_ms, 4),
        }
        for metric, (start_event, end_event) in FIRST_TOKEN_STAGE_PAIRS.items():
            start = event_times.get(start_event)
            end = event_times.get(end_event)
            if start is not None and end is not None:
                row[metric] = round(max(0.0, (end - start) * 1000), 4)
        rows.append(row)

    summary: dict[str, dict] = {}
    if rows:
        for metric in [
            "client_headers_wait_ms",
            "client_header_to_first_sse_ms",
            "client_server_prefill_ms",
            *FIRST_TOKEN_STAGE_PAIRS.keys(),
        ]:
            values = [
                row[metric] for row in rows
                if isinstance(row.get(metric), (int, float))
            ]
            if values:
                summary[metric] = compute_stats(values)
        summary["matched_request_count"] = len(rows)
        summary["available_request_event_count"] = len(by_req)
    else:
        summary["matched_request_count"] = 0
        summary["available_request_event_count"] = len(by_req)

    return rows, summary


# ---------------------------------------------------------------------------
# GRPO prompt formatter
# ---------------------------------------------------------------------------
def format_prompt(problem_text: str) -> str:
    """Format a math problem as a chat prompt for vLLM."""
    return (
        "Solve the following math problem step by step. "
        "Put your final answer in \\boxed{}.\n\n"
        f"{problem_text}"
    )


def build_payload(prompt_text: str, model: str, max_tokens: int,
                  temperature: float, top_p: float) -> dict:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt_text},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": True,
    }


# ---------------------------------------------------------------------------
# Gauge waveform plotter (from 03b)
# ---------------------------------------------------------------------------
def generate_gauge_plots(step_poll_samples: dict, plots_dir: Path):
    """Generate gauge waveform PNGs from per-step poll_samples."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[05d] WARNING: matplotlib not available, skipping plots")
        return

    METRICS = [
        ("running",   "num_requests_running",   "#2563eb"),
        ("waiting",   "num_requests_waiting",   "#dc2626"),
        ("kv_cache",  "KV Cache Usage (%)",      "#16a34a"),
    ]

    steps = sorted(step_poll_samples.keys())

    # 1. Per-step individual plots
    for step in steps:
        samples = step_poll_samples[step]
        if not samples:
            continue
        fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
        fig.suptitle(f"GRPO Step {step}  ({len(samples)} samples, "
                     f"{samples[-1]['t']:.0f}ms duration)",
                     fontsize=14, fontweight="bold")
        for ax, (key, label, color) in zip(axes, METRICS):
            times = [s["t"] for s in samples]
            values = [s.get(key, 0) for s in samples]
            if not times:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=12, color="gray")
                ax.set_ylabel(label)
                continue
            t_arr = np.array(times)
            v_arr = np.array(values)
            ax.fill_between(t_arr, v_arr, alpha=0.3, color=color)
            ax.plot(t_arr, v_arr, linewidth=0.6, color=color)
            ax.set_ylabel(label, fontsize=10)
            ax.set_ylim(bottom=0)
            ax.grid(True, alpha=0.3)
            max_idx = np.argmax(v_arr)
            ax.annotate(f"max={v_arr[max_idx]:.0f}",
                        xy=(t_arr[max_idx], v_arr[max_idx]),
                        xytext=(10, 10), textcoords="offset points",
                        fontsize=9, color=color, fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color=color, lw=0.8))
        axes[-1].set_xlabel("Time (ms)", fontsize=10)
        plt.tight_layout()
        fname = plots_dir / f"05d_gauge_step{step}.png"
        fig.savefig(fname, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  {fname}")

    # 2. Overview: running across all steps
    if len(steps) > 1:
        fig, axes = plt.subplots(len(steps), 1, figsize=(14, 2.5 * len(steps)),
                                 sharex=False)
        if len(steps) == 1:
            axes = [axes]
        fig.suptitle("num_requests_running — All GRPO Steps", fontsize=14, fontweight="bold")
        for ax, step in zip(axes, steps):
            samples = step_poll_samples[step]
            if not samples:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=10, color="gray")
                ax.set_ylabel(f"Step={step}")
                continue
            times = [s["t"] for s in samples]
            values = [s.get("running", 0) for s in samples]
            t_arr = np.array(times)
            v_arr = np.array(values)
            t_norm = (t_arr - t_arr[0]) / (t_arr[-1] - t_arr[0]) * 100
            ax.fill_between(t_norm, v_arr, alpha=0.3, color="#2563eb")
            ax.plot(t_norm, v_arr, linewidth=0.5, color="#2563eb")
            ax.set_ylabel(f"Step={step}\n(max={max(values):.0f})", fontsize=9)
            ax.set_ylim(0, max(max(values) * 1.15, 10))
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel("Step Progress (%)", fontsize=10)
        plt.tight_layout()
        fname = plots_dir / "05d_gauge_running_overview.png"
        fig.savefig(fname, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  {fname}")


# ---------------------------------------------------------------------------
# Main GRPO loop
# ---------------------------------------------------------------------------
async def main():
    url = args.url.rstrip("/")
    print(f"[05d] HTTP-based GRPO Rollout Driver")
    print(f"[05d] Server:         {url}")
    print(f"[05d] Model:          {args.model}")
    print(f"[05d] Steps:          {args.num_steps}")
    print(f"[05d] Rollout N:      {args.rollout_n}")
    print(f"[05d] Batch size:     {args.train_batch_size}")
    print(f"[05d] Concurrency:    {args.rollout_n * args.train_batch_size}")
    print(f"[05d] Max resp len:   {args.max_response_length}")
    print(f"[05d] Temperature:    {args.temperature}")
    print(f"[05d] Output:         {out_path}")
    print()

    # Load dataset
    dataset = load_dataset(args.data_path, args.max_samples)
    print(f"[05d] Loaded {len(dataset)} samples from {args.data_path}")

    scraper = VllmMetricsScraper(url)
    jaeger = JaegerTraceFetcher(args.jaeger_url)
    all_traces: list[RequestTrace] = []
    step_summaries: list[dict] = []
    step_vllm_metrics: dict[str, dict] = {}
    step_poll_samples: dict[int, list[dict]] = {}
    reward_timings: list[dict] = []

    # Check server availability
    async with aiohttp.ClientSession() as probe_session:
        baseline_metrics = await scraper.scrape(probe_session)
        if baseline_metrics:
            print(f"[05d] vLLM metrics endpoint detected")
        else:
            print(f"[05d] vLLM /metrics not available — using client-side timing only")

        jaeger_ok = False
        try:
            svc = await jaeger._discover_service(probe_session)
            print(f"[05d] Jaeger detected (service={svc})")
            jaeger_ok = True
        except Exception:
            print(f"[05d] Jaeger not available — skipping OTel traces")

    t_global_start = time.perf_counter()
    concurrency = args.rollout_n * args.train_batch_size

    # Unlimited connector by default (limit=0) to avoid aiohttp's default
    # 100-connection cap bottlenecking high-concurrency rollouts (256+).
    connector = aiohttp.TCPConnector(
        limit=args.connector_limit,
        limit_per_host=args.connector_limit_per_host,
    )
    async with aiohttp.ClientSession(
        connector=connector,
        trace_configs=[build_http_trace_config()],
    ) as session:
        for step in range(args.num_steps):
            print(f"\n{'='*70}")
            print(f"  GRPO Step {step}/{args.num_steps}")
            print(f"{'='*70}")

            # Select batch of prompts
            batch_start = (step * args.train_batch_size) % len(dataset)
            batch_samples = []
            for i in range(args.train_batch_size):
                idx = (batch_start + i) % len(dataset)
                batch_samples.append(dataset[idx])

            # Build all rollout payloads: N rollouts per prompt
            sem = asyncio.Semaphore(concurrency)
            tasks = []
            task_meta = []  # (prompt_idx, rollout_idx, sample)
            for pi, sample in enumerate(batch_samples):
                prompt_text = format_prompt(sample["prompt"])
                for ri in range(args.rollout_n):
                    payload = build_payload(
                        prompt_text, args.model,
                        args.max_response_length,
                        args.temperature, args.top_p,
                    )
                    tasks.append(trace_one_request(
                        session, url, payload, sem, args.request_timeout,
                        step, pi, ri,
                    ))
                    task_meta.append((pi, ri, sample))

            # Start gauge poller
            poller = MetricsPoller(url, args.poll_interval) if args.poll_interval > 0 else None
            step_start_us = int(time.time() * 1e6)
            if poller:
                poller.start()

            # Scrape baseline vLLM metrics
            baseline_step = await scraper.scrape(session)

            # Execute all rollouts concurrently
            t_rollout_start = time.perf_counter()
            print(f"  Dispatching {len(tasks)} rollouts (B={args.train_batch_size} x G={args.rollout_n})...")
            results = await asyncio.gather(*tasks)
            t_rollout_end = time.perf_counter()
            t_rollout_ms = (t_rollout_end - t_rollout_start) * 1000

            # Stop poller
            poll_samples = []
            if poller:
                poll_samples = await poller.stop()
            step_poll_samples[step] = poll_samples

            step_end_us = int(time.time() * 1e6)

            # Scrape post-step vLLM metrics
            post_step = await scraper.scrape(session)
            vllm_delta = {}
            if baseline_step and post_step:
                vllm_delta = compute_metrics_delta(baseline_step, post_step)

            scheduler_events = []
            scheduler_malformed = 0
            scheduler_summary = {}
            server_first_token_stages = []
            server_first_token_stage_summary = {}
            if args.scheduler_trace_path:
                scheduler_events, scheduler_malformed = load_scheduler_trace_events(
                    Path(args.scheduler_trace_path),
                    step_start_us,
                    step_end_us,
                    args.scheduler_trace_window_padding_ms,
                )
                scheduler_summary = summarize_scheduler_trace_events(scheduler_events)
                server_first_token_stages, server_first_token_stage_summary = (
                    derive_first_token_stages(results, scheduler_events)
                )

            # Fetch OTel spans
            if jaeger_ok:
                try:
                    async with aiohttp.ClientSession() as j_session:
                        spans = await jaeger.fetch_spans(
                            j_session,
                            step_start_us - int(5 * 1e6),
                            step_end_us + int(5 * 1e6),
                        )
                        if spans:
                            latencies = jaeger.extract_gen_ai_latencies(spans)
                            for tag_name, values in latencies.items():
                                key = "otel_" + tag_name.replace("gen_ai.latency.", "").replace(".", "_")
                                vllm_delta[key] = jaeger.compute_percentiles(values)
                            vllm_delta["otel_span_count"] = len(spans)
                except Exception:
                    pass

            # Collect traces
            all_traces.extend(results)

            # Compute rewards
            t_reward_start = time.perf_counter()
            rewards = []
            for ri, (trace, (pi, rollout_i, sample)) in enumerate(zip(results, task_meta)):
                if trace.success and trace.output_text:
                    reward = compute_score(
                        sample.get("data_source", "unknown"),
                        trace.output_text,
                        sample["answer"],
                    )
                else:
                    reward = 0.0
                rewards.append(reward)
                reward_timings.append({
                    "step": step,
                    "prompt_idx": pi,
                    "rollout_idx": rollout_i,
                    "reward": reward,
                    "success": trace.success,
                })
            t_reward_end = time.perf_counter()
            t_reward_ms = (t_reward_end - t_reward_start) * 1000

            # Compute GRPO advantages (normalize within each prompt group)
            advantages = []
            for pi in range(args.train_batch_size):
                group_rewards = [
                    rewards[gi] for gi in range(len(rewards))
                    if task_meta[gi][0] == pi
                ]
                group_mean = sum(group_rewards) / len(group_rewards) if group_rewards else 0
                group_std = (sum((r - group_mean) ** 2 for r in group_rewards) / len(group_rewards)) ** 0.5 if len(group_rewards) > 1 else 1.0
                group_std = max(group_std, 1e-8)
                for gi in range(len(rewards)):
                    if task_meta[gi][0] == pi:
                        advantages.append((rewards[gi] - group_mean) / group_std)

            # Per-step statistics
            successful = [r for r in results if r.success]
            step_traces = list(results)

            # Group completion time (from step start to each request's end)
            completion_offsets = sorted(
                [(r.t_e2e_ms) for r in results if r.success]
            )

            reward_sum = sum(rewards)
            reward_mean = reward_sum / len(rewards) if rewards else 0

            step_summary = {
                "step": step,
                "t_rollout_ms": round(t_rollout_ms, 2),
                "t_reward_ms": round(t_reward_ms, 2),
                "t_step_total_ms": round(t_rollout_ms + t_reward_ms, 2),
                "num_requests": len(results),
                "num_successful": len(successful),
                "reward_sum": round(reward_sum, 4),
                "reward_mean": round(reward_mean, 4),
                "advantage_mean": round(sum(advantages) / len(advantages), 4) if advantages else 0,
                "output_tokens_total": sum(r.num_output_tokens for r in successful),
                "output_token_throughput": round(
                    sum(r.num_output_tokens for r in successful) / (t_rollout_ms / 1000), 2
                ) if t_rollout_ms > 0 else 0,
            }

            # Per-request latency stats
            if successful:
                for key in [
                    "t_serialize_ms", "t_sem_wait_ms", "t_http_connect_ms",
                    "t_http_conn_queued_ms", "t_http_dns_ms", "t_http_tcp_connect_ms",
                    "t_http_request_send_ms", "t_http_response_headers_wait_ms",
                    "t_response_header_to_first_sse_byte_ms",
                    "t_first_sse_byte_to_first_token_ms",
                    "t_first_byte_ms", "t_server_prefill_ms", "t_prefill_ms",
                    "t_decode_ms", "t_response_parse_ms", "t_e2e_ms",
                ]:
                    values = [getattr(r, key) for r in successful]
                    step_summary[f"per_request_{key}"] = compute_stats(values)

                # Group completion percentiles
                if completion_offsets:
                    n = len(completion_offsets)
                    step_summary["group_completion_p50_ms"] = round(completion_offsets[int(n * 0.50)], 2)
                    step_summary["group_completion_p95_ms"] = round(completion_offsets[min(int(n * 0.95), n - 1)], 2)
                    step_summary["group_completion_p99_ms"] = round(completion_offsets[min(int(n * 0.99), n - 1)], 2)

            step_summary["vllm_metrics_delta"] = vllm_delta
            step_summary["poll_samples"] = poll_samples
            step_summary["poll_sample_count"] = len(poll_samples)
            step_summary["scheduler_trace_available"] = bool(scheduler_events)
            step_summary["scheduler_trace_event_count"] = len(scheduler_events)
            step_summary["scheduler_trace_malformed_count"] = scheduler_malformed
            step_summary["scheduler_trace_summary"] = scheduler_summary
            step_summary["scheduler_schedule_ticks"] = [
                e for e in scheduler_events if e.get("event") == "schedule_tick"
            ][:200]
            step_summary["scheduler_kv_allocations"] = [
                e for e in scheduler_events if e.get("event") == "kv_cache_allocate"
            ][:200]
            step_summary["scheduler_cache_lookups"] = [
                e for e in scheduler_events if e.get("event") == "kv_cache_lookup"
            ][:200]
            step_summary["server_first_token_stage_summary"] = (
                server_first_token_stage_summary
            )
            step_summary["server_first_token_stages"] = server_first_token_stages[:200]
            step_summaries.append(step_summary)
            step_vllm_metrics[str(step)] = vllm_delta

            # Print step summary
            print(f"  Rollout: {t_rollout_ms:.0f}ms | Reward: {t_reward_ms:.1f}ms | "
                  f"OK: {len(successful)}/{len(results)} | "
                  f"Reward mean: {reward_mean:.3f} | "
                  f"Tok/s: {step_summary.get('output_token_throughput', 0):.0f}")
            if successful:
                e2e_stats = step_summary.get("per_request_t_e2e_ms", {})
                print(f"  Per-request E2E: P50={e2e_stats.get('p50', 0):.0f}ms  "
                      f"P95={e2e_stats.get('p95', 0):.0f}ms  "
                      f"P99={e2e_stats.get('p99', 0):.0f}ms")
                prefill_stats = step_summary.get("per_request_t_server_prefill_ms", {})
                print(f"  Server Prefill:   P50={prefill_stats.get('p50', 0):.0f}ms  "
                      f"P95={prefill_stats.get('p95', 0):.0f}ms")
                decode_stats = step_summary.get("per_request_t_decode_ms", {})
                print(f"  Decode:           P50={decode_stats.get('p50', 0):.0f}ms  "
                      f"P95={decode_stats.get('p95', 0):.0f}ms")
            if scheduler_summary:
                print(
                    "  Scheduler trace: "
                    f"ticks={scheduler_summary.get('num_schedule_ticks', 0)} "
                    f"tokens={scheduler_summary.get('scheduled_tokens_total', 0)} "
                    f"alloc_fail={scheduler_summary.get('kv_allocate_failure_count', 0)} "
                    f"preempt={scheduler_summary.get('preempted_req_count', 0)}"
                )

            if poll_samples:
                print(MetricsPoller.render_ascii(
                    poll_samples, "running", width=70, height=8,
                    title=f"  [Step={step}] num_requests_running"))

        t_global_end = time.perf_counter()
        t_total_s = t_global_end - t_global_start

    # -------------------------------------------------------------------
    # Aggregate statistics
    # -------------------------------------------------------------------
    all_successful = [r for r in all_traces if r.success]
    aggregate_request_stats = {}
    if all_successful:
        for key in [
            "t_serialize_ms", "t_sem_wait_ms", "t_http_connect_ms",
            "t_http_conn_queued_ms", "t_http_dns_ms", "t_http_tcp_connect_ms",
            "t_http_request_send_ms", "t_http_response_headers_wait_ms",
            "t_response_header_to_first_sse_byte_ms",
            "t_first_sse_byte_to_first_token_ms",
            "t_first_byte_ms", "t_server_prefill_ms", "t_prefill_ms",
            "t_decode_ms", "t_response_parse_ms", "t_e2e_ms",
            "num_output_tokens",
        ]:
            values = [getattr(r, key) for r in all_successful]
            aggregate_request_stats[key] = compute_stats(values)

    # Rollout step stats
    step_rollout_times = [s["t_rollout_ms"] for s in step_summaries]
    step_reward_times = [s["t_reward_ms"] for s in step_summaries]
    step_total_times = [s["t_step_total_ms"] for s in step_summaries]
    step_rewards = [s["reward_mean"] for s in step_summaries]
    step_throughputs = [s.get("output_token_throughput", 0) for s in step_summaries]

    # -------------------------------------------------------------------
    # Build raw traces
    # -------------------------------------------------------------------
    raw_traces = []
    for r in all_traces:
        raw_traces.append({
            "step": r.step,
            "prompt_idx": r.prompt_idx,
            "rollout_idx": r.rollout_idx,
            "prompt_tokens": r.prompt_tokens,
            "vllm_request_id": r.vllm_request_id,
            "t_serialize_ms": round(r.t_serialize_ms, 4),
            "t_sem_wait_ms": round(r.t_sem_wait_ms, 4),
            "t_http_connect_ms": round(r.t_http_connect_ms, 4),
            "t_http_conn_queued_ms": round(r.t_http_conn_queued_ms, 4),
            "t_http_dns_ms": round(r.t_http_dns_ms, 4),
            "t_http_tcp_connect_ms": round(r.t_http_tcp_connect_ms, 4),
            "t_http_request_send_ms": round(r.t_http_request_send_ms, 4),
            "t_http_response_headers_wait_ms": round(r.t_http_response_headers_wait_ms, 4),
            "t_response_header_to_first_sse_byte_ms": round(r.t_response_header_to_first_sse_byte_ms, 4),
            "t_first_sse_byte_to_first_token_ms": round(r.t_first_sse_byte_to_first_token_ms, 4),
            "t_first_byte_ms": round(r.t_first_byte_ms, 4),
            "t_server_prefill_ms": round(r.t_server_prefill_ms, 4),
            "t_prefill_ms": round(r.t_prefill_ms, 4),
            "t_decode_ms": round(r.t_decode_ms, 4),
            "t_response_parse_ms": round(r.t_response_parse_ms, 4),
            "t_e2e_ms": round(r.t_e2e_ms, 4),
            "num_output_tokens": r.num_output_tokens,
            "output_text": r.output_text[:500],  # Truncate for JSON size
            "success": r.success,
            "error": r.error,
        })

    # -------------------------------------------------------------------
    # Save JSON
    # -------------------------------------------------------------------
    output = {
        "benchmark": "grpo_http_driver",
        "server_url": url,
        "model": args.model,
        "num_steps": args.num_steps,
        "rollout_n": args.rollout_n,
        "train_batch_size": args.train_batch_size,
        "concurrency": concurrency,
        "max_prompt_length": args.max_prompt_length,
        "max_response_length": args.max_response_length,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "data_path": args.data_path,
        "num_dataset_samples": len(dataset),
        "total_elapsed_s": round(t_total_s, 2),
        "num_requests_total": len(all_traces),
        "num_successful_total": len(all_successful),
        "aggregate_request_stats": aggregate_request_stats,
        "step_summaries": step_summaries,
        "vllm_metrics": step_vllm_metrics,
        "reward_summary": compute_stats([r["reward"] for r in reward_timings]),
        "rollout_step_times": compute_stats(step_rollout_times),
        "reward_step_times": compute_stats(step_reward_times),
        "total_step_times": compute_stats(step_total_times),
        "step_reward_means": compute_stats(step_rewards),
        "step_throughputs": compute_stats(step_throughputs),
        "raw_traces": raw_traces,
    }

    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\n[05d] Results saved to {out_path}")

    # -------------------------------------------------------------------
    # Generate gauge plots
    # -------------------------------------------------------------------
    if not args.no_plots and step_poll_samples:
        print(f"\n[05d] Generating gauge waveform plots...")
        generate_gauge_plots(step_poll_samples, plots_dir)

    # -------------------------------------------------------------------
    # Console summary
    # -------------------------------------------------------------------
    print(f"\n{'='*130}")
    print(f"HTTP GRPO ROLLOUT DRIVER — Summary")
    print(f"{'='*130}")
    print(f"  Server:     {url}")
    print(f"  Model:      {args.model}")
    print(f"  Steps:      {args.num_steps}")
    print(f"  B x G:      {args.train_batch_size} x {args.rollout_n} = {concurrency}")
    print(f"  Total time: {t_total_s:.1f}s")
    print(f"  Requests:   {len(all_traces)} ({len(all_successful)} OK)")
    print()

    # Rollout timing table
    header = (f"{'Step':>4s} {'Rollout':>10s} {'Reward':>10s} {'Total':>10s} "
              f"{'OK':>5s} {'Reward':>8s} {'Tok/s':>8s} "
              f"{'E2E_P50':>10s} {'E2E_P95':>10s} {'Prefill_P95':>12s} {'Decode_P95':>12s}")
    print(header)
    print("-" * 130)
    for s in step_summaries:
        e2e = s.get("per_request_t_e2e_ms", {})
        prefill = s.get("per_request_t_server_prefill_ms", {})
        decode = s.get("per_request_t_decode_ms", {})
        print(f"{s['step']:4d} "
              f"{s['t_rollout_ms']:8.0f}ms "
              f"{s['t_reward_ms']:8.1f}ms "
              f"{s['t_step_total_ms']:8.0f}ms "
              f"{s['num_successful']:5d} "
              f"{s['reward_mean']:8.3f} "
              f"{s.get('output_token_throughput', 0):6.0f}t/s "
              f"{e2e.get('p50', 0):8.0f}ms "
              f"{e2e.get('p95', 0):8.0f}ms "
              f"{prefill.get('p95', 0):10.0f}ms "
              f"{decode.get('p95', 0):10.0f}ms")

    # Aggregate stats
    if aggregate_request_stats:
        print(f"\n{'='*130}")
        print(f"AGGREGATE PER-REQUEST TIMING (all steps, {len(all_successful)} successful)")
        print(f"{'='*130}")
        header2 = f"{'Metric':>42s} {'Mean':>10s} {'P50':>10s} {'P95':>10s} {'P99':>10s} {'Min':>10s} {'Max':>10s}"
        print(header2)
        print("-" * 130)
        for key, label in [
            ("t_serialize_ms", "Serialize"),
            ("t_sem_wait_ms", "Semaphore Wait"),
            ("t_http_connect_ms", "HTTP Connect"),
            ("t_http_conn_queued_ms", "HTTP Conn Queued"),
            ("t_http_dns_ms", "DNS"),
            ("t_http_tcp_connect_ms", "TCP Connect"),
            ("t_http_request_send_ms", "Request Send"),
            ("t_http_response_headers_wait_ms", "Response Headers Wait"),
            ("t_response_header_to_first_sse_byte_ms", "Header → 1st SSE"),
            ("t_first_sse_byte_to_first_token_ms", "1st SSE → 1st Token"),
            ("t_first_byte_ms", "First Byte (sem+http)"),
            ("t_server_prefill_ms", "Server Prefill (TTFT)"),
            ("t_prefill_ms", "Prefill (TTFT - 1st byte)"),
            ("t_decode_ms", "Decode"),
            ("t_response_parse_ms", "Response Parse"),
            ("t_e2e_ms", "E2E"),
            ("num_output_tokens", "Output Tokens"),
        ]:
            st = aggregate_request_stats.get(key, {})
            if not st:
                continue
            unit = "" if key == "num_output_tokens" else "ms"
            print(f"{label:>42s} "
                  f"{st['mean']:8.2f}{unit} "
                  f"{st['p50']:8.2f}{unit} "
                  f"{st['p95']:8.2f}{unit} "
                  f"{st['p99']:8.2f}{unit} "
                  f"{st['min']:8.2f}{unit} "
                  f"{st['max']:8.2f}{unit}")

    # vLLM server-side metrics
    has_server_metrics = any(v for v in step_vllm_metrics.values() if v)
    if has_server_metrics:
        print(f"\n{'='*110}")
        print(f"vLLM SERVER-SIDE METRICS (from /metrics endpoint)")
        print(f"{'='*110}")
        header3 = (f"{'Step':>4s} {'QueueAvg':>10s} {'PrefillAvg':>10s} "
                   f"{'DecodeAvg':>10s} {'TTFTEvg':>10s} {'E2EAvg':>10s} "
                   f"{'ITLAvg':>10s} {'Preempt':>8s} {'Run':>5s} {'Wait':>5s} {'KV%':>5s}")
        print(header3)
        print("-" * 110)
        for step_idx in range(args.num_steps):
            vm = step_vllm_metrics.get(str(step_idx), {})
            if not vm:
                print(f"{step_idx:>4d} {'N/A':>10s}")
                continue
            q = vm.get("vllm_queue_time_avg", 0) * 1000
            p = vm.get("vllm_prefill_time_avg", 0) * 1000
            d = vm.get("vllm_decode_time_avg", 0) * 1000
            t = vm.get("vllm_ttft_avg", 0) * 1000
            e = vm.get("vllm_e2e_avg", 0) * 1000
            itl = vm.get("vllm_itl_avg", 0) * 1000
            pr = vm.get("vllm_num_preemptions", 0)
            r = vm.get("vllm_requests_running", 0)
            w = vm.get("vllm_requests_waiting", 0)
            kv = vm.get("vllm_kv_cache_usage", 0) * 100
            print(f"{step_idx:>4d} {q:>8.1f}ms {p:>8.1f}ms "
                  f"{d:>8.1f}ms {t:>8.1f}ms {e:>8.1f}ms {itl:>8.1f}ms "
                  f"{pr:>8.0f} {r:>5.0f} {w:>5.0f} {kv:>5.1f}%")

    print(f"{'='*130}")


if __name__ == "__main__":
    asyncio.run(main())
