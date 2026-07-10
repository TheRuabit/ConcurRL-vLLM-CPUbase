#!/usr/bin/env python3
"""
High-Concurrency Async Profiler Client
=======================================
Issues burst requests concurrently via aiohttp while capturing detailed
latency metrics per request. Sequentially steps through concurrency tiers,
dumping raw performance arrays to JSON.

Three-layer timing decomposition:
  t_serialize        — client JSON serialization time
  t_sem_wait_ms      — semaphore wait (client-side concurrency gate)
  t_http_connect_ms  — POST → first HTTP byte (server queue + network)
  t_http_conn_queued_ms — aiohttp connector pool wait
  t_http_dns_ms         — DNS lookup
  t_http_tcp_connect_ms — TCP connection establishment
  t_http_request_send_ms — request body send after headers
  t_http_response_headers_wait_ms — request sent → response headers
  t_response_header_to_first_sse_byte_ms — response headers → first SSE line
  t_first_sse_byte_to_first_token_ms — first SSE line → first content token
  t_server_prefill   — POST → first content token (TTFT from client perspective)
  t_prefill          — TTFT − first_byte (pure GPU attention over input)
  t_decode           — first → last content token interval
  t_response_parse   — first byte → end of SSE stream

Server-side metrics (from vLLM /metrics endpoint):
  vllm_queue_time_s    — request_queue_time_seconds (scheduler queue)
  vllm_prefill_time_s  — request_prefill_time_seconds
  vllm_decode_time_s   — request_decode_time_seconds

Default: input 16k tokens → output 64 tokens

Usage:
    python script/03_concurrency_driver.py --url http://localhost:8000
    python script/03_concurrency_driver.py --scenarios 32 64 128 256
    python script/03_concurrency_driver.py --num-batches 5 --warmup-batches 2

Output:
    result/03_concurrency_driver.json
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
    description="High-concurrency async profiler — sweeps 32 to 1024"
)
parser.add_argument("--url", default="http://localhost:8000",
                    help="vLLM server URL")
parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B",
                    help="Model name for API requests")
parser.add_argument("--output", default=None,
                    help="Output JSON path")
parser.add_argument("--input-tokens", type=int, default=16000,
                    help="Simulated input token count")
parser.add_argument("--output-tokens", type=int, default=64,
                    help="Max output tokens per request")
parser.add_argument("--scenarios", nargs="+", type=int,
                    default=[32, 64, 128, 256, 512, 1024],
                    help="Concurrency levels to test")
parser.add_argument("--num-batches", type=int, default=3,
                    help="Number of measurement batches per concurrency level")
parser.add_argument("--warmup-batches", type=int, default=1,
                    help="Warmup batches before measurement")
parser.add_argument("--request-timeout", type=int, default=600,
                    help="Per-request timeout in seconds")
parser.add_argument("--connector-limit", type=int, default=None,
                    help=("aiohttp TCPConnector total connection limit. "
                          "Omit for aiohttp default; use 0 for unlimited."))
parser.add_argument("--connector-limit-per-host", type=int, default=None,
                    help=("aiohttp TCPConnector per-host connection limit. "
                          "Omit for aiohttp default; use 0 for unlimited."))
parser.add_argument("--jaeger-url", default="http://localhost:16686",
                    help="Jaeger query API URL for OTel trace fetching")
parser.add_argument("--poll-interval", type=int, default=20,
                    help="Metrics poll interval in ms for gauge time-series (0=disabled)")
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
# vLLM Prometheus metrics scraper
# ---------------------------------------------------------------------------
# All histogram metric names we want to capture from vLLM /metrics
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


def percentile_from_buckets(buckets: list[tuple[float, int]], total_count: int,
                             pct: float) -> float:
    """Compute a percentile from sorted (upper_bound, cum_count) pairs using
    linear interpolation.  Returns 0 if no data."""
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
    # target beyond last bucket — return last bound
    return buckets[-1][0] if buckets else 0.0


class VllmMetricsScraper:
    """Scrapes vLLM's /metrics Prometheus endpoint for histogram + gauge metrics."""

    def __init__(self, base_url: str):
        self.metrics_url = f"{base_url.rstrip('/')}/metrics"

    async def scrape(self, session: aiohttp.ClientSession) -> dict:
        """Fetch and parse vLLM Prometheus metrics. Returns key metrics dict."""
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
        """Parse Prometheus text format, extract histogram buckets + gauges."""
        result = {}

        # ── 1. Gauge metrics ──
        gauge_patterns = {
            "vllm_requests_running": r'vllm:num_requests_running\{[^}]*\}\s+([\d.eE+\-]+)',
            "vllm_requests_waiting": r'vllm:num_requests_waiting\{[^}]*\}\s+([\d.eE+\-]+)',
            "vllm_kv_cache_usage":   r'vllm:kv_cache_usage_perc\{[^}]*\}\s+([\d.eE+\-]+)',
            "vllm_waiting_capacity": r'vllm:num_requests_waiting_by_reason\{[^}]*reason="capacity"[^}]*\}\s+([\d.eE+\-]+)',
            "vllm_waiting_deferred": r'vllm:num_requests_waiting_by_reason\{[^}]*reason="deferred"[^}]*\}\s+([\d.eE+\-]+)',
        }
        # ── 1b. Counter metrics (cumulative, need delta) ──
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

        # ── 2. Histogram metrics: extract buckets, sum, count ──
        for prefix, metric_name in HISTOGRAM_METRICS:
            # Escape dots in metric name for regex
            escaped = re.escape(metric_name)
            # bucket lines: metric_name_bucket{...,le="BOUND"} VALUE
            bucket_re = re.compile(
                rf'^{escaped}_bucket\{{[^}}]*le="([^"]+)"\}}\s+([\d.eE+\-]+)$',
                re.MULTILINE,
            )
            sum_re = re.compile(
                rf'^{escaped}_sum\{{[^}}]*\}}\s+([\d.eE+\-]+)$',
                re.MULTILINE,
            )
            count_re = re.compile(
                rf'^{escaped}_count\{{[^}}]*\}}\s+([\d.eE+\-]+)$',
                re.MULTILINE,
            )

            buckets = []
            for m in bucket_re.finditer(text):
                le, val = m.group(1), float(m.group(2))
                if le != "+Inf":
                    buckets.append((float(le), int(val)))
                else:
                    # +Inf bucket == total_count
                    pass
            # Sort by upper bound
            buckets.sort(key=lambda x: x[0])

            m_sum = sum_re.search(text)
            m_count = count_re.search(text)

            if m_sum:
                result[f"{prefix}_sum"] = float(m_sum.group(1))
            if m_count:
                result[f"{prefix}_count"] = float(m_count.group(1))

            # Store raw buckets for delta computation
            if buckets:
                result[f"{prefix}_buckets"] = buckets

            # Compute percentiles from cumulative buckets
            total = result.get(f"{prefix}_count", 0)
            if buckets and total > 0:
                # Convert cum counts to per-bucket counts for delta computation
                # But for percentile we can use cumulative directly
                result[f"{prefix}_p50"] = percentile_from_buckets(buckets, total, 50)
                result[f"{prefix}_p95"] = percentile_from_buckets(buckets, total, 95)
                result[f"{prefix}_p99"] = percentile_from_buckets(buckets, total, 99)
            # Average from sum/count
            if f"{prefix}_sum" in result and total > 0:
                result[f"{prefix}_avg"] = result[f"{prefix}_sum"] / total

        return result


async def scrape_vllm_metrics(scraper: VllmMetricsScraper,
                               session: aiohttp.ClientSession) -> dict:
    """Wrapper to scrape metrics, returns empty dict on failure."""
    return await scraper.scrape(session)


def _subtract_buckets(b_buckets: list[tuple[float, int]],
                      a_buckets: list[tuple[float, int]]) -> list[tuple[float, int]]:
    """Given two cumulative bucket snapshots (before, after), return the delta
    buckets (per-scenario) as cumulative form for percentile computation."""
    # Build dict from bounds -> count for both
    b_map = {bound: cnt for bound, cnt in b_buckets}
    a_map = {bound: cnt for bound, cnt in a_buckets}
    all_bounds = sorted(set(b_map.keys()) | set(a_map.keys()))
    result = []
    for bound in all_bounds:
        delta = a_map.get(bound, 0) - b_map.get(bound, 0)
        result.append((bound, max(0, delta)))
    # Re-accumulate since deltas may not be cumulative
    cum = 0
    cumulative = []
    for bound, cnt in result:
        cum += cnt
        cumulative.append((bound, cum))
    return cumulative


def compute_metrics_delta(before: dict, after: dict) -> dict:
    """Compute per-scenario metrics from two cumulative metric snapshots."""
    delta = {}

    for prefix, _metric_name in HISTOGRAM_METRICS:
        # Average from sum/count
        sum_b = before.get(f"{prefix}_sum", 0)
        sum_a = after.get(f"{prefix}_sum", 0)
        cnt_b = before.get(f"{prefix}_count", 0)
        cnt_a = after.get(f"{prefix}_count", 0)
        d_sum = sum_a - sum_b
        d_cnt = cnt_a - cnt_b
        if d_cnt > 0:
            delta[f"{prefix}_avg"] = d_sum / d_cnt
            delta[f"{prefix}_count"] = d_cnt

        # Percentiles from bucket deltas
        b_bkts = before.get(f"{prefix}_buckets", [])
        a_bkts = after.get(f"{prefix}_buckets", [])
        if b_bkts and a_bkts:
            dk = _subtract_buckets(b_bkts, a_bkts)
            total = dk[-1][1] if dk else 0
            if total > 0:
                delta[f"{prefix}_p50"] = percentile_from_buckets(dk, total, 50)
                delta[f"{prefix}_p95"] = percentile_from_buckets(dk, total, 95)
                delta[f"{prefix}_p99"] = percentile_from_buckets(dk, total, 99)

    # Snapshot gauges (use 'after' values directly)
    for key in ["vllm_requests_running", "vllm_requests_waiting", "vllm_kv_cache_usage",
                "vllm_waiting_capacity", "vllm_waiting_deferred"]:
        if key in after:
            delta[key] = after[key]
    # Preemptions counter: compute delta (per-scenario count)
    if "vllm_num_preemptions_total" in after and "vllm_num_preemptions_total" in before:
        delta["vllm_num_preemptions"] = after["vllm_num_preemptions_total"] - before["vllm_num_preemptions_total"]
    elif "vllm_num_preemptions_total" in after:
        delta["vllm_num_preemptions"] = after["vllm_num_preemptions_total"]
    return delta


# ---------------------------------------------------------------------------
# Jaeger OTel trace fetcher (for tokenization + other detailed spans)
# ---------------------------------------------------------------------------
class JaegerTraceFetcher:
    """Fetches OTel spans from Jaeger to extract tokenization time etc."""

    def __init__(self, base_url: str = "http://localhost:16686"):
        self.api_url = f"{base_url.rstrip('/')}/api/traces"
        self.service_name: Optional[str] = None  # auto-detected

    async def _discover_service(self, session: aiohttp.ClientSession) -> str:
        """Find the vLLM service name in Jaeger."""
        if self.service_name:
            return self.service_name
        try:
            url = self.api_url.replace("/traces", "/services")
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                data = await resp.json()
                services = data.get("data", [])
                # 1. Prefer explicit 'vllm' name
                for s in services:
                    if "vllm" in s.lower():
                        self.service_name = s
                        return s
                # 2. Prefer 'unknown_service' (vLLM default when no service name set)
                for s in services:
                    if s == "unknown_service":
                        self.service_name = s
                        return s
                # 3. Any non-jaeger service
                for s in services:
                    if "jaeger" not in s.lower():
                        self.service_name = s
                        return s
                self.service_name = services[0] if services else "unknown_service"
                return self.service_name
        except Exception:
            self.service_name = "unknown_service"
            return self.service_name

    async def fetch_spans(self, session: aiohttp.ClientSession,
                          start_us: int, end_us: int,
                          limit: int = 20000) -> list[dict]:
        """Fetch traces from Jaeger for the given time window. Returns flat span list."""
        service = await self._discover_service(session)
        try:
            params = {
                "service": service,
                "start": str(start_us),
                "end": str(end_us),
                "limit": str(limit),
            }
            async with session.get(
                self.api_url,
                params=params,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
                traces = data.get("data", [])
                spans = []
                for trace in traces:
                    for span in trace.get("spans", []):
                        spans.append(span)
                return spans
        except Exception:
            return []

    @staticmethod
    def _is_llm_request_span(span: dict) -> bool:
        return "request" in span.get("operationName", "").lower()

    @classmethod
    def filter_llm_request_spans(
        cls,
        spans: list[dict],
        start_us: int,
        end_us: int,
    ) -> list[dict]:
        result = []
        for span in spans:
            if not cls._is_llm_request_span(span):
                continue
            span_start = int(span.get("startTime", 0) or 0)
            if start_us <= span_start <= end_us:
                result.append(span)
        return result

    @staticmethod
    def extract_span_durations(spans: list[dict], span_name_contains: str) -> list[float]:
        """Extract durations (in seconds) for spans whose operationName matches."""
        durations = []
        for span in spans:
            op = span.get("operationName", "")
            if span_name_contains.lower() in op.lower():
                dur_us = span.get("duration", 0)
                if dur_us > 0:
                    durations.append(dur_us / 1e6)  # microseconds -> seconds
        return durations

    @staticmethod
    def extract_gen_ai_latencies(spans: list[dict]) -> dict[str, list[float]]:
        """Extract gen_ai.latency.* tag values from llm_request spans.
        Returns dict of metric_name -> list of values (in seconds)."""
        result: dict[str, list[float]] = {}
        for span in spans:
            op = span.get("operationName", "")
            if "request" not in op.lower():
                continue
            tags = {t["key"]: t.get("value", 0) for t in span.get("tags", [])}
            dur_us = span.get("duration", 0)
            if dur_us <= 0:
                continue

            # Extract known gen_ai.latency tags
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

            # Tokenization overhead = total span - accounted latency phases
            tok_overhead = span_total - accounted
            if tok_overhead > 0:
                result.setdefault("tokenization_overhead", []).append(tok_overhead)

            # Also store raw span duration as e2e from server perspective
            result.setdefault("e2e", []).append(span_total)

        return result

    @staticmethod
    def extract_llm_request_timeline(
        spans: list[dict],
        scenario_start_us: int,
    ) -> dict[str, dict]:
        """Summarize server-side llm_request span timestamps.

        All returned values are in seconds. Offsets are relative to the scenario
        measurement start so they can be compared across concurrency levels.
        """
        starts = []
        ends = []
        durations = []
        first_token_offsets = []
        inference_end_offsets = []

        for span in spans:
            start_us = int(span.get("startTime", 0) or 0)
            duration_us = int(span.get("duration", 0) or 0)
            if start_us <= 0 or duration_us <= 0:
                continue

            start_s = (start_us - scenario_start_us) / 1e6
            duration_s = duration_us / 1e6
            end_s = start_s + duration_s
            starts.append(start_s)
            ends.append(end_s)
            durations.append(duration_s)

            tags = {t["key"]: t.get("value", 0) for t in span.get("tags", [])}
            ttft = tags.get("gen_ai.latency.time_to_first_token", 0)
            inference = tags.get("gen_ai.latency.time_in_model_inference", 0)
            try:
                ttft = float(ttft)
            except (TypeError, ValueError):
                ttft = 0
            try:
                inference = float(inference)
            except (TypeError, ValueError):
                inference = 0
            if ttft > 0:
                first_token_offsets.append(start_s + ttft)
            if inference > 0:
                inference_end_offsets.append(start_s + inference)

        return {
            "server_span_start_offset": JaegerTraceFetcher.compute_percentiles(starts),
            "server_span_first_token_offset": JaegerTraceFetcher.compute_percentiles(first_token_offsets),
            "server_span_inference_end_offset": JaegerTraceFetcher.compute_percentiles(inference_end_offsets),
            "server_span_end_offset": JaegerTraceFetcher.compute_percentiles(ends),
            "server_span_duration": JaegerTraceFetcher.compute_percentiles(durations),
        }

    @staticmethod
    def compute_percentiles(values: list[float]) -> dict:
        """Compute P50/P95/P99/mean/min/max from a list of values."""
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
            "count": n,
            "p50": _pct(50),
            "p95": _pct(95),
            "p99": _pct(99),
            "mean": sum(values) / n,
            "min": values[0],
            "max": values[-1],
        }


# ---------------------------------------------------------------------------
# Continuous metrics poller (gauge time-series)
# ---------------------------------------------------------------------------
class MetricsPoller:
    """Continuously polls vLLM /metrics at a fixed interval to capture
    running/waiting/kv_cache gauge values as a time-series."""

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
                sample = {"t": round(t * 1000, 1)}  # ms since start
                # Extract gauges with simple regex
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
        """Render an ASCII time-series chart for a given gauge key."""
        if not samples:
            return f"  (no data for {key})"
        values = [s.get(key, 0) for s in samples]
        times = [s.get("t", 0) for s in samples]
        max_val = max(values) if values else 1
        if max_val == 0:
            max_val = 1
        t_min, t_max = times[0], times[-1]
        t_range = t_max - t_min if t_max > t_min else 1

        # Bin into `width` columns
        bins: list[list[float]] = [[] for _ in range(width)]
        for t, v in zip(times, values):
            idx = int((t - t_min) / t_range * (width - 1))
            idx = max(0, min(width - 1, idx))
            bins[idx].append(v)

        # Average per bin
        bin_vals = []
        for b in bins:
            bin_vals.append(sum(b) / len(b) if b else 0)

        # Render rows
        lines = []
        if title:
            lines.append(f"  {title}")
        lines.append(f"  {key} (max={max_val:.0f})")
        for row in range(height, -1, -1):
            threshold = max_val * row / height
            line = "  │"
            for bv in bin_vals:
                line += "█" if bv >= threshold else " "
            line += f" {threshold:>6.0f}"
            lines.append(line)
        lines.append("  └" + "─" * width)
        # Time axis labels
        lines.append(f"  {t_min:>8.0f}ms{' ' * (width - 14)}{t_max:>8.0f}ms")
        return "\n".join(lines)


def _get_http_trace(ctx) -> Optional[dict]:
    trace_request_ctx = getattr(ctx, "trace_request_ctx", None)
    if isinstance(trace_request_ctx, dict):
        return trace_request_ctx.get("http_trace")
    return None


def build_http_trace_config() -> aiohttp.TraceConfig:
    """Collect aiohttp client lifecycle timings for benchmark requests."""
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
# Single request profiler
# ---------------------------------------------------------------------------
@dataclass
class RequestTrace:
    concurrency_level: int = 0
    batch: int = 0
    idx: int = 0
    input_tokens: int = 0
    t_serialize_ms: float = 0.0
    t_sem_wait_ms: float = 0.0       # NEW: semaphore (client concurrency gate)
    t_http_connect_ms: float = 0.0   # NEW: POST → first HTTP byte (server queue + net)
    t_http_conn_queued_ms: float = 0.0
    t_http_dns_ms: float = 0.0
    t_http_tcp_connect_ms: float = 0.0
    t_http_request_send_ms: float = 0.0
    t_http_response_headers_wait_ms: float = 0.0
    t_response_header_to_first_sse_byte_ms: float = 0.0
    t_first_sse_byte_to_first_token_ms: float = 0.0
    t_first_byte_ms: float = 0.0     # Kept for backward compat = sem_wait + http_connect
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

    # Phase 1: Semaphore wait (client-side concurrency gate)
    t_pre_sem = time.perf_counter()
    async with semaphore:
        t_post_sem = time.perf_counter()
        t.t_sem_wait_ms = (t_post_sem - t_pre_sem) * 1000

        # Phase 2: HTTP request (includes server queuing)
        t_post = time.perf_counter()
        http_trace = {"request_start": t_post}
        try:
            t_first_byte = None

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
                # Backward compat: first_byte = sem_wait + http_connect
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
    print(f"[03_driver] Connector:    limit={args.connector_limit if args.connector_limit is not None else '<aiohttp default>'}, "
          f"limit_per_host={args.connector_limit_per_host if args.connector_limit_per_host is not None else '<aiohttp default>'}")
    print()

    scraper = VllmMetricsScraper(url)
    jaeger = JaegerTraceFetcher(args.jaeger_url)
    all_traces: list[RequestTrace] = []
    scenario_summaries: dict[str, dict] = {}
    scenario_vllm_metrics: dict[str, dict] = {}

    connector = None
    if args.connector_limit is not None or args.connector_limit_per_host is not None:
        connector_kwargs = {}
        if args.connector_limit is not None:
            connector_kwargs["limit"] = args.connector_limit
        if args.connector_limit_per_host is not None:
            connector_kwargs["limit_per_host"] = args.connector_limit_per_host
        connector = aiohttp.TCPConnector(**connector_kwargs)
    else:
        # Default: unlimited connections to avoid aiohttp's 100-connection cap
        # bottlenecking high-concurrency sweeps (256+).
        connector = aiohttp.TCPConnector(limit=0, limit_per_host=0)

    async with aiohttp.ClientSession(
        connector=connector,
        trace_configs=[build_http_trace_config()],
    ) as session:
        # Scrape baseline metrics before benchmark
        baseline_metrics = await scrape_vllm_metrics(scraper, session)
        if baseline_metrics:
            print(f"[03_driver] vLLM metrics endpoint detected — "
                  f"server-side queue/prefill/decode times will be recorded")
        else:
            print(f"[03_driver] vLLM /metrics not available — "
                  f"using client-side timing only")

        # Check Jaeger availability
        jaeger_ok = False
        try:
            svc = await jaeger._discover_service(session)
            print(f"[03_driver] Jaeger detected — OTel spans will be fetched "
                  f"(service={svc})")
            jaeger_ok = True
        except Exception:
            print(f"[03_driver] Jaeger not available at {args.jaeger_url} — "
                  f"skipping OTel trace fetching")
        print()

        for concurrency in args.scenarios:
            print(f"{'─'*70}")
            print(f"  Concurrency Level: {concurrency}")
            print(f"{'─'*70}")

            # Warmup
            for w in range(args.warmup_batches):
                await run_batch(
                    session, url, args.model,
                    concurrency, -1,  # warmup batch marker
                    args.input_tokens, args.output_tokens,
                    args.request_timeout,
                )

            # Measurement — with continuous gauge polling
            scenario_traces: list[RequestTrace] = []
            poller = MetricsPoller(url, args.poll_interval) if args.poll_interval > 0 else None
            scenario_start_us = int(time.time() * 1e6)
            if poller:
                poller.start()
            for b in range(args.num_batches):
                batch_results = await run_batch(
                    session, url, args.model,
                    concurrency, b,
                    args.input_tokens, args.output_tokens,
                    args.request_timeout,
                )
                scenario_traces.extend(batch_results)
            poll_samples = []
            if poller:
                poll_samples = await poller.stop()
            scenario_end_us = int(time.time() * 1e6)

            # Scrape vLLM metrics after this concurrency level
            post_metrics = await scrape_vllm_metrics(scraper, session)
            # Delta: compute per-request averages for this scenario
            if post_metrics and baseline_metrics:
                vllm_delta = compute_metrics_delta(baseline_metrics, post_metrics)
            elif post_metrics:
                # No baseline — use raw snapshot gauges only
                vllm_delta = {}
                for key in ["vllm_requests_running", "vllm_requests_waiting",
                            "vllm_kv_cache_usage"]:
                    if key in post_metrics:
                        vllm_delta[key] = post_metrics[key]
            else:
                vllm_delta = {}

            # Fetch OTel spans from Jaeger for this scenario
            if jaeger_ok:
                # Fetch with a small buffer, then filter by the exact measurement
                # window to avoid mixing warmup or neighboring concurrency levels.
                spans = await jaeger.fetch_spans(
                    session,
                    scenario_start_us - int(5 * 1e6),
                    scenario_end_us + int(5 * 1e6),
                )
                if spans:
                    spans = jaeger.filter_llm_request_spans(
                        spans, scenario_start_us, scenario_end_us
                    )
                    # Extract gen_ai.latency tags from llm_request spans
                    latencies = jaeger.extract_gen_ai_latencies(spans)
                    for tag_name, values in latencies.items():
                        # Sanitize key: gen_ai.latency.time_in_queue -> otel_time_in_queue
                        key = "otel_" + tag_name.replace("gen_ai.latency.", "").replace(".", "_")
                        vllm_delta[key] = jaeger.compute_percentiles(values)
                    vllm_delta.update(
                        jaeger.extract_llm_request_timeline(spans, scenario_start_us)
                    )
                    vllm_delta["otel_llm_request_span_count"] = len(spans)
                    vllm_delta["scenario_start_unix_us"] = scenario_start_us
                    vllm_delta["scenario_end_unix_us"] = scenario_end_us

                    # Collect unique span operation names for discovery
                    span_ops = set()
                    for s in spans:
                        op = s.get("operationName", "")
                        if op:
                            span_ops.add(op)
                    vllm_delta["otel_span_operations"] = sorted(span_ops)

            scenario_vllm_metrics[str(concurrency)] = vllm_delta
            baseline_metrics = post_metrics  # Update baseline for next scenario

            all_traces.extend(scenario_traces)

            # Per-scenario summary
            successful = [r for r in scenario_traces if r.success]
            if successful:
                summary = {
                    "concurrency": concurrency,
                    "num_requests": len(scenario_traces),
                    "num_successful": len(successful),
                    "t_serialize_ms": compute_stats([r.t_serialize_ms for r in successful]),
                    "t_sem_wait_ms": compute_stats([r.t_sem_wait_ms for r in successful]),
                    "t_http_connect_ms": compute_stats([r.t_http_connect_ms for r in successful]),
                    "t_http_conn_queued_ms": compute_stats([r.t_http_conn_queued_ms for r in successful]),
                    "t_http_dns_ms": compute_stats([r.t_http_dns_ms for r in successful]),
                    "t_http_tcp_connect_ms": compute_stats([r.t_http_tcp_connect_ms for r in successful]),
                    "t_http_request_send_ms": compute_stats([r.t_http_request_send_ms for r in successful]),
                    "t_http_response_headers_wait_ms": compute_stats([r.t_http_response_headers_wait_ms for r in successful]),
                    "t_response_header_to_first_sse_byte_ms": compute_stats([r.t_response_header_to_first_sse_byte_ms for r in successful]),
                    "t_first_sse_byte_to_first_token_ms": compute_stats([r.t_first_sse_byte_to_first_token_ms for r in successful]),
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

            # Store poll time-series
            vllm_delta["poll_samples"] = poll_samples

            # Render ASCII waveform for this scenario
            if poll_samples:
                print(MetricsPoller.render_ascii(
                    poll_samples, "running", width=70, height=10,
                    title=f"  [Conc={concurrency}] num_requests_running"))
                print(MetricsPoller.render_ascii(
                    poll_samples, "waiting", width=70, height=10,
                    title=f"  [Conc={concurrency}] num_requests_waiting"))
                print()

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
            "t_sem_wait_ms": round(t.t_sem_wait_ms, 20),
            "t_http_connect_ms": round(t.t_http_connect_ms, 20),
            "t_http_conn_queued_ms": round(t.t_http_conn_queued_ms, 20),
            "t_http_dns_ms": round(t.t_http_dns_ms, 20),
            "t_http_tcp_connect_ms": round(t.t_http_tcp_connect_ms, 20),
            "t_http_request_send_ms": round(t.t_http_request_send_ms, 20),
            "t_http_response_headers_wait_ms": round(t.t_http_response_headers_wait_ms, 20),
            "t_response_header_to_first_sse_byte_ms": round(t.t_response_header_to_first_sse_byte_ms, 20),
            "t_first_sse_byte_to_first_token_ms": round(t.t_first_sse_byte_to_first_token_ms, 20),
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
        "connector_limit": args.connector_limit,
        "connector_limit_per_host": args.connector_limit_per_host,
        "scenarios": args.scenarios,
        "summaries": scenario_summaries,
        "vllm_metrics": scenario_vllm_metrics,
        "raw_traces": raw_traces,
    }

    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\n[03_driver] Results saved to {out_path}")

    # -------------------------------------------------------------------
    # Summary table
    # -------------------------------------------------------------------
    print("\n" + "=" * 150)
    print("CONCURRENCY SWEEP — Latency Breakdown (P95, ms)")
    print("=" * 150)
    header = (f"{'Conc':>6s} {'Req':>6s} {'OK':>5s} "
              f"{'Serialize':>9s} {'SemWait':>9s} {'HTTP':>9s} {'1stByte':>9s} "
              f"{'TTFT':>9s} {'Prefill':>9s} {'Decode':>9s} {'E2E':>9s}")
    print(header)
    print("-" * 150)

    for concurrency in args.scenarios:
        s = scenario_summaries.get(str(concurrency), {})
        if "error" in s:
            print(f"{concurrency:>6d} {s['num_requests']:>6d} {'FAIL':>5s}")
        else:
            n = s["num_successful"]
            print(f"{concurrency:>6d} {s['num_requests']:>6d} {n:>5d} "
                  f"{s['t_serialize_ms']['p95']:>7.2f}ms "
                  f"{s['t_sem_wait_ms']['p95']:>7.2f}ms "
                  f"{s['t_http_connect_ms']['p95']:>7.2f}ms "
                  f"{s['t_first_byte_ms']['p95']:>7.2f}ms "
                  f"{s['t_server_prefill_ms']['p95']:>7.2f}ms "
                  f"{s['t_prefill_ms']['p95']:>7.2f}ms "
                  f"{s['t_decode_ms']['p95']:>7.2f}ms "
                  f"{s['t_e2e_ms']['p95']:>7.2f}ms")
    print("=" * 150)

    # vLLM server-side metrics table — Averages + gauges
    has_server_metrics = any(
        v for v in scenario_vllm_metrics.values() if v
    )
    if has_server_metrics:
        print("\n" + "=" * 110)
        print("vLLM SERVER-SIDE METRICS — Averages (from /metrics endpoint)")
        print("=" * 120)
        header2 = (f"{'Conc':>6s} {'QueueAvg':>10s} {'PrefillAvg':>10s} "
                   f"{'DecodeAvg':>10s} {'TTFTEvg':>10s} {'E2EAvg':>10s} "
                   f"{'ITLAvg':>10s} {'Preempt':>8s} {'W.Cap':>6s} {'W.Def':>6s} "
                   f"{'Run':>5s} {'Wait':>5s} {'KV%':>5s}")
        print(header2)
        print("-" * 120)
        for concurrency in args.scenarios:
            vm = scenario_vllm_metrics.get(str(concurrency), {})
            if not vm:
                print(f"{concurrency:>6d} {'N/A':>10s}")
                continue
            q = vm.get("vllm_queue_time_avg", 0) * 1000
            p = vm.get("vllm_prefill_time_avg", 0) * 1000
            d = vm.get("vllm_decode_time_avg", 0) * 1000
            t = vm.get("vllm_ttft_avg", 0) * 1000
            e = vm.get("vllm_e2e_avg", 0) * 1000
            itl = vm.get("vllm_itl_avg", 0) * 1000
            pr = vm.get("vllm_num_preemptions", 0)
            wc = vm.get("vllm_waiting_capacity", 0)
            wd = vm.get("vllm_waiting_deferred", 0)
            r = vm.get("vllm_requests_running", 0)
            w = vm.get("vllm_requests_waiting", 0)
            kv = vm.get("vllm_kv_cache_usage", 0) * 100
            print(f"{concurrency:>6d} {q:>8.1f}ms {p:>8.1f}ms "
                  f"{d:>8.1f}ms {t:>8.1f}ms {e:>8.1f}ms {itl:>8.1f}ms "
                  f"{pr:>8.0f} {wc:>6.0f} {wd:>6.0f} "
                  f"{r:>5.0f} {w:>5.0f} {kv:>5.1f}%")
        print("=" * 120)

        # ── Percentile tables for key histograms ──
        pct_metrics = [
            ("Queue Time",       "vllm_queue_time"),
            ("Prefill Time",     "vllm_prefill_time"),
            ("Decode Time",      "vllm_decode_time"),
            ("TTFT",             "vllm_ttft"),
            ("E2E Latency",      "vllm_e2e"),
            ("Inference Time",   "vllm_inference_time"),
            ("Inter-Token Lat",  "vllm_itl"),
            ("Time/Out Token",   "vllm_time_per_out_tok"),
            ("HTTP Duration",    "http_request_duration"),
        ]
        for title, prefix in pct_metrics:
            # Check if any scenario has data for this metric
            has_data = any(
                f"{prefix}_p50" in scenario_vllm_metrics.get(str(c), {})
                for c in args.scenarios
            )
            if not has_data:
                continue
            print(f"\n  {title} (ms)")
            print(f"  {'Conc':>6s} {'Count':>8s} {'P50':>10s} {'P95':>10s} {'P99':>10s} {'Avg':>10s}")
            print(f"  {'-'*56}")
            for concurrency in args.scenarios:
                vm = scenario_vllm_metrics.get(str(concurrency), {})
                cnt = vm.get(f"{prefix}_count", 0)
                p50 = vm.get(f"{prefix}_p50", 0) * 1000
                p95 = vm.get(f"{prefix}_p95", 0) * 1000
                p99 = vm.get(f"{prefix}_p99", 0) * 1000
                avg = vm.get(f"{prefix}_avg", 0) * 1000
                print(f"  {concurrency:>6d} {cnt:>8.0f} {p50:>8.2f}ms {p95:>8.2f}ms "
                      f"{p99:>8.2f}ms {avg:>8.2f}ms")

    # ── OTel gen_ai.latency detailed breakdown ──
    otel_prefixes = [
        "otel_time_in_queue",
        "otel_time_in_model_prefill",
        "otel_time_in_model_decode",
        "otel_time_in_model_inference",
        "otel_time_to_first_token",
        "otel_tokenization_overhead",
        "otel_e2e",
        "otel_otel_e2e",  # legacy key from v3 runs
    ]
    has_otel = any(
        any(scenario_vllm_metrics.get(str(c), {}).get(p, {}).get("count", 0) > 0
            for p in otel_prefixes)
        for c in args.scenarios
    )
    if has_otel:
        print(f"\n{'='*100}")
        print("OTel DETAILED LATENCY BREAKDOWN (from Jaeger llm_request spans)")
        print("="*100)
        otel_labels = [
            ("Tokenization Overhead",  "otel_tokenization_overhead"),
            ("Time in Queue",          "otel_time_in_queue"),
            ("Model Prefill",          "otel_time_in_model_prefill"),
            ("Model Decode",           "otel_time_in_model_decode"),
            ("Model Inference",        "otel_time_in_model_inference"),
            ("Time to First Token",    "otel_time_to_first_token"),
            ("Server E2E (span)",      "otel_e2e"),
        ]
        for title, prefix in otel_labels:
            has_data = any(
                scenario_vllm_metrics.get(str(c), {}).get(prefix, {}).get("count", 0) > 0
                for c in args.scenarios
            )
            if not has_data:
                continue
            print(f"\n  {title} (ms)")
            print(f"  {'Conc':>6s} {'Count':>8s} {'P50':>12s} {'P95':>12s} "
                  f"{'P99':>12s} {'Avg':>12s} {'Min':>12s} {'Max':>12s}")
            print(f"  {'-'*82}")
            for concurrency in args.scenarios:
                vm = scenario_vllm_metrics.get(str(concurrency), {})
                stats = vm.get(prefix, {})
                if not stats or stats.get("count", 0) == 0:
                    print(f"  {concurrency:>6d} {'—':>8s}")
                    continue
                cnt = stats["count"]
                p50 = stats.get("p50", 0) * 1000
                p95 = stats.get("p95", 0) * 1000
                p99 = stats.get("p99", 0) * 1000
                avg = stats.get("mean", 0) * 1000
                mn = stats.get("min", 0) * 1000
                mx = stats.get("max", 0) * 1000
                print(f"  {concurrency:>6d} {cnt:>8.0f} {p50:>10.3f}ms {p95:>10.3f}ms "
                      f"{p99:>10.3f}ms {avg:>10.3f}ms {mn:>10.3f}ms {mx:>10.3f}ms")

        # Print discovered span operations once
        first_with_ops = next(
            (scenario_vllm_metrics.get(str(c), {}).get("otel_span_operations", [])
             for c in args.scenarios
             if scenario_vllm_metrics.get(str(c), {}).get("otel_span_operations")),
            [],
        )
        if first_with_ops:
            print(f"\n  Discovered OTel span operations: {', '.join(first_with_ops[:30])}")
        print("="*100)


if __name__ == "__main__":
    asyncio.run(main())
