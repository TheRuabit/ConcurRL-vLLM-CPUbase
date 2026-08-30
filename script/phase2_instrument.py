#!/usr/bin/env python3
"""
Phase 2 Instrumentation — Monkey-patch Module
==============================================
Injects per-request timing, vLLM /metrics scraping, and Jaeger OTel trace
fetching into veRL's GRPO training loop without modifying the veRL package.

Import this module BEFORE running veRL:
    import script.phase2_instrument as inst
    # ... run veRL training ...
    data = inst.get_instrumentation_data()

Patched entry points:
  - LLMServerClient.generate()  → per-request latency
  - LLMServerManager._initialize_llm_servers()  → server address discovery
  - AgentLoopManager.generate_sequences()  → step-level rollout timing
"""

import asyncio
import json
import logging
import re
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import aiohttp

logger = logging.getLogger("phase2_instrument")

# ---------------------------------------------------------------------------
# Global state — collected data lives here
# ---------------------------------------------------------------------------
_step_counter = 0
_rollout_steps: list[dict] = []       # per-step rollout summaries
_request_records: list[dict] = []     # per-request timing records
_server_addresses: list[str] = []     # vLLM server HTTP addresses (host:port)
_t_first_request: float = 0.0        # absolute time of first generate() call
_t_origin: float = 0.0               # time origin (set on first patch apply)

# Prometheus gauge time-series during rollout
_poll_samples: list[dict] = []
_poll_task: Optional[asyncio.Task] = None
_poll_stop_event: Optional[asyncio.Event] = None

# Jaeger data (optional)
_jaeger_data: dict = {}

# Lock for thread-safe list appends
_lock = threading.Lock()


def _safe_append(lst: list, item):
    with _lock:
        lst.append(item)


# ---------------------------------------------------------------------------
# Prometheus /metrics scraper (reused from Phase 1)
# ---------------------------------------------------------------------------
HISTOGRAM_METRICS = [
    ("vllm_queue_time", "vllm:request_queue_time_seconds"),
    ("vllm_prefill_time", "vllm:request_prefill_time_seconds"),
    ("vllm_decode_time", "vllm:request_decode_time_seconds"),
    ("vllm_ttft", "vllm:time_to_first_token_seconds"),
    ("vllm_e2e", "vllm:e2e_request_latency_seconds"),
    ("vllm_inference_time", "vllm:request_inference_time_seconds"),
    ("vllm_itl", "vllm:inter_token_latency_seconds"),
    ("vllm_time_per_out_tok", "vllm:request_time_per_output_token_seconds"),
    ("vllm_kv_computed_tokens", "vllm:request_prefill_kv_computed_tokens"),
]


def _percentile_from_buckets(buckets: list[tuple[float, int]], total_count: int,
                              pct: float) -> float:
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


def parse_prometheus_metrics(text: str) -> dict:
    """Parse Prometheus text format — gauges, counters, histograms."""
    result = {}

    # Gauges
    gauge_patterns = {
        "vllm_requests_running": r'vllm:num_requests_running\{[^}]*\}\s+([\d.eE+\-]+)',
        "vllm_requests_waiting": r'vllm:num_requests_waiting\{[^}]*\}\s+([\d.eE+\-]+)',
        "vllm_kv_cache_usage": r'vllm:kv_cache_usage_perc\{[^}]*\}\s+([\d.eE+\-]+)',
    }
    for key, pattern in gauge_patterns.items():
        m = re.search(pattern, text)
        if m:
            result[key] = float(m.group(1))

    # Counters
    counter_patterns = {
        "vllm_num_preemptions_total": r'vllm:num_preemptions_total\{[^}]*\}\s+([\d.eE+\-]+)',
        "vllm_prefix_cache_queries": r'vllm:prefix_cache_queries\{[^}]*\}\s+([\d.eE+\-]+)',
        "vllm_prefix_cache_hits": r'vllm:prefix_cache_hits\{[^}]*\}\s+([\d.eE+\-]+)',
    }
    for key, pattern in counter_patterns.items():
        m = re.search(pattern, text)
        if m:
            result[key] = float(m.group(1))

    # Histograms
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
            result[f"{prefix}_p50"] = _percentile_from_buckets(buckets, total, 50)
            result[f"{prefix}_p95"] = _percentile_from_buckets(buckets, total, 95)
            result[f"{prefix}_p99"] = _percentile_from_buckets(buckets, total, 99)
        if f"{prefix}_sum" in result and total > 0:
            result[f"{prefix}_avg"] = result[f"{prefix}_sum"] / total

    return result


def _subtract_buckets(a_buckets: list[tuple[float, int]],
                       b_buckets: list[tuple[float, int]]) -> list[tuple[float, int]]:
    """Compute histogram delta (after - before) as cumulative form."""
    a_map = {bound: cnt for bound, cnt in a_buckets}
    b_map = {bound: cnt for bound, cnt in b_buckets}
    all_bounds = sorted(set(a_map.keys()) | set(b_map.keys()))
    deltas = []
    for bound in all_bounds:
        delta = a_map.get(bound, 0) - b_map.get(bound, 0)
        deltas.append((bound, max(0, delta)))
    cum = 0
    cumulative = []
    for bound, cnt in deltas:
        cum += cnt
        cumulative.append((bound, cum))
    return cumulative


def compute_metrics_delta(before: dict, after: dict) -> dict:
    """Compute per-scenario metrics delta from two cumulative snapshots."""
    delta = {}
    for prefix, _ in HISTOGRAM_METRICS:
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
            dk = _subtract_buckets(a_bkts, b_bkts)
            total = dk[-1][1] if dk else 0
            if total > 0:
                delta[f"{prefix}_p50"] = _percentile_from_buckets(dk, total, 50)
                delta[f"{prefix}_p95"] = _percentile_from_buckets(dk, total, 95)
                delta[f"{prefix}_p99"] = _percentile_from_buckets(dk, total, 99)

    # Prefix cache hit rate from counter deltas
    queries_a = after.get("vllm_prefix_cache_queries", 0)
    queries_b = before.get("vllm_prefix_cache_queries", 0)
    hits_a = after.get("vllm_prefix_cache_hits", 0)
    hits_b = before.get("vllm_prefix_cache_hits", 0)
    d_queries = queries_a - queries_b
    d_hits = hits_a - hits_b
    if d_queries > 0:
        delta["vllm_prefix_cache_hit_rate"] = d_hits / d_queries
        delta["vllm_prefix_cache_queries"] = d_queries
        delta["vllm_prefix_cache_hits"] = d_hits

    # Gauges (use 'after' directly)
    for key in ["vllm_requests_running", "vllm_requests_waiting", "vllm_kv_cache_usage"]:
        if key in after:
            delta[key] = after[key]

    # Preemptions delta
    if "vllm_num_preemptions_total" in after:
        delta["vllm_num_preemptions"] = (
            after["vllm_num_preemptions_total"] - before.get("vllm_num_preemptions_total", 0)
        )

    return delta


async def scrape_vllm_metrics(base_url: str, session: aiohttp.ClientSession) -> dict:
    """Fetch and parse vLLM /metrics endpoint."""
    url = f"http://{base_url}/metrics"
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
            if resp.status != 200:
                return {}
            text = await resp.text()
            return parse_prometheus_metrics(text)
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Metrics poller (background thread during rollout)
# ---------------------------------------------------------------------------
class _MetricsPoller:
    """Polls vLLM /metrics at a fixed interval during rollout steps."""

    def __init__(self, server_addresses: list[str], interval_ms: int = 20):
        self.server_addresses = server_addresses
        self.interval = interval_ms / 1000.0
        self._task: Optional[asyncio.Task] = None
        self._samples: list[dict] = []
        self._t0: float = 0

    async def _poll_loop(self):
        import urllib.request
        proxy_handler = urllib.request.ProxyHandler({})
        opener = urllib.request.build_opener(proxy_handler)

        while True:
            t = time.perf_counter() - self._t0
            sample = {"t": round(t * 1000, 1)}
            for addr in self.server_addresses:
                try:
                    req = urllib.request.Request(f"http://{addr}/metrics")
                    resp = opener.open(req, timeout=2)
                    text = resp.read().decode()
                    for key, pattern in [
                        ("running", r'vllm:num_requests_running\{[^}]*\}\s+([\d.eE+\-]+)'),
                        ("waiting", r'vllm:num_requests_waiting\{[^}]*\}\s+([\d.eE+\-]+)'),
                        ("kv_cache", r'vllm:kv_cache_usage_perc\{[^}]*\}\s+([\d.eE+\-]+)'),
                        ("preemptions", r'vllm:num_preemptions_total\{[^}]*\}\s+([\d.eE+\-]+)'),
                    ]:
                        m = re.search(pattern, text)
                        if m:
                            sample[f"{key}_{addr}"] = float(m.group(1))
                except Exception:
                    pass
            self._samples.append(sample)
            await asyncio.sleep(self.interval)

    def start(self):
        self._samples = []
        self._t0 = time.perf_counter()
        try:
            self._task = asyncio.ensure_future(self._poll_loop())
        except RuntimeError:
            pass

    async def stop(self) -> list[dict]:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None
        return self._samples


# ---------------------------------------------------------------------------
# Jaeger OTel trace fetcher (reused from Phase 1)
# ---------------------------------------------------------------------------
class _JaegerTraceFetcher:
    """Fetches OTel spans from Jaeger for a given time window."""

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

    async def fetch_spans(self, session: aiohttp.ClientSession,
                          start_us: int, end_us: int,
                          limit: int = 20000) -> list[dict]:
        service = await self._discover_service(session)
        try:
            params = {
                "service": service,
                "start": str(start_us),
                "end": str(end_us),
                "limit": str(limit),
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
    def extract_gen_ai_latencies(spans: list[dict]) -> dict[str, list[float]]:
        result: dict[str, list[float]] = {}
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
    def compute_percentiles(values: list[float]) -> dict:
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
# Monkey-patch: LLMServerManager._initialize_llm_servers()
# ---------------------------------------------------------------------------
_external_vllm_url: str | None = None  # Set to skip vLLM launch

def set_external_vllm_url(url: str | None):
    """Set an external vLLM server URL. When set, _initialize_llm_servers
    skips launching vLLM and connects to the existing server."""
    global _external_vllm_url
    _external_vllm_url = url

def _patch_server_manager():
    """Patch LLMServerManager to capture vLLM server addresses."""
    from verl.workers.rollout.llm_server import LLMServerManager

    orig_init = LLMServerManager._initialize_llm_servers

    async def _patched_init(self, *args, **kwargs):
        global _server_addresses
        if _external_vllm_url:
            # Skip vLLM launch; connect to external server
            url = _external_vllm_url.rstrip("/")
            if url.startswith("http://"):
                url = url[7:]
            elif url.startswith("https://"):
                url = url[8:]
            self.server_addresses = [url]
            self.rollout_replicas = []
            self.server_handles = []
            _server_addresses = [url]
            logger.info(f"[phase2_instrument] Using external vLLM server: {url}")
            return
        result = await orig_init(self, *args, **kwargs)
        _server_addresses = list(self.server_addresses)
        logger.info(f"[phase2_instrument] Captured vLLM server addresses: {_server_addresses}")
        return result

    LLMServerManager._initialize_llm_servers = _patched_init

    # Also patch _init_global_load_balancer for external server
    orig_init_lb = LLMServerManager._init_global_load_balancer

    async def _patched_init_lb(self, *args, **kwargs):
        if _external_vllm_url:
            # Skip load balancer init for external server — create a minimal one
            from verl.workers.rollout.llm_server import GlobalRequestLoadBalancer
            self.global_load_balancer = GlobalRequestLoadBalancer.remote(
                servers={}, max_cache_size=1024,
            )
            logger.info("[phase2_instrument] Skipped load balancer init (external server)")
            return
        return await orig_init_lb(self, *args, **kwargs)

    LLMServerManager._init_global_load_balancer = _patched_init_lb


# ---------------------------------------------------------------------------
# Monkey-patch: vLLMRollout._ensure_server_handle()
# ---------------------------------------------------------------------------
def _patch_vllm_rollout():
    """Patch vLLMRollout (ServerAdapter) to skip Ray actor lookup when using external server."""
    from verl.workers.rollout.vllm_rollout.vllm_rollout import ServerAdapter

    orig_ensure = ServerAdapter._ensure_server_handle

    def _patched_ensure(self) -> bool:
        if _external_vllm_url:
            return False  # Skip Ray actor lookup for external server
        return orig_ensure(self)

    ServerAdapter._ensure_server_handle = _patched_ensure

    # Patch update_weights to skip server_handle calls for external server
    orig_update = ServerAdapter.update_weights

    async def _patched_update_weights(self, weights, global_steps=None, **kwargs):
        if _external_vllm_url:
            import time as _time
            start_time = _time.time()
            future = await self._execute_method(
                "update_weights_from_ipc",
                non_block=True,
                kwargs={**kwargs, "use_shm": self.use_shm},
            )
            from verl.workers.rollout.vllm_rollout.utils import BucketedWeightSender
            bucket_size_mb = self.config.checkpoint_engine.update_weights_bucket_megabytes
            sender = BucketedWeightSender(
                zmq_handle=self.zmq_handle,
                bucket_size_mb=bucket_size_mb,
                use_shm=self.use_shm,
            )
            await sender.async_send_weights(weights)
            if future is not None:
                await future
            if self.replica_rank == 0 and self.rollout_rank == 0:
                logger.info(f"update_weights done (external), time cost: {_time.time() - start_time:.2f}s")
            return
        return await orig_update(self, weights, global_steps=global_steps, **kwargs)

    ServerAdapter.update_weights = _patched_update_weights


# ---------------------------------------------------------------------------
# Monkey-patch: LLMServerClient.generate()
# ---------------------------------------------------------------------------
def _patch_server_client():
    """Patch LLMServerClient.generate() to record per-request timing."""
    from verl.workers.rollout.llm_server import LLMServerClient

    orig_generate = LLMServerClient.generate.__wrapped__ \
        if hasattr(LLMServerClient.generate, '__wrapped__') else LLMServerClient.generate

    async def _instrumented_generate(
        self,
        request_id,
        *,
        prompt_ids,
        sampling_params,
        image_data=None,
        video_data=None,
        audio_data=None,
        mm_processor_kwargs=None,
        **kwargs,
    ):
        global _step_counter, _t_first_request

        t_start = time.perf_counter()
        if _t_first_request == 0.0:
            _t_first_request = t_start

        try:
            result = await orig_generate(
                self,
                request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=image_data,
                video_data=video_data,
                audio_data=audio_data,
                mm_processor_kwargs=mm_processor_kwargs,
                **kwargs,
            )
            t_end = time.perf_counter()
            output_tokens = len(result.token_ids) if result and hasattr(result, 'token_ids') and result.token_ids else 0

            record = {
                "request_id": request_id,
                "step": _step_counter,
                "t_start_abs": t_start,
                "t_end_abs": t_end,
                "t_start_offset_ms": (t_start - _t_origin) * 1000,
                "t_end_offset_ms": (t_end - _t_origin) * 1000,
                "latency_ms": (t_end - t_start) * 1000,
                "prompt_len": len(prompt_ids) if prompt_ids else 0,
                "output_tokens": output_tokens,
                "success": True,
                "error": "",
            }
            _safe_append(_request_records, record)
            return result

        except Exception as e:
            t_end = time.perf_counter()
            record = {
                "request_id": request_id,
                "step": _step_counter,
                "t_start_abs": t_start,
                "t_end_abs": t_end,
                "t_start_offset_ms": (t_start - _t_origin) * 1000,
                "t_end_offset_ms": (t_end - _t_origin) * 1000,
                "latency_ms": (t_end - t_start) * 1000,
                "prompt_len": len(prompt_ids) if prompt_ids else 0,
                "output_tokens": 0,
                "success": False,
                "error": str(e)[:300],
            }
            _safe_append(_request_records, record)
            raise

    LLMServerClient.generate = _instrumented_generate


# ---------------------------------------------------------------------------
# Monkey-patch: AgentLoopManager.generate_sequences()
# ---------------------------------------------------------------------------
def _patch_agent_loop_manager():
    """Patch AgentLoopManager.generate_sequences() to record step-level rollout timing."""
    from verl.experimental.agent_loop.agent_loop import AgentLoopManager
    from verl.utils.ray_utils import auto_await

    orig_generate = AgentLoopManager.generate_sequences

    @auto_await
    async def _instrumented_generate_sequences(self, prompts):
        global _step_counter, _t_origin

        if _t_origin == 0.0:
            _t_origin = time.perf_counter()

        # Snapshot request count before this step
        pre_step_count = len(_request_records)

        # Start metrics poller (25ms interval from request start)
        poller = None
        if _server_addresses:
            poller = _MetricsPoller(_server_addresses, interval_ms=25)
            poller.start()

        # Scrape baseline vLLM metrics
        baseline_metrics = {}
        if _server_addresses:
            try:
                async with aiohttp.ClientSession() as session:
                    for addr in _server_addresses:
                        m = await scrape_vllm_metrics(addr, session)
                        if m:
                            baseline_metrics[addr] = m
                            break
            except Exception:
                pass

        step_start_us = int(time.time() * 1e6)
        t_rollout_start = time.perf_counter()

        # Run the original generate_sequences
        result = await orig_generate(self, prompts)

        t_rollout_end = time.perf_counter()
        step_end_us = int(time.time() * 1e6)

        # Stop poller
        poll_samples = []
        if poller:
            poll_samples = await poller.stop()

        # Scrape post-step vLLM metrics
        post_metrics = {}
        vllm_delta = {}
        if _server_addresses:
            try:
                async with aiohttp.ClientSession() as session:
                    for addr in _server_addresses:
                        m = await scrape_vllm_metrics(addr, session)
                        if m:
                            post_metrics[addr] = m
                            break
            except Exception:
                pass

        if baseline_metrics and post_metrics:
            # Use first server's metrics for now
            b_addr = list(baseline_metrics.keys())[0]
            p_addr = list(post_metrics.keys())[0]
            vllm_delta = compute_metrics_delta(baseline_metrics[b_addr], post_metrics[p_addr])

        # Collect per-request records for this step
        step_records = _request_records[pre_step_count:]
        t_rollout_ms = (t_rollout_end - t_rollout_start) * 1000

        # Group completion time: from step start to each request's end
        if step_records:
            completion_offsets = sorted(
                [(r["t_end_abs"] - t_rollout_start) * 1000 for r in step_records]
            )
            n = len(completion_offsets)
            group_completion_p50 = completion_offsets[int(n * 0.50)] if n > 0 else 0
            group_completion_p95 = completion_offsets[min(int(n * 0.95), n - 1)] if n > 0 else 0
            group_completion_p99 = completion_offsets[min(int(n * 0.99), n - 1)] if n > 0 else 0

            # Output token throughput
            total_output_tokens = sum(r.get("output_tokens", 0) for r in step_records)
            throughput = total_output_tokens / (t_rollout_ms / 1000) if t_rollout_ms > 0 else 0

            # Per-request latency stats
            latencies = [r["latency_ms"] for r in step_records if r.get("success")]
            latencies.sort()
            num_ok = len(latencies)
            request_latency_p50 = latencies[int(num_ok * 0.50)] if num_ok > 0 else 0
            request_latency_p95 = latencies[min(int(num_ok * 0.95), num_ok - 1)] if num_ok > 0 else 0
            request_latency_p99 = latencies[min(int(num_ok * 0.99), num_ok - 1)] if num_ok > 0 else 0
        else:
            group_completion_p50 = group_completion_p95 = group_completion_p99 = 0
            total_output_tokens = 0
            throughput = 0
            request_latency_p50 = request_latency_p95 = request_latency_p99 = 0
            num_ok = 0

        # Jaeger trace fetching (optional)
        jaeger_step_data = {}
        if _server_addresses and step_end_us > step_start_us:
            # Will be populated if --jaeger-url is configured
            pass

        step_summary = {
            "step": _step_counter,
            "t_rollout_ms": round(t_rollout_ms, 2),
            "num_requests": len(step_records),
            "num_successful": num_ok,
            "group_completion_p50_ms": round(group_completion_p50, 2),
            "group_completion_p95_ms": round(group_completion_p95, 2),
            "group_completion_p99_ms": round(group_completion_p99, 2),
            "output_tokens_total": total_output_tokens,
            "output_token_throughput": round(throughput, 2),
            "request_latency_p50_ms": round(request_latency_p50, 2),
            "request_latency_p95_ms": round(request_latency_p95, 2),
            "request_latency_p99_ms": round(request_latency_p99, 2),
            "vllm_metrics_delta": vllm_delta,
            "poll_samples": poll_samples,
            "step_start_unix_us": step_start_us,
            "step_end_unix_us": step_end_us,
        }

        if jaeger_step_data:
            step_summary["jaeger"] = jaeger_step_data

        _safe_append(_rollout_steps, step_summary)
        _step_counter += 1

        return result

    AgentLoopManager.generate_sequences = _instrumented_generate_sequences


# ---------------------------------------------------------------------------
# Jaeger integration (called from 05_grpo_instrumented.py if --jaeger-url set)
# ---------------------------------------------------------------------------
async def fetch_jaeger_data_for_step(step_idx: int, start_us: int, end_us: int,
                                      jaeger_url: str) -> dict:
    """Fetch Jaeger OTel spans for a specific rollout step."""
    fetcher = _JaegerTraceFetcher(jaeger_url)
    try:
        async with aiohttp.ClientSession() as session:
            spans = await fetcher.fetch_spans(session, start_us - 5_000_000, end_us + 5_000_000)
            if not spans:
                return {}

            # Filter to step window
            filtered = []
            for span in spans:
                span_start = int(span.get("startTime", 0) or 0)
                if start_us <= span_start <= end_us:
                    filtered.append(span)

            if not filtered:
                return {}

            latencies = _JaegerTraceFetcher.extract_gen_ai_latencies(filtered)
            result = {}
            for tag_name, values in latencies.items():
                key = "otel_" + tag_name.replace("gen_ai.latency.", "").replace(".", "_")
                result[key] = _JaegerTraceFetcher.compute_percentiles(values)

            result["otel_span_count"] = len(filtered)
            span_ops = set()
            for s in filtered:
                op = s.get("operationName", "")
                if op:
                    span_ops.add(op)
            result["otel_span_operations"] = sorted(span_ops)
            return result
    except Exception as e:
        logger.warning(f"[phase2_instrument] Jaeger fetch failed: {e}")
        return {}


# ---------------------------------------------------------------------------
# Apply all patches
# ---------------------------------------------------------------------------
_patched = False


def apply_patches():
    """Apply all monkey-patches. Call once before training."""
    global _patched, _t_origin
    if _patched:
        return
    _t_origin = time.perf_counter()

    _patch_server_manager()
    _patch_server_client()
    _patch_agent_loop_manager()
    _patch_vllm_rollout()

    _patched = True
    logger.info("[phase2_instrument] All patches applied")


# ---------------------------------------------------------------------------
# Data export
# ---------------------------------------------------------------------------
def get_instrumentation_data() -> dict:
    """Return all collected instrumentation data."""
    # Strip absolute times (not useful in output), keep offsets
    clean_requests = []
    for r in _request_records:
        clean = {k: v for k, v in r.items() if not k.endswith("_abs")}
        clean_requests.append(clean)

    return {
        "server_addresses": _server_addresses,
        "num_rollout_steps": len(_rollout_steps),
        "num_requests": len(clean_requests),
        "rollout_steps": _rollout_steps,
        "request_records": clean_requests,
        "jaeger_data": _jaeger_data,
    }


def reset():
    """Reset all collected data (useful for sweep mode)."""
    global _step_counter, _t_origin, _t_first_request
    _rollout_steps.clear()
    _request_records.clear()
    _poll_samples.clear()
    _jaeger_data.clear()
    _server_addresses.clear()
    _step_counter = 0
    _t_origin = 0.0
    _t_first_request = 0.0


# Auto-apply on import
apply_patches()
