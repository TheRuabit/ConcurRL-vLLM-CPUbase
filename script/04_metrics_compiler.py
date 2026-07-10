#!/usr/bin/env python3
"""
Metrics Compiler & Markdown Visualizer
=======================================
Reads raw JSON from 03_concurrency_driver.json, computes P50/P95/P99
aggregations across all latency parameters for each concurrency tier,
and generates:
  1. result/04_metrics_compiler.json — structured analytics
  2. PHASE_1_SUMMARY.md — scannable benchmark table

CPU/GPU timing decomposition:
  CPU-side: serialize + semaphore_wait + HTTP_connect
  GPU-side: prefill + decode

Usage:
    python script/04_metrics_compiler.py
    python script/04_metrics_compiler.py --input result/03_concurrency_driver.json

Output:
    result/04_metrics_compiler.json
    PHASE_1_SUMMARY.md
"""

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Parse CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Metrics compiler — JSON aggregator and Markdown visualizer"
)
parser.add_argument("--input", default=None,
                    help="Path to 03_concurrency_driver.json")
parser.add_argument("--output-json", default=None,
                    help="Output JSON path")
parser.add_argument("--output-md", default=None,
                    help="Output Markdown path")
args = parser.parse_args()

# Resolve paths
script_dir = Path(__file__).resolve().parent
project_dir = script_dir.parent

if args.input:
    in_path = Path(args.input)
else:
    in_path = project_dir / "result" / "03_concurrency_driver.json"

if args.output_json:
    out_json = Path(args.output_json)
else:
    out_json = project_dir / "result" / "04_metrics_compiler.json"

if args.output_md:
    out_md = Path(args.output_md)
else:
    out_md = project_dir / "PHASE_1_SUMMARY.md"

out_json.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Percentile helpers
# ---------------------------------------------------------------------------
def percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = int(len(sorted_vals) * p)
    return sorted_vals[min(idx, len(sorted_vals) - 1)]


def compute_full_stats(values: list[float], ndigits: int = 4) -> dict:
    if not values:
        return {"mean": 0, "p50": 0, "p95": 0, "p99": 0, "min": 0, "max": 0, "count": 0}
    n = len(values)
    s = sorted(values)
    return {
        "mean": round(sum(values) / n, ndigits),
        "p50": round(percentile(s, 0.50), ndigits),
        "p95": round(percentile(s, 0.95), ndigits),
        "p99": round(percentile(s, 0.99), ndigits),
        "min": round(s[0], ndigits),
        "max": round(s[-1], ndigits),
        "count": n,
    }


HTTP_SUBPHASE_KEYS = [
    "t_http_conn_queued_ms",
    "t_http_dns_ms",
    "t_http_tcp_connect_ms",
    "t_http_request_send_ms",
    "t_http_response_headers_wait_ms",
    "t_response_header_to_first_sse_byte_ms",
    "t_first_sse_byte_to_first_token_ms",
]


REDUNDANT_DISPLAY_KEYS = {
    # Duplicates t_prefill_ms in the current streaming measurements.
    "t_response_header_to_first_sse_byte_ms",
    # Derived from t_sem_wait_ms + t_http_connect_ms.
    "t_first_byte_ms",
    # Largely duplicates t_decode_ms for streamed responses.
    "t_response_parse_ms",
}


def format_stat(tier: dict, key: str, stat: str = "p95", ndigits: int = 1) -> str:
    values = tier.get(key)
    if not values:
        return "n/a"
    return f"{values.get(stat, 0):.{ndigits}f}"


def format_seconds_stat(stats: dict, stat: str = "p95", ndigits: int = 1) -> str:
    if not stats or stats.get("count", 0) == 0:
        return "n/a"
    return f"{stats.get(stat, 0) * 1000:.{ndigits}f}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"[04_compiler] Metrics Compiler")
    print(f"[04_compiler] Input:  {in_path}")

    if not in_path.exists():
        print(f"[04_compiler] ERROR: Input file not found: {in_path}")
        print(f"[04_compiler] Run 03_concurrency_driver.py first")
        sys.exit(1)

    data = json.loads(in_path.read_text(encoding="utf-8"))
    raw_traces = data.get("raw_traces", [])
    scenarios = data.get("scenarios", [])
    model = data.get("model", "unknown")
    input_tokens = data.get("input_tokens", 0)
    output_tokens = data.get("output_tokens", 0)
    num_batches = data.get("num_batches", 0)
    vllm_metrics = data.get("vllm_metrics", {})

    if not raw_traces:
        print(f"[04_compiler] ERROR: No raw traces in input file")
        sys.exit(1)

    print(f"[04_compiler] Model:  {model}")
    print(f"[04_compiler] Traces: {len(raw_traces)}")
    # Group by concurrency level
    grouped: dict[int, list[dict]] = {}
    for t in raw_traces:
        level = t.get("concurrency_level", 0)
        if level not in grouped:
            grouped[level] = []
        grouped[level].append(t)

    # Compute per-tier statistics
    metric_keys = [
        "t_serialize_ms",
        "t_sem_wait_ms",
        "t_http_connect_ms",
        "t_http_conn_queued_ms",
        "t_http_dns_ms",
        "t_http_tcp_connect_ms",
        "t_http_request_send_ms",
        "t_http_response_headers_wait_ms",
        "t_response_header_to_first_sse_byte_ms",
        "t_first_sse_byte_to_first_token_ms",
        "t_first_byte_ms",
        "t_server_prefill_ms",
        "t_prefill_ms",
        "t_decode_ms",
        "t_response_parse_ms",
        "t_e2e_ms",
    ]

    compiled_tiers = {}
    for level in sorted(grouped.keys()):
        traces = grouped[level]
        successful = [t for t in traces if t.get("success", False)]
        failed = [t for t in traces if not t.get("success", False)]

        if not successful:
            compiled_tiers[str(level)] = {
                "concurrency": level,
                "total_requests": len(traces),
                "successful": 0,
                "failed": len(failed),
                "error": "All requests failed",
            }
            continue

        tier_stats = {
            "concurrency": level,
            "total_requests": len(traces),
            "successful": len(successful),
            "failed": len(failed),
            "metric_availability": {},
        }

        for key in metric_keys:
            values = [t[key] for t in successful if key in t]
            tier_stats["metric_availability"][key] = {
                "count": len(values),
                "coverage_percent": round(len(values) / len(successful) * 100, 2),
            }
            if values:
                tier_stats[key] = compute_full_stats(values)

        # Derived CPU/GPU breakdown.
        # New display semantics:
        #   measured_cpu_time_ms: only measured client CPU work.
        #   client_network_wait_time_ms: client-side waits and elapsed network I/O.
        #   server_header_wait_time_ms: request sent -> response headers. This is
        #       the largest non-GPU bucket, but it is not measured CPU compute.
        measured_cpu_times = [t["t_serialize_ms"] for t in successful]
        client_network_wait_times = [
            t.get("t_sem_wait_ms", 0)
            + t.get("t_http_conn_queued_ms", 0)
            + t.get("t_http_dns_ms", 0)
            + t.get("t_http_tcp_connect_ms", 0)
            + t.get("t_http_request_send_ms", 0)
            for t in successful
        ]
        server_header_wait_times = [
            t.get("t_http_response_headers_wait_ms", t.get("t_http_connect_ms", 0))
            for t in successful
        ]

        # Backward-compatible historical buckets.
        client_times = [t["t_serialize_ms"] + t.get("t_sem_wait_ms", 0)
                        for t in successful]
        network_queue_times = [t.get("t_http_connect_ms", t["t_first_byte_ms"] - t.get("t_sem_wait_ms", 0))
                               for t in successful]
        gpu_times = [t["t_prefill_ms"] + t["t_decode_ms"]
                     for t in successful]
        total_times = [
            c + nw + sh + g for c, nw, sh, g in zip(
                measured_cpu_times,
                client_network_wait_times,
                server_header_wait_times,
                gpu_times,
            )
        ]

        tier_stats["measured_cpu_time_ms"] = compute_full_stats(measured_cpu_times)
        tier_stats["client_network_wait_time_ms"] = compute_full_stats(client_network_wait_times)
        tier_stats["server_header_wait_time_ms"] = compute_full_stats(server_header_wait_times)
        tier_stats["client_time_ms"] = compute_full_stats(client_times)
        tier_stats["network_queue_time_ms"] = compute_full_stats(network_queue_times)
        tier_stats["gpu_time_ms"] = compute_full_stats(gpu_times)
        tier_stats["total_time_ms"] = compute_full_stats(total_times)

        # Backward compat: cpu_time = client + network_queue
        cpu_times = [c + n for c, n in zip(client_times, network_queue_times)]
        tier_stats["cpu_time_ms"] = compute_full_stats(cpu_times)

        # Percentages from mean
        measured_cpu_mean = tier_stats["measured_cpu_time_ms"]["mean"]
        client_network_wait_mean = tier_stats["client_network_wait_time_ms"]["mean"]
        server_header_wait_mean = tier_stats["server_header_wait_time_ms"]["mean"]
        gpu_mean = tier_stats["gpu_time_ms"]["mean"]
        total_mean = tier_stats["total_time_ms"]["mean"]
        tier_stats["measured_cpu_percent"] = round(
            (measured_cpu_mean / total_mean * 100) if total_mean > 0 else 0, 2
        )
        tier_stats["client_network_wait_percent"] = round(
            (client_network_wait_mean / total_mean * 100) if total_mean > 0 else 0, 2
        )
        tier_stats["server_header_wait_percent"] = round(
            (server_header_wait_mean / total_mean * 100) if total_mean > 0 else 0, 2
        )
        tier_stats["gpu_percent"] = round(
            (gpu_mean / total_mean * 100) if total_mean > 0 else 0, 2
        )
        # Historical aliases.
        tier_stats["client_percent"] = tier_stats["measured_cpu_percent"]
        tier_stats["network_queue_percent"] = round(
            (tier_stats["network_queue_time_ms"]["mean"] / total_mean * 100)
            if total_mean > 0 else 0, 2
        )
        # Backward compat
        cpu_mean = tier_stats["cpu_time_ms"]["mean"]
        tier_stats["cpu_percent"] = round(
            (cpu_mean / total_mean * 100) if total_mean > 0 else 0, 2
        )

        compiled_tiers[str(level)] = tier_stats

    # -------------------------------------------------------------------
    # Save JSON
    # -------------------------------------------------------------------
    output_json = {
        "benchmark": "metrics_compiler_aggregation",
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "num_batches": num_batches,
        "scenarios": scenarios,
        "timing_breakdown_semantics": {
            "measured_cpu_time_ms": (
                "Only directly measured client CPU work. Currently this is JSON "
                "serialization; response parsing is not counted because the timer "
                "mostly overlaps streamed decode wait."
            ),
            "client_network_wait_time_ms": (
                "Client semaphore, connector, DNS, TCP, and request-send elapsed "
                "time. This is neither CPU compute nor GPU compute."
            ),
            "server_header_wait_time_ms": (
                "Request sent to response headers from the client perspective. "
                "This is a server/queue/admission wait bucket, not measured CPU "
                "or GPU compute."
            ),
            "client_time_ms": "Historical field: client-side JSON serialization plus local semaphore wait.",
            "network_queue_time_ms": (
                "Historical field: "
                "POST-to-response-header wait from the client perspective. This "
                "includes network, HTTP stack, server queue, and server work before "
                "response headers; it is not pure CPU compute time."
            ),
            "gpu_time_ms": "Client-observed prefill plus decode intervals.",
            "cpu_time_ms": (
                "Backward-compatible alias for client_time_ms + "
                "network_queue_time_ms; interpret as non-GPU/client-observed wait, "
                "not measured CPU compute."
            ),
        },
        "server_side_metrics": vllm_metrics,
        "tiers": compiled_tiers,
    }

    out_json.write_text(json.dumps(output_json, indent=2, ensure_ascii=False))
    print(f"[04_compiler] JSON saved to {out_json}")

    # -------------------------------------------------------------------
    # Generate Markdown
    # -------------------------------------------------------------------
    lines = []
    lines.append("# Phase 1: Concurrency Scaling Validation — Benchmark Results\n")
    lines.append(f"**Model:** {model}  ")
    lines.append(f"**Input Tokens:** {input_tokens:,}  ")
    lines.append(f"**Output Tokens:** {output_tokens}  ")
    lines.append(f"**Batches:** {num_batches}  ")
    lines.append(f"**Total Traces:** {len(raw_traces)}\n")

    # CPU vs GPU breakdown first.
    lines.append("## Timing Breakdown (Mean, Mutually Exclusive Buckets)\n")
    lines.append("| Concurrency | CPU Measured | Client/Network Wait | "
                 "Server/Header Wait | GPU Time | Total | CPU% | Wait% | GPU% |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")

    for level in scenarios:
        tier = compiled_tiers.get(str(level), {})
        if not tier or tier.get("successful", 0) == 0:
            continue
        cpu = tier.get("measured_cpu_time_ms", {})
        client_network = tier.get("client_network_wait_time_ms", {})
        server_header = tier.get("server_header_wait_time_ms", {})
        gpu = tier.get("gpu_time_ms", {})
        total = tier.get("total_time_ms", {})
        wait_percent = (
            tier.get("client_network_wait_percent", 0)
            + tier.get("server_header_wait_percent", 0)
        )
        lines.append(
            f"| **{level}** | "
            f"{cpu.get('mean', 0):.1f}ms | "
            f"{client_network.get('mean', 0):.1f}ms | "
            f"{server_header.get('mean', 0):.1f}ms | "
            f"{gpu.get('mean', 0):.1f}ms | "
            f"{total.get('mean', 0):.1f}ms | "
            f"{tier.get('measured_cpu_percent', 0):.1f}% | "
            f"{wait_percent:.1f}% | "
            f"{tier.get('gpu_percent', 0):.1f}% |"
        )

    # Main latency table (P95)
    lines.append("\n## Latency Breakdown (P95, ms)\n")
    lines.append("| Concurrency | Serialize ($P_{95}$) | "
                 "Client/Network Wait ($P_{95}$) | Server/Header Wait ($P_{95}$) | "
                 "Prefill ($P_{95}$) | Decode ($P_{95}$) | "
                 "E2E ($P_{95}$) | Status |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")

    for level in scenarios:
        tier = compiled_tiers.get(str(level), {})
        if not tier or tier.get("successful", 0) == 0:
            lines.append(f"| **{level}** | — | — | — | — | — | — | — | Failed |")
            continue

        s = tier
        ser = format_stat(s, "t_serialize_ms")
        client_network = format_stat(s, "client_network_wait_time_ms")
        server_header = format_stat(s, "server_header_wait_time_ms")
        pre = format_stat(s, "t_prefill_ms")
        dec = format_stat(s, "t_decode_ms")
        e2e = format_stat(s, "t_e2e_ms")

        e2e_p95 = s.get('t_e2e_ms', {}).get('p95', 0)
        if e2e_p95 < 500:
            status = "Nominal"
        elif e2e_p95 < 2000:
            status = "Queue Contention"
        elif e2e_p95 < 5000:
            status = "Thrashing"
        else:
            status = "Breakdown"

        lines.append(
            f"| **{level}** | {ser} | {client_network} | "
            f"{server_header} | {pre} | {dec} | {e2e} | {status} |"
        )

    # Detailed stats table (mean/P50/P95/P99)
    lines.append("\n## Detailed Statistics (ms)\n")
    lines.append("| Concurrency | Metric | Mean | P50 | P95 | P99 | Min | Max |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")

    metric_labels = {
        "t_serialize_ms": "Client Serialization",
        "t_sem_wait_ms": "Semaphore Wait (Client Gate)",
        "t_http_connect_ms": "HTTP Connect (Queue+Net)",
        "t_http_conn_queued_ms": "HTTP Connector Pool Wait",
        "t_http_dns_ms": "HTTP DNS Lookup",
        "t_http_tcp_connect_ms": "HTTP TCP Connect",
        "t_http_request_send_ms": "HTTP Request Send",
        "t_http_response_headers_wait_ms": "HTTP Response Headers Wait",
        "t_first_sse_byte_to_first_token_ms": "First SSE Byte to First Token",
        "t_server_prefill_ms": "TTFT (Server Prefill)",
        "t_prefill_ms": "GPU Prefill (TTFT - 1stByte)",
        "t_decode_ms": "GPU Decode",
        "t_e2e_ms": "End-to-End",
    }

    for level in scenarios:
        tier = compiled_tiers.get(str(level), {})
        if not tier or tier.get("successful", 0) == 0:
            continue
        for key, label in metric_labels.items():
            if key in REDUNDANT_DISPLAY_KEYS:
                continue
            st = tier.get(key, {})
            if not st:
                continue
            lines.append(
                f"| **{level}** | {label} | "
                f"{st['mean']:.2f} | {st['p50']:.2f} | "
                f"{st['p95']:.2f} | {st['p99']:.2f} | "
                f"{st['min']:.2f} | {st['max']:.2f} |"
            )

    # HTTP subphase breakdown
    lines.append("\n## HTTP Subphase Breakdown (P95, ms)\n")
    lines.append("| Concurrency | Total HTTP | Conn Pool | DNS | TCP Connect | "
                 "Request Send | Response Headers |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for level in scenarios:
        tier = compiled_tiers.get(str(level), {})
        if not tier or tier.get("successful", 0) == 0:
            continue
        has_http_subphase = any(key in tier for key in HTTP_SUBPHASE_KEYS)
        if not has_http_subphase:
            lines.append(
                f"| **{level}** | {format_stat(tier, 't_http_connect_ms')} | "
                "n/a | n/a | n/a | n/a | n/a |"
            )
            continue
        lines.append(
            f"| **{level}** | "
            f"{format_stat(tier, 't_http_connect_ms')} | "
            f"{format_stat(tier, 't_http_conn_queued_ms')} | "
            f"{format_stat(tier, 't_http_dns_ms')} | "
            f"{format_stat(tier, 't_http_tcp_connect_ms')} | "
            f"{format_stat(tier, 't_http_request_send_ms')} | "
            f"{format_stat(tier, 't_http_response_headers_wait_ms')} |"
        )

    # Server-side OTel breakdown when Jaeger spans are available.
    has_otel = any(
        vllm_metrics.get(str(level), {}).get("otel_e2e", {}).get("count", 0) > 0
        for level in scenarios
    )
    if has_otel:
        lines.append("\n## Server-side OTel Breakdown (P95, ms)\n")
        lines.append("| Concurrency | Client RespHdr Wait | Server Queue | "
                     "Server TTFT | Model Prefill | Model Decode | Server E2E | "
                     "Span Count |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for level in scenarios:
            tier = compiled_tiers.get(str(level), {})
            vm = vllm_metrics.get(str(level), {})
            if not tier or not vm:
                continue
            lines.append(
                f"| **{level}** | "
                f"{format_stat(tier, 't_http_response_headers_wait_ms')} | "
                f"{format_seconds_stat(vm.get('otel_time_in_queue', {}))} | "
                f"{format_seconds_stat(vm.get('otel_time_to_first_token', {}))} | "
                f"{format_seconds_stat(vm.get('otel_time_in_model_prefill', {}))} | "
                f"{format_seconds_stat(vm.get('otel_time_in_model_decode', {}))} | "
                f"{format_seconds_stat(vm.get('otel_e2e', {}))} | "
                f"{vm.get('otel_llm_request_span_count', vm.get('otel_e2e', {}).get('count', 0))} |"
            )

        has_timeline = any(
            vllm_metrics.get(str(level), {}).get("server_span_start_offset", {}).get("count", 0) > 0
            for level in scenarios
        )
        if has_timeline:
            lines.append("\n## Server-side Span Timeline (P95 offset from scenario start, ms)\n")
            lines.append("| Concurrency | Request Start | First Token | "
                         "Inference End | Request End | Span Duration |")
            lines.append("| --- | --- | --- | --- | --- | --- |")
            for level in scenarios:
                vm = vllm_metrics.get(str(level), {})
                if not vm:
                    continue
                lines.append(
                    f"| **{level}** | "
                    f"{format_seconds_stat(vm.get('server_span_start_offset', {}))} | "
                    f"{format_seconds_stat(vm.get('server_span_first_token_offset', {}))} | "
                    f"{format_seconds_stat(vm.get('server_span_inference_end_offset', {}))} | "
                    f"{format_seconds_stat(vm.get('server_span_end_offset', {}))} | "
                    f"{format_seconds_stat(vm.get('server_span_duration', {}))} |"
                )

    md_content = "\n".join(lines) + "\n"
    out_md.write_text(md_content, encoding="utf-8")
    print(f"[04_compiler] Markdown saved to {out_md}")

    # -------------------------------------------------------------------
    # Console summary
    # -------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("TIMING BREAKDOWN — Mean, mutually exclusive buckets")
    print("=" * 100)
    header = (f"{'Conc':>6s} {'CPU':>12s} {'ClientNet':>12s} "
              f"{'SrvHeader':>12s} {'GPU':>12s} {'Total':>12s} "
              f"{'CPU%':>7s} {'Wait%':>7s} {'GPU%':>7s}")
    print(header)
    print("-" * 100)
    for level in scenarios:
        tier = compiled_tiers.get(str(level), {})
        if not tier or tier.get("successful", 0) == 0:
            print(f"{level:>6d} {'FAIL':>12s}")
            continue
        wait_percent = (
            tier.get("client_network_wait_percent", 0)
            + tier.get("server_header_wait_percent", 0)
        )
        print(f"{level:>6d} "
              f"{tier.get('measured_cpu_time_ms', {}).get('mean', 0):>10.2f}ms "
              f"{tier.get('client_network_wait_time_ms', {}).get('mean', 0):>10.2f}ms "
              f"{tier.get('server_header_wait_time_ms', {}).get('mean', 0):>10.2f}ms "
              f"{tier.get('gpu_time_ms', {}).get('mean', 0):>10.2f}ms "
              f"{tier.get('total_time_ms', {}).get('mean', 0):>10.2f}ms "
              f"{tier.get('measured_cpu_percent', 0):>6.1f}% "
              f"{wait_percent:>6.1f}% "
              f"{tier.get('gpu_percent', 0):>6.1f}%")
    print("=" * 100)

    print("\n" + "=" * 150)
    print("METRICS COMPILER — P95 Latency Summary (ms)")
    print("=" * 150)
    header = (f"{'Conc':>6s} {'Ser':>8s} {'CliNet':>8s} {'SrvHdr':>8s} "
              f"{'Prefill':>8s} {'Decode':>8s} {'E2E':>8s} "
              f"{'CPU%':>5s} {'GPU%':>5s}")
    print(header)
    print("-" * 150)

    for level in scenarios:
        tier = compiled_tiers.get(str(level), {})
        if not tier or tier.get("successful", 0) == 0:
            print(f"{level:>6d} {'FAIL':>8s}")
            continue
        print(f"{level:>6d} "
              f"{tier.get('t_serialize_ms', {}).get('p95', 0):>6.2f}ms "
              f"{tier.get('client_network_wait_time_ms', {}).get('p95', 0):>6.2f}ms "
              f"{tier.get('server_header_wait_time_ms', {}).get('p95', 0):>6.2f}ms "
              f"{tier.get('t_prefill_ms', {}).get('p95', 0):>6.2f}ms "
              f"{tier.get('t_decode_ms', {}).get('p95', 0):>6.2f}ms "
              f"{tier.get('t_e2e_ms', {}).get('p95', 0):>6.2f}ms "
              f"{tier.get('measured_cpu_percent', 0):>4.1f}% "
              f"{tier.get('gpu_percent', 0):>4.1f}%")
    print("=" * 150)

    if any(
        vllm_metrics.get(str(level), {}).get("otel_e2e", {}).get("count", 0) > 0
        for level in scenarios
    ):
        print("\n" + "=" * 150)
        print("SERVER-SIDE OTEL BREAKDOWN — P95 (ms)")
        print("=" * 150)
        header = (f"{'Conc':>6s} {'RespHdr':>9s} {'SrvQueue':>9s} "
                  f"{'SrvTTFT':>9s} {'Prefill':>9s} {'Decode':>9s} "
                  f"{'SrvE2E':>9s} {'Spans':>7s}")
        print(header)
        print("-" * 150)
        for level in scenarios:
            tier = compiled_tiers.get(str(level), {})
            vm = vllm_metrics.get(str(level), {})
            if not tier or not vm:
                continue
            print(f"{level:>6d} "
                  f"{format_stat(tier, 't_http_response_headers_wait_ms'):>9s} "
                  f"{format_seconds_stat(vm.get('otel_time_in_queue', {})):>9s} "
                  f"{format_seconds_stat(vm.get('otel_time_to_first_token', {})):>9s} "
                  f"{format_seconds_stat(vm.get('otel_time_in_model_prefill', {})):>9s} "
                  f"{format_seconds_stat(vm.get('otel_time_in_model_decode', {})):>9s} "
                  f"{format_seconds_stat(vm.get('otel_e2e', {})):>9s} "
                  f"{vm.get('otel_llm_request_span_count', vm.get('otel_e2e', {}).get('count', 0)):>7}")
        print("=" * 150)

    print("\n" + "=" * 150)
    print("HTTP SUBPHASE BREAKDOWN — P95 (ms)")
    print("=" * 150)
    header = (f"{'Conc':>6s} {'HTTP':>9s} {'ConnQ':>9s} {'DNS':>9s} "
              f"{'TCP':>9s} {'Send':>9s} {'RespHdr':>9s}")
    print(header)
    print("-" * 150)
    for level in scenarios:
        tier = compiled_tiers.get(str(level), {})
        if not tier or tier.get("successful", 0) == 0:
            print(f"{level:>6d} {'FAIL':>9s}")
            continue
        def p95_text(key: str) -> str:
            values = tier.get(key)
            if not values:
                return "n/a"
            return f"{values.get('p95', 0):.2f}ms"
        print(f"{level:>6d} "
              f"{p95_text('t_http_connect_ms'):>9s} "
              f"{p95_text('t_http_conn_queued_ms'):>9s} "
              f"{p95_text('t_http_dns_ms'):>9s} "
              f"{p95_text('t_http_tcp_connect_ms'):>9s} "
              f"{p95_text('t_http_request_send_ms'):>9s} "
              f"{p95_text('t_http_response_headers_wait_ms'):>9s}")
    print("=" * 150)


if __name__ == "__main__":
    main()
