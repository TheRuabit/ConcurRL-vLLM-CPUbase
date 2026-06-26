#!/usr/bin/env python3
"""
Metrics Compiler & Markdown Visualizer
=======================================
Reads raw JSON from 03_concurrency_driver.json, computes P50/P95/P99
aggregations across all 5 latency parameters for each concurrency tier,
and generates:
  1. result/04_metrics_compiler.json — structured analytics
  2. PHASE_1_SUMMARY.md — scannable benchmark table

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
        "t_http_overhead_ms",
        "t_server_prefill_ms",
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
        }

        for key in metric_keys:
            values = [t[key] for t in successful if key in t]
            tier_stats[key] = compute_full_stats(values)

        # Derived: CPU vs GPU breakdown
        cpu_times = [t["t_serialize_ms"] + t["t_http_overhead_ms"]
                     for t in successful]
        gpu_times = [t["t_server_prefill_ms"] + t["t_decode_ms"]
                     for t in successful]
        total_times = [c + g for c, g in zip(cpu_times, gpu_times)]

        tier_stats["cpu_time_ms"] = compute_full_stats(cpu_times)
        tier_stats["gpu_time_ms"] = compute_full_stats(gpu_times)
        tier_stats["total_time_ms"] = compute_full_stats(total_times)

        # Percentages from mean
        cpu_mean = tier_stats["cpu_time_ms"]["mean"]
        gpu_mean = tier_stats["gpu_time_ms"]["mean"]
        total_mean = tier_stats["total_time_ms"]["mean"]
        tier_stats["cpu_percent"] = round(
            (cpu_mean / total_mean * 100) if total_mean > 0 else 0, 2
        )
        tier_stats["gpu_percent"] = round(
            (gpu_mean / total_mean * 100) if total_mean > 0 else 0, 2
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

    # Main latency table (P95)
    lines.append("## Latency Breakdown (P95, ms)\n")
    lines.append("| Concurrency | Client Serialization ($P_{95}$) | "
                 "Server Overhead ($P_{95}$) | GPU Prefill ($P_{95}$) | "
                 "GPU Decode ($P_{95}$) | Response Parsing ($P_{95}$) | "
                 "Status / Degradation Source |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")

    for level in scenarios:
        tier = compiled_tiers.get(str(level), {})
        if not tier or tier.get("successful", 0) == 0:
            lines.append(f"| **{level}** | — | — | — | — | — | Failed |")
            continue

        s = tier
        ser = f"{s['t_serialize_ms']['p95']:.1f}ms"
        ohs = f"{s['t_http_overhead_ms']['p95']:.1f}ms"
        pre = f"{s['t_server_prefill_ms']['p95']:.1f}ms"
        dec = f"{s['t_decode_ms']['p95']:.1f}ms"
        par = f"{s['t_response_parse_ms']['p95']:.1f}ms"

        e2e_p95 = s['t_e2e_ms']['p95']
        if e2e_p95 < 500:
            status = "Nominal execution"
        elif e2e_p95 < 2000:
            status = "Initial Queue Contention"
        elif e2e_p95 < 5000:
            status = "Context-Switch Thrashing"
        else:
            status = "Server Breakdown"

        lines.append(f"| **{level}** | {ser} | {ohs} | {pre} | {dec} | {par} | {status} |")

    # Detailed stats table (mean/P50/P95/P99)
    lines.append("\n## Detailed Statistics (ms)\n")
    lines.append("| Concurrency | Metric | Mean | P50 | P95 | P99 | Min | Max |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")

    metric_labels = {
        "t_serialize_ms": "Client Serialization",
        "t_http_overhead_ms": "Server Overhead",
        "t_server_prefill_ms": "GPU Prefill (TTFT)",
        "t_decode_ms": "GPU Decode",
        "t_response_parse_ms": "Response Parsing",
        "t_e2e_ms": "End-to-End",
    }

    for level in scenarios:
        tier = compiled_tiers.get(str(level), {})
        if not tier or tier.get("successful", 0) == 0:
            continue
        for key, label in metric_labels.items():
            st = tier.get(key, {})
            if not st:
                continue
            lines.append(
                f"| **{level}** | {label} | "
                f"{st['mean']:.2f} | {st['p50']:.2f} | "
                f"{st['p95']:.2f} | {st['p99']:.2f} | "
                f"{st['min']:.2f} | {st['max']:.2f} |"
            )

    # CPU vs GPU breakdown
    lines.append("\n## CPU vs GPU Time Breakdown\n")
    lines.append("| Concurrency | CPU Time (mean) | GPU Time (mean) | "
                 "Total (mean) | CPU% | GPU% |")
    lines.append("| --- | --- | --- | --- | --- | --- |")

    for level in scenarios:
        tier = compiled_tiers.get(str(level), {})
        if not tier or tier.get("successful", 0) == 0:
            continue
        cpu = tier.get("cpu_time_ms", {})
        gpu = tier.get("gpu_time_ms", {})
        total = tier.get("total_time_ms", {})
        lines.append(
            f"| **{level}** | "
            f"{cpu.get('mean', 0):.1f}ms | "
            f"{gpu.get('mean', 0):.1f}ms | "
            f"{total.get('mean', 0):.1f}ms | "
            f"{tier.get('cpu_percent', 0):.1f}% | "
            f"{tier.get('gpu_percent', 0):.1f}% |"
        )

    md_content = "\n".join(lines) + "\n"
    out_md.write_text(md_content, encoding="utf-8")
    print(f"[04_compiler] Markdown saved to {out_md}")

    # -------------------------------------------------------------------
    # Console summary
    # -------------------------------------------------------------------
    print("\n" + "=" * 110)
    print("METRICS COMPILER — P95 Latency Summary (ms)")
    print("=" * 110)
    header = (f"{'Conc':>6s} {'Serialize':>12s} {'HTTP_OH':>12s} "
              f"{'Prefill':>12s} {'Decode':>12s} {'Parse':>12s} "
              f"{'E2E':>12s} {'CPU%':>7s} {'GPU%':>7s}")
    print(header)
    print("-" * 110)

    for level in scenarios:
        tier = compiled_tiers.get(str(level), {})
        if not tier or tier.get("successful", 0) == 0:
            print(f"{level:>6d} {'FAIL':>12s}")
            continue
        print(f"{level:>6d} "
              f"{tier['t_serialize_ms']['p95']:>10.2f}ms "
              f"{tier['t_http_overhead_ms']['p95']:>10.2f}ms "
              f"{tier['t_server_prefill_ms']['p95']:>10.2f}ms "
              f"{tier['t_decode_ms']['p95']:>10.2f}ms "
              f"{tier['t_response_parse_ms']['p95']:>10.2f}ms "
              f"{tier['t_e2e_ms']['p95']:>10.2f}ms "
              f"{tier.get('cpu_percent', 0):>6.1f}% "
              f"{tier.get('gpu_percent', 0):>6.1f}%")
    print("=" * 110)


if __name__ == "__main__":
    main()
