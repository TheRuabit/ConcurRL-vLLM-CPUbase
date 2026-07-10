#!/usr/bin/env python3
"""
Phase 2 Rollout Metrics Compiler
==================================
Reads instrumented rollout data from 05_grpo_instrumented.json and computes:
  - Per-step rollout decomposition
  - Per-request latency stats (P50/P95/P99/mean)
  - P95 group completion time
  - Output token throughput
  - vLLM server-side metrics (Prometheus histograms + gauges)
  - OTel traces from Jaeger (if available)

Outputs:
    result/07b_phase2_rollout_metrics.json
    PHASE_2_ROLLOUT_SUMMARY.md

Usage:
    python script/07b_phase2_rollout_metrics.py
    python script/07b_phase2_rollout_metrics.py --input result/05_grpo_instrumented.json
"""

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Parse CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Phase 2 rollout metrics compiler"
)
parser.add_argument("--input", default=None,
                    help="Path to 05_grpo_instrumented.json")
parser.add_argument("--output-json", default=None,
                    help="Output JSON path")
parser.add_argument("--output-md", default=None,
                    help="Output Markdown path")
args = parser.parse_args()

script_dir = Path(__file__).resolve().parent
project_dir = script_dir.parent

if args.input:
    in_path = Path(args.input)
else:
    in_path = project_dir / "result" / "05_grpo_instrumented.json"

if args.output_json:
    out_json = Path(args.output_json)
else:
    out_json = project_dir / "result" / "07b_phase2_rollout_metrics.json"

if args.output_md:
    out_md = Path(args.output_md)
else:
    out_md = project_dir / "PHASE_2_ROLLOUT_SUMMARY.md"

out_json.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------
def percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = int(len(sorted_vals) * p)
    return sorted_vals[min(idx, len(sorted_vals) - 1)]


def compute_stats(values: list[float], ndigits: int = 4) -> dict:
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
    print(f"[07b_compiler] Phase 2 Rollout Metrics Compiler")
    print(f"[07b_compiler] Input:  {in_path}")

    if not in_path.exists():
        print(f"[07b_compiler] ERROR: Input file not found: {in_path}")
        print(f"[07b_compiler] Run 05_grpo_instrumented.py first")
        sys.exit(1)

    data = json.loads(in_path.read_text(encoding="utf-8"))
    rollout_steps = data.get("rollout_steps", [])
    request_records = data.get("request_records", [])

    if not rollout_steps:
        print(f"[07b_compiler] ERROR: No rollout steps in input file")
        sys.exit(1)

    model = data.get("model", "unknown")
    rollout_n = data.get("rollout_n", 0)
    train_batch_size = data.get("train_batch_size", 0)
    concurrency = data.get("concurrency", rollout_n * train_batch_size)
    num_epochs = data.get("num_epochs", 0)
    total_elapsed = data.get("total_elapsed_s", 0)
    enable_otel = data.get("enable_otel", False)

    print(f"[07b_compiler] Model:       {model}")
    print(f"[07b_compiler] Concurrency: {concurrency} (B={train_batch_size}, G={rollout_n})")
    print(f"[07b_compiler] Steps:       {len(rollout_steps)}")
    print(f"[07b_compiler] Requests:    {len(request_records)}")

    # -------------------------------------------------------------------
    # A. Per-step rollout decomposition
    # -------------------------------------------------------------------
    step_keys = [
        "t_rollout_ms", "num_requests", "num_successful",
        "group_completion_p50_ms", "group_completion_p95_ms", "group_completion_p99_ms",
        "output_tokens_total", "output_token_throughput",
        "request_latency_p50_ms", "request_latency_p95_ms", "request_latency_p99_ms",
    ]

    step_stats = {}
    for key in step_keys:
        values = [s.get(key, 0) for s in rollout_steps if key in s]
        if values:
            step_stats[key] = compute_stats(values)

    # -------------------------------------------------------------------
    # B. Per-request latency stats
    # -------------------------------------------------------------------
    request_stats = {}
    if request_records:
        success_reqs = [r for r in request_records if r.get("success")]
        if success_reqs:
            for key in ["latency_ms", "prompt_len", "output_tokens"]:
                values = [r.get(key, 0) for r in success_reqs if key in r]
                if values:
                    request_stats[key] = compute_stats(values)

    # -------------------------------------------------------------------
    # C. vLLM server-side metrics (from first step's delta)
    # -------------------------------------------------------------------
    vllm_metrics = {}
    first_step = rollout_steps[0] if rollout_steps else {}
    vllm_delta = first_step.get("vllm_metrics_delta", {})
    if vllm_delta:
        for key, val in vllm_delta.items():
            if isinstance(val, (int, float)):
                vllm_metrics[key] = val

    # Aggregate vLLM metrics across all steps
    vllm_aggregated = {}
    metric_keys_to_agg = [
        "vllm_queue_time_p50", "vllm_queue_time_p95", "vllm_queue_time_p99",
        "vllm_prefill_time_p50", "vllm_prefill_time_p95", "vllm_prefill_time_p99",
        "vllm_decode_time_p50", "vllm_decode_time_p95", "vllm_decode_time_p99",
        "vllm_ttft_p50", "vllm_ttft_p95", "vllm_ttft_p99",
        "vllm_itl_p50", "vllm_itl_p95", "vllm_itl_p99",
        "vllm_e2e_p50", "vllm_e2e_p95", "vllm_e2e_p99",
        "vllm_prefix_cache_hit_rate", "vllm_num_preemptions",
    ]
    for key in metric_keys_to_agg:
        values = []
        for step in rollout_steps:
            v = step.get("vllm_metrics_delta", {}).get(key)
            if v is not None:
                values.append(v)
        if values:
            vllm_aggregated[key] = compute_stats(values)

    # -------------------------------------------------------------------
    # D. OTel / Jaeger metrics
    # -------------------------------------------------------------------
    jaeger_metrics = {}
    if enable_otel:
        jaeger_keys = [
            "otel_time_in_queue_p50", "otel_time_in_queue_p95",
            "otel_time_in_model_prefill_p50", "otel_time_in_model_prefill_p95",
            "otel_time_in_model_decode_p50", "otel_time_in_model_decode_p95",
            "otel_time_in_model_inference_p50", "otel_time_in_model_inference_p95",
            "otel_time_to_first_token_p50", "otel_time_to_first_token_p95",
            "otel_tokenization_overhead_p50", "otel_tokenization_overhead_p95",
            "otel_e2e_p50", "otel_e2e_p95",
        ]
        for key in jaeger_keys:
            values = []
            for step in rollout_steps:
                v = step.get("jaeger", {}).get(key)
                if v is not None:
                    values.append(v)
            if values:
                jaeger_metrics[key] = compute_stats(values)

    # -------------------------------------------------------------------
    # Save JSON
    # -------------------------------------------------------------------
    output_json = {
        "benchmark": "phase2_rollout_metrics",
        "model": model,
        "rollout_n": rollout_n,
        "train_batch_size": train_batch_size,
        "concurrency": concurrency,
        "num_epochs": num_epochs,
        "num_steps": len(rollout_steps),
        "num_requests_total": len(request_records),
        "total_elapsed_s": total_elapsed,
        "enable_otel": enable_otel,
        "step_stats": step_stats,
        "request_stats": request_stats,
        "vllm_metrics_first_step": vllm_metrics,
        "vllm_metrics_aggregated": vllm_aggregated,
        "jaeger_metrics": jaeger_metrics,
        "step_details": rollout_steps,
    }

    out_json.write_text(json.dumps(output_json, indent=2, ensure_ascii=False))
    print(f"[07b_compiler] JSON saved to {out_json}")

    # -------------------------------------------------------------------
    # Generate Markdown
    # -------------------------------------------------------------------
    lines = []
    lines.append("# Phase 2: GRPO Rollout — Deep Instrumentation Report\n")
    lines.append(f"**Model:** {model}  ")
    lines.append(f"**Concurrency (B×G):** {concurrency} (B={train_batch_size}, G={rollout_n})  ")
    lines.append(f"**Epochs:** {num_epochs}  ")
    lines.append(f"**Steps:** {len(rollout_steps)}  ")
    lines.append(f"**Total Requests:** {len(request_records)}  ")
    lines.append(f"**Total Time:** {total_elapsed:.1f}s  ")
    lines.append(f"**OTel:** {'enabled' if enable_otel else 'disabled'}\n")

    # Rollout timing
    lines.append("## Rollout Step Timing\n")
    lines.append("| Metric | Mean | P50 | P95 | Min | Max |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for key, label in [
        ("t_rollout_ms", "Rollout Total (ms)"),
        ("group_completion_p50_ms", "Group Completion P50 (ms)"),
        ("group_completion_p95_ms", "Group Completion P95 (ms)"),
        ("group_completion_p99_ms", "Group Completion P99 (ms)"),
        ("output_tokens_total", "Output Tokens / Step"),
        ("output_token_throughput", "Output Token Throughput (tok/s)"),
    ]:
        st = step_stats.get(key)
        if not st:
            continue
        lines.append(
            f"| {label} | {st['mean']:.1f} | {st['p50']:.1f} | "
            f"{st['p95']:.1f} | {st['min']:.1f} | {st['max']:.1f} |"
        )

    # Per-request latency
    if request_stats:
        lines.append("\n## Per-Request Latency (ms)\n")
        lines.append("| Metric | Mean | P50 | P95 | P99 | Min | Max | Count |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for key, label in [
            ("latency_ms", "Request Latency"),
            ("prompt_len", "Prompt Length (tokens)"),
            ("output_tokens", "Output Tokens"),
        ]:
            st = request_stats.get(key)
            if not st:
                continue
            lines.append(
                f"| {label} | {st['mean']:.1f} | {st['p50']:.1f} | "
                f"{st['p95']:.1f} | {st['p99']:.1f} | {st['min']:.1f} | "
                f"{st['max']:.1f} | {st['count']} |"
            )

    # Per-request latency from step stats
    lines.append("\n## Per-Request Latency (from step aggregation)\n")
    lines.append("| Metric | Mean | P50 | P95 | Min | Max |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for key, label in [
        ("request_latency_p50_ms", "Request Latency P50"),
        ("request_latency_p95_ms", "Request Latency P95"),
        ("request_latency_p99_ms", "Request Latency P99"),
    ]:
        st = step_stats.get(key)
        if not st:
            continue
        lines.append(
            f"| {label} | {st['mean']:.1f} | {st['p50']:.1f} | "
            f"{st['p95']:.1f} | {st['min']:.1f} | {st['max']:.1f} |"
        )

    # vLLM server-side metrics
    if vllm_aggregated:
        lines.append("\n## vLLM Server-Side Metrics (Prometheus)\n")
        lines.append("| Metric | Mean | P50 | P95 | P99 |")
        lines.append("| --- | --- | --- | --- | --- |")
        vllm_labels = {
            "vllm_queue_time_p95": "Queue Time P95 (s)",
            "vllm_prefill_time_p95": "Prefill Time P95 (s)",
            "vllm_decode_time_p95": "Decode Time P95 (s)",
            "vllm_ttft_p95": "TTFT P95 (s)",
            "vllm_itl_p95": "Inter-Token Latency P95 (s)",
            "vllm_e2e_p95": "E2E P95 (s)",
            "vllm_prefix_cache_hit_rate": "Prefix Cache Hit Rate",
            "vllm_num_preemptions": "Preemptions",
        }
        for key, label in vllm_labels.items():
            st = vllm_aggregated.get(key)
            if not st:
                continue
            lines.append(
                f"| {label} | {st['mean']:.4f} | {st['p50']:.4f} | "
                f"{st['p95']:.4f} | {st['p99']:.4f} |"
            )

    # OTel / Jaeger metrics
    if jaeger_metrics:
        lines.append("\n## OTel Traces (Jaeger)\n")
        lines.append("| Metric | Mean | P50 | P95 | P99 |")
        lines.append("| --- | --- | --- | --- | --- |")
        otel_labels = {
            "otel_time_in_queue_p95": "Queue Time P95 (s)",
            "otel_time_in_model_prefill_p95": "Prefill Time P95 (s)",
            "otel_time_in_model_decode_p95": "Decode Time P95 (s)",
            "otel_time_in_model_inference_p95": "Inference Time P95 (s)",
            "otel_time_to_first_token_p95": "TTFT P95 (s)",
            "otel_tokenization_overhead_p95": "Tokenization Overhead P95 (s)",
            "otel_e2e_p95": "E2E P95 (s)",
        }
        for key, label in otel_labels.items():
            st = jaeger_metrics.get(key)
            if not st:
                continue
            lines.append(
                f"| {label} | {st['mean']:.4f} | {st['p50']:.4f} | "
                f"{st['p95']:.4f} | {st['p99']:.4f} |"
            )

    # Per-step detail table
    if rollout_steps:
        lines.append("\n## Per-Step Detail\n")
        lines.append("| Step | Rollout(ms) | Reqs | OK | P50_group | P95_group | Tokens | Tok/s | P95_lat |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for step in rollout_steps:
            lines.append(
                f"| {step['step']} | "
                f"{step.get('t_rollout_ms', 0):.0f} | "
                f"{step.get('num_requests', 0)} | "
                f"{step.get('num_successful', 0)} | "
                f"{step.get('group_completion_p50_ms', 0):.0f} | "
                f"{step.get('group_completion_p95_ms', 0):.0f} | "
                f"{step.get('output_tokens_total', 0)} | "
                f"{step.get('output_token_throughput', 0):.0f} | "
                f"{step.get('request_latency_p95_ms', 0):.0f} |"
            )

    md_content = "\n".join(lines) + "\n"
    out_md.write_text(md_content, encoding="utf-8")
    print(f"[07b_compiler] Markdown saved to {out_md}")

    # -------------------------------------------------------------------
    # Console summary
    # -------------------------------------------------------------------
    print(f"\n{'='*100}")
    print(f"PHASE 2 — Rollout Instrumentation Summary")
    print(f"{'='*100}")
    print(f"  Concurrency (B×G): {concurrency}")
    print(f"  Steps:    {len(rollout_steps)}")
    print(f"  Requests: {len(request_records)}")
    print(f"  Total:    {total_elapsed:.1f}s")

    if step_stats.get("t_rollout_ms"):
        s = step_stats["t_rollout_ms"]
        print(f"  Rollout:  mean={s['mean']:.0f}ms  p95={s['p95']:.0f}ms")
    if step_stats.get("group_completion_p95_ms"):
        s = step_stats["group_completion_p95_ms"]
        print(f"  P95 Group Completion: mean={s['mean']:.0f}ms  p95={s['p95']:.0f}ms")
    if step_stats.get("output_token_throughput"):
        s = step_stats["output_token_throughput"]
        print(f"  Output Throughput:    mean={s['mean']:.0f} tok/s  p95={s['p95']:.0f} tok/s")
    if step_stats.get("request_latency_p95_ms"):
        s = step_stats["request_latency_p95_ms"]
        print(f"  Request Latency P95:  mean={s['mean']:.0f}ms  p95={s['p95']:.0f}ms")

    if vllm_aggregated.get("vllm_prefix_cache_hit_rate"):
        s = vllm_aggregated["vllm_prefix_cache_hit_rate"]
        print(f"  Prefix Cache Hit Rate: mean={s['mean']:.2%}")

    print(f"{'='*100}")


if __name__ == "__main__":
    main()
