"""
04_metrics_compiler.py - Metric Consolidation & Analytics Pipeline

Reads raw JSON from 03_concurrency_driver.json.
Calculates P50, P95, P99 for all 5 latency parameters per concurrency tier.
Outputs:
  - ./result/04_metrics_compiler.json (structured analytics)
  - Appends markdown table to ./PHASE_1_SUMMARY.md
"""

import argparse
import json
import os
import statistics

METRIC_KEYS = [
    "client_serialization_ms",
    "server_overhead_ms",
    "gpu_prefill_ms",
    "gpu_decode_ms",
    "response_parsing_ms",
]


def percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100.0)
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_data) else f
    d = k - f
    return sorted_data[f] + d * (sorted_data[c] - sorted_data[f])


def compute_stats(values: list[float]) -> dict:
    clean = [v for v in values if v is not None]
    if not clean:
        return {"p50": 0, "p95": 0, "p99": 0, "mean": 0, "count": 0}
    return {
        "p50": round(percentile(clean, 50), 2),
        "p95": round(percentile(clean, 95), 2),
        "p99": round(percentile(clean, 99), 2),
        "mean": round(statistics.mean(clean), 2),
        "stdev": round(statistics.stdev(clean), 2) if len(clean) > 1 else 0,
        "count": len(clean),
    }


def analyze_tier(records: list[dict]) -> dict:
    tier_stats = {}
    for key in METRIC_KEYS:
        values = [r.get(key) for r in records if key in r and r.get(key) is not None]
        tier_stats[key] = compute_stats(values)
    total_values = [r.get("total_ms") for r in records if "total_ms" in r]
    tier_stats["total_ms"] = compute_stats(total_values)
    return tier_stats


def format_status(tier: int, stats: dict) -> str:
    overhead_p95 = stats.get("server_overhead_ms", {}).get("p95", 0)
    if tier <= 128:
        return "Nominal execution"
    if overhead_p95 < 50:
        return "Nominal execution"
    if overhead_p95 < 150:
        return "Initial Queue Contention"
    if overhead_p95 < 300:
        return "Context-Switch Thrashing"
    return "Server Vibe Breakdown"


def build_markdown_table(all_stats: dict) -> str:
    header = (
        "| Concurrency Level | Client Serialization (P95) | Server Overhead (P95) | "
        "GPU Prefill (P95) | GPU Decode (P95) | Response Parsing (P95) | Status / Degradation Source |\n"
    )
    sep = "| ----------------- | -------------------------- | --------------------- | " \
          "------------------ | ---------------- | ---------------------- | --------------------------- |\n"
    rows = ""
    for tier in sorted(all_stats.keys(), key=int):
        s = all_stats[tier]
        rows += (
            f"| **{tier}** | "
            f"{s['client_serialization_ms']['p95']}ms | "
            f"{s['server_overhead_ms']['p95']}ms | "
            f"{s['gpu_prefill_ms']['p95']}ms | "
            f"{s['gpu_decode_ms']['p95']}ms | "
            f"{s['response_parsing_ms']['p95']}ms | "
            f"{format_status(int(tier), s)} |\n"
        )
    return header + sep + rows


def main():
    parser = argparse.ArgumentParser(description="ConcurRL Metrics Compiler")
    parser.add_argument("--input", default="./result/03_concurrency_driver.json")
    parser.add_argument("--output", default="./result/04_metrics_compiler.json")
    parser.add_argument("--summary", default="./PHASE_1_SUMMARY.md")
    args = parser.parse_args()

    print("=" * 60)
    print("  ConcurRL Metrics Compiler")
    print("=" * 60)

    with open(args.input, "r") as f:
        raw = json.load(f)

    all_stats = {}
    for tier, records in raw.get("results", {}).items():
        valid = [r for r in records if "error" not in r]
        all_stats[tier] = analyze_tier(valid)
        print(f"  Tier {tier:>5s}: {len(valid):>5d} valid records")

    compiled = {"source": args.input, "tiers": all_stats}
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(compiled, f, indent=2)
    print(f"\n[+] Analytics written to {args.output}")

    md_table = build_markdown_table(all_stats)
    with open(args.summary, "w") as f:
        f.write("# Phase 1 Summary: Concurrency Scaling Validation\n\n")
        f.write("## Benchmark Results\n\n")
        f.write(md_table)
        f.write("\n")
    print(f"[+] Summary table written to {args.summary}")
    print("[*] Done.\n")


if __name__ == "__main__":
    main()
