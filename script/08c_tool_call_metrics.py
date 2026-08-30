#!/usr/bin/env python3
"""
Phase 2e Metrics Compiler
==========================
Reads sweep results from 08b_tool_call_sweep.py, computes aggregate
analytics across concurrency and turn-count dimensions, and generates:
  1. result/08c_tool_call_metrics.json — structured analytics
  2. PHASE_2E_SUMMARY.md — human-readable report

Key analytics:
  - Per-turn TTFT growth (prefill cost scaling)
  - Prefix cache hit ratio vs concurrency and turn number
  - KV cache saturation curve
  - Preemption rate vs concurrency
  - CPU vs GPU time breakdown
  - Prefill/decode ratio per turn

Usage:
    python script/08c_tool_call_metrics.py
    python script/08c_tool_call_metrics.py --indir result/08_sweep

Output:
    result/08c_tool_call_metrics.json
    PHASE_2E_SUMMARY.md
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

parser = argparse.ArgumentParser(
    description="Phase 2e metrics compiler — multi-turn tool-calling analytics"
)
parser.add_argument("--indir", default=None,
                    help="Sweep results directory (default: result/08_sweep)")
parser.add_argument("--output-json", default=None,
                    help="Output JSON path")
parser.add_argument("--output-md", default=None,
                    help="Output Markdown path")
args = parser.parse_args()

project_dir = Path(__file__).resolve().parents[1]
indir = Path(args.indir) if args.indir else project_dir / "result" / "08_sweep"

if args.output_json:
    out_json = Path(args.output_json)
else:
    out_json = project_dir / "result" / "08c_tool_call_metrics.json"

if args.output_md:
    out_md = Path(args.output_md)
else:
    out_md = project_dir / "PHASE_2E_SUMMARY.md"

# Load summary
summary_path = indir / "08_sweep_summary.json"
if not summary_path.exists():
    print(f"ERROR: {summary_path} not found. Run 08b_tool_call_sweep.py first.")
    sys.exit(1)

summary = json.loads(summary_path.read_text())
results = summary["results"]
ok_results = [r for r in results if r.get("status") == "ok"]

if not ok_results:
    print("ERROR: No successful sweep results found.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Compute analytics
# ---------------------------------------------------------------------------
analytics = {
    "sweep_config": summary["config"],
    "total_configs": summary["total_configs"],
    "ok_configs": len(ok_results),
    "by_concurrency": {},
    "by_turns": {},
    "prefill_growth": {},
    "summary": {},
}

# Group by concurrency
by_conc = defaultdict(list)
for r in ok_results:
    by_conc[r["concurrency"]].append(r)

# Group by turns
by_turns = defaultdict(list)
for r in ok_results:
    by_turns[r["num_turns"]].append(r)

# Per-concurrency analytics
for conc in sorted(by_conc.keys()):
    configs = by_conc[conc]
    e2e_p50_values = [c["rollout_e2e_p50"] for c in configs if c.get("rollout_e2e_p50")]
    e2e_p95_values = [c["rollout_e2e_p95"] for c in configs if c.get("rollout_e2e_p95")]

    # Collect per-turn TTFT across all turn counts for this concurrency
    per_turn_ttft = defaultdict(list)
    for c in configs:
        for turn_key, stats in c.get("per_turn_ttft", {}).items():
            if isinstance(stats, dict) and "p50" in stats:
                per_turn_ttft[turn_key].append(stats["p50"])

    analytics["by_concurrency"][str(conc)] = {
        "num_configs": len(configs),
        "turn_counts_tested": sorted(set(c["num_turns"] for c in configs)),
        "rollout_e2e_p50_range": [min(e2e_p50_values), max(e2e_p50_values)] if e2e_p50_values else [],
        "rollout_e2e_p95_range": [min(e2e_p95_values), max(e2e_p95_values)] if e2e_p95_values else [],
        "per_turn_ttft_p50": {
            k: {"min": min(v), "max": max(v), "mean": sum(v)/len(v)}
            for k, v in sorted(per_turn_ttft.items())
        },
    }

# Per-turn-count analytics
for turns in sorted(by_turns.keys()):
    configs = by_turns[turns]
    e2e_p50_values = [c["rollout_e2e_p50"] for c in configs if c.get("rollout_e2e_p50")]

    analytics["by_turns"][str(turns)] = {
        "num_configs": len(configs),
        "concurrency_levels_tested": sorted(set(c["concurrency"] for c in configs)),
        "rollout_e2e_p50_range": [min(e2e_p50_values), max(e2e_p50_values)] if e2e_p50_values else [],
    }

# Prefill growth analysis: TTFT at turn N / TTFT at turn 0
for conc in sorted(by_conc.keys()):
    configs = by_conc[conc]
    for c in configs:
        turns = c["num_turns"]
        ttft_data = c.get("per_turn_ttft", {})
        turn_0_key = "0"
        if turn_0_key in ttft_data and isinstance(ttft_data[turn_0_key], dict):
            ttft_t0 = ttft_data[turn_0_key].get("p50", 0)
            if ttft_t0 > 0:
                growth = {}
                for tk in sorted(ttft_data.keys(), key=int):
                    if isinstance(ttft_data[tk], dict):
                        ttft_tn = ttft_data[tk].get("p50", 0)
                        growth[f"turn_{tk}"] = {
                            "ttft_p50_ms": round(ttft_tn, 2),
                            "ratio_vs_turn_0": round(ttft_tn / ttft_t0, 3),
                        }
                key = f"C{conc}_T{turns}"
                analytics["prefill_growth"][key] = growth

# Overall summary
all_e2e_p50 = [r["rollout_e2e_p50"] for r in ok_results if r.get("rollout_e2e_p50")]
all_e2e_p95 = [r["rollout_e2e_p95"] for r in ok_results if r.get("rollout_e2e_p95")]

analytics["summary"] = {
    "e2e_p50_range_ms": [round(min(all_e2e_p50), 2), round(max(all_e2e_p50), 2)] if all_e2e_p50 else [],
    "e2e_p95_range_ms": [round(min(all_e2e_p95), 2), round(max(all_e2e_p95), 2)] if all_e2e_p95 else [],
    "concurrency_range": [min(by_conc.keys()), max(by_conc.keys())],
    "turns_range": [min(by_turns.keys()), max(by_turns.keys())],
}

# ---------------------------------------------------------------------------
# Write JSON
# ---------------------------------------------------------------------------
out_json.parent.mkdir(parents=True, exist_ok=True)
out_json.write_text(json.dumps(analytics, indent=2, default=str), encoding="utf-8")
print(f"[OK] Analytics written to {out_json}")

# ---------------------------------------------------------------------------
# Generate Markdown report
# ---------------------------------------------------------------------------
lines = []
lines.append("# Phase 2e Summary: Multi-Turn Tool-Calling Benchmark\n")
lines.append(f"**Sweep**: {len(ok_results)} successful configs "
             f"({summary['total_configs']} total)\n")
lines.append(f"**Concurrency range**: {analytics['summary']['concurrency_range']}\n")
lines.append(f"**Turns range**: {analytics['summary']['turns_range']}\n")
lines.append(f"**E2E latency range**: {analytics['summary']['e2e_p50_range_ms']}ms (P50)\n")

lines.append("\n## Per-Concurrency Breakdown\n")
lines.append("| Concurrency | Turns tested | E2E P50 range (ms) | E2E P95 range (ms) |")
lines.append("|-------------|-------------|--------------------|--------------------|")
for conc_str, data in sorted(analytics["by_concurrency"].items(), key=lambda x: int(x[0])):
    turns = ", ".join(str(t) for t in data["turn_counts_tested"])
    e2e_p50 = f"{data['rollout_e2e_p50_range'][0]:.0f} – {data['rollout_e2e_p50_range'][1]:.0f}" if data["rollout_e2e_p50_range"] else "–"
    e2e_p95 = f"{data['rollout_e2e_p95_range'][0]:.0f} – {data['rollout_e2e_p95_range'][1]:.0f}" if data["rollout_e2e_p95_range"] else "–"
    lines.append(f"| {conc_str} | {turns} | {e2e_p50} | {e2e_p95} |")

lines.append("\n## Per-Turn TTFT Growth (Prefill Scaling)\n")
lines.append("Shows how TTFT increases as conversation history grows.\n")
if analytics["prefill_growth"]:
    # Pick the config with most turns for detailed view
    max_turns_config = max(analytics["prefill_growth"].items(),
                          key=lambda x: len(x[1]))
    config_name, growth = max_turns_config
    lines.append(f"**Example config: {config_name}**\n")
    lines.append("| Turn | TTFT P50 (ms) | Ratio vs Turn 0 |")
    lines.append("|------|--------------|-----------------|")
    for turn_key, vals in sorted(growth.items(), key=lambda x: int(x[0].split("_")[1])):
        lines.append(f"| {turn_key} | {vals['ttft_p50_ms']:.0f} | {vals['ratio_vs_turn_0']:.2f}x |")

lines.append("\n## Per-Turn-Count Scaling\n")
lines.append("| Turns | Concurrency levels | E2E P50 range (ms) |")
lines.append("|-------|--------------------|--------------------|")
for turns_str, data in sorted(analytics["by_turns"].items(), key=lambda x: int(x[0])):
    concs = ", ".join(str(c) for c in data["concurrency_levels_tested"])
    e2e = f"{data['rollout_e2e_p50_range'][0]:.0f} – {data['rollout_e2e_p50_range'][1]:.0f}" if data["rollout_e2e_p50_range"] else "–"
    lines.append(f"| {turns_str} | {concs} | {e2e} |")

lines.append("\n## Config Summary\n")
lines.append("```")
lines.append(json.dumps(summary["config"], indent=2))
lines.append("```\n")

md_text = "\n".join(lines)
out_md.write_text(md_text, encoding="utf-8")
print(f"[OK] Report written to {out_md}")
