#!/usr/bin/env python3
"""
Phase 2e Plot Generator
========================
Reads sweep results from 08b_tool_call_sweep.py and generates publication-
quality plots for multi-turn tool-calling benchmark analysis.

Plots:
  1. 08e_turn_latency_scaling.png    — TTFT per turn (lines per concurrency)
  2. 08e_prefix_cache_hit.png        — Cache hit ratio vs concurrency
  3. 08e_kv_cache_saturation.png     — Peak KV cache usage vs concurrency
  4. 08e_e2e_scaling.png             — E2E rollout latency vs concurrency
  5. 08e_throughput_scaling.png       — Output token throughput vs concurrency
  6. 08e_prefill_decode_ratio.png    — Prefill vs decode time per turn

Usage:
    python script/08d_tool_call_plot.py
    python script/08d_tool_call_plot.py --indir result/08_sweep --outdir result/plots

Output:
    result/plots/08e_*.png
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

parser = argparse.ArgumentParser(description="Phase 2e plot generator")
parser.add_argument("--indir", default=None, help="Sweep results directory")
parser.add_argument("--outdir", default=None, help="Output plot directory")
args = parser.parse_args()

project_dir = Path(__file__).resolve().parents[1]
indir = Path(args.indir) if args.indir else project_dir / "result" / "08_sweep"
outdir = Path(args.outdir) if args.outdir else project_dir / "result" / "plots"
outdir.mkdir(parents=True, exist_ok=True)

# Load summary
summary_path = indir / "08_sweep_summary.json"
if not summary_path.exists():
    print(f"ERROR: {summary_path} not found")
    exit(1)

summary = json.loads(summary_path.read_text())
results = summary["results"]
ok_results = [r for r in results if r.get("status") == "ok"]
ok_results.sort(key=lambda r: (r["concurrency"], r["num_turns"]))

if not ok_results:
    print("ERROR: No successful results to plot")
    exit(1)

# Load individual config data for detailed per-turn plots
configs_data = {}
for r in ok_results:
    tag = r["config_tag"]
    config_path = indir / tag / "08_tool_call_driver.json"
    if config_path.exists():
        configs_data[tag] = json.loads(config_path.read_text())

# Color palette
COLORS = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c",
          "#0891b2", "#4f46e5", "#c026d3"]


# ──────────────────────────────────────────────────────────────────────
# 1. Turn Latency Scaling — TTFT per turn
# ──────────────────────────────────────────────────────────────────────
def plot_turn_latency_scaling():
    """TTFT per turn, one line per concurrency level."""
    # Group by concurrency
    by_conc = defaultdict(list)
    for r in ok_results:
        by_conc[r["concurrency"]].append(r)

    fig, ax = plt.subplots(figsize=(12, 7))

    for i, conc in enumerate(sorted(by_conc.keys())):
        configs = by_conc[conc]
        # Pick the config with the most turns for this concurrency
        best = max(configs, key=lambda c: c["num_turns"])
        ttft_data = best.get("per_turn_ttft", {})
        if not ttft_data:
            continue

        turns = sorted(ttft_data.keys(), key=int)
        ttft_p50 = [ttft_data[t].get("p50", 0) if isinstance(ttft_data[t], dict) else 0 for t in turns]

        ax.plot(
            [int(t) for t in turns], ttft_p50,
            marker="o", linewidth=2, markersize=6,
            color=COLORS[i % len(COLORS)],
            label=f"C={conc}",
        )

    ax.set_xlabel("Turn Index", fontsize=12)
    ax.set_ylabel("TTFT P50 (ms)", fontsize=12)
    ax.set_title("Time-to-First-Token per Turn (Prefill Scaling)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    fname = outdir / "08e_turn_latency_scaling.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {fname}")


# ──────────────────────────────────────────────────────────────────────
# 2. E2E Scaling — Rollout E2E latency vs concurrency
# ──────────────────────────────────────────────────────────────────────
def plot_e2e_scaling():
    """E2E rollout latency vs concurrency, one line per turn count."""
    by_turns = defaultdict(list)
    for r in ok_results:
        by_turns[r["num_turns"]].append(r)

    fig, ax = plt.subplots(figsize=(12, 7))

    for i, turns in enumerate(sorted(by_turns.keys())):
        configs = sorted(by_turns[turns], key=lambda c: c["concurrency"])
        concs = [c["concurrency"] for c in configs]
        e2e_p50 = [c.get("rollout_e2e_p50", 0) for c in configs]
        e2e_p95 = [c.get("rollout_e2e_p95", 0) for c in configs]

        color = COLORS[i % len(COLORS)]
        ax.plot(concs, e2e_p50, marker="o", linewidth=2, markersize=6,
                color=color, label=f"T={turns} P50")
        ax.plot(concs, e2e_p95, marker="s", linewidth=1.5, markersize=5,
                color=color, linestyle="--", alpha=0.6, label=f"T={turns} P95")

    ax.set_xlabel("Concurrency (parallel rollouts)", fontsize=12)
    ax.set_ylabel("Rollout E2E Latency (ms)", fontsize=12)
    ax.set_title("End-to-End Rollout Latency Scaling", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

    fname = outdir / "08e_e2e_scaling.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {fname}")


# ──────────────────────────────────────────────────────────────────────
# 3. Per-Turn TTFT Heatmap
# ──────────────────────────────────────────────────────────────────────
def plot_turn_ttft_heatmap():
    """Heatmap of TTFT P50 per (concurrency, turn) cell."""
    # Collect all unique concurrencies and turn counts
    all_concs = sorted(set(r["concurrency"] for r in ok_results))
    all_turns = sorted(set(r["num_turns"] for r in ok_results))

    # Build matrix: rows = concurrency, cols = turn index
    max_turns = max(all_turns)
    matrix = np.full((len(all_concs), max_turns), np.nan)

    for r in ok_results:
        ci = all_concs.index(r["concurrency"])
        ttft_data = r.get("per_turn_ttft", {})
        for tk, vals in ttft_data.items():
            ti = int(tk)
            if ti < max_turns and isinstance(vals, dict):
                matrix[ci, ti] = vals.get("p50", np.nan)

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest")

    ax.set_yticks(range(len(all_concs)))
    ax.set_yticklabels([str(c) for c in all_concs])
    ax.set_xticks(range(max_turns))
    ax.set_xticklabels([str(t) for t in range(max_turns)])
    ax.set_xlabel("Turn Index", fontsize=12)
    ax.set_ylabel("Concurrency", fontsize=12)
    ax.set_title("TTFT P50 (ms) — Concurrency × Turn Heatmap", fontsize=14, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("TTFT P50 (ms)", fontsize=10)

    # Annotate cells
    for i in range(len(all_concs)):
        for j in range(max_turns):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                        fontsize=7, color="black" if val < np.nanmax(matrix) * 0.7 else "white")

    fname = outdir / "08e_turn_ttft_heatmap.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {fname}")


# ──────────────────────────────────────────────────────────────────────
# 4. KV Cache Saturation
# ──────────────────────────────────────────────────────────────────────
def plot_kv_cache_saturation():
    """Peak KV cache utilization vs concurrency from poll_samples."""
    fig, ax = plt.subplots(figsize=(12, 7))

    for tag, data in sorted(configs_data.items()):
        conc = data["config"]["concurrency"]
        turns = data["config"]["num_turns"]
        batches = data.get("batches", [])
        if not batches:
            continue

        # Collect peak kv_cache from poll_samples across batches
        peak_values = []
        for b in batches:
            samples = b.get("poll_samples", [])
            if samples:
                kv_values = [s.get("kv_cache", 0) for s in samples]
                peak_values.append(max(kv_values) if kv_values else 0)

        if peak_values:
            avg_peak = np.mean(peak_values) * 100  # convert to percentage
            ax.scatter(conc, avg_peak, s=80, marker="o",
                      color=COLORS[list(configs_data.keys()).index(tag) % len(COLORS)],
                      zorder=5)
            ax.annotate(f"T={turns}", (conc, avg_peak),
                       textcoords="offset points", xytext=(8, 4), fontsize=8)

    ax.set_xlabel("Concurrency", fontsize=12)
    ax.set_ylabel("Peak KV Cache Utilization (%)", fontsize=12)
    ax.set_title("KV Cache Saturation Curve", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_ylim(0, 105)

    fname = outdir / "08e_kv_cache_saturation.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {fname}")


# ──────────────────────────────────────────────────────────────────────
# 5. Prefill vs Decode per Turn
# ──────────────────────────────────────────────────────────────────────
def plot_prefill_decode_ratio():
    """Stacked bar chart of prefill vs decode time per turn."""
    # Pick the config with highest concurrency and most turns
    if not configs_data:
        print("  [skip] No detailed config data for prefill/decode plot")
        return

    # Find config with most turns
    best_tag = max(configs_data.keys(),
                   key=lambda t: configs_data[t]["config"]["num_turns"])
    data = configs_data[best_tag]
    batches = data.get("batches", [])
    if not batches:
        return

    # Aggregate per-turn prefill and decode across rollouts in first batch
    batch = batches[0]
    rollout_results = batch.get("rollout_results", [])

    per_turn_prefill = defaultdict(list)
    per_turn_decode = defaultdict(list)

    for rollout in rollout_results:
        for t in rollout.get("turn_traces", []):
            if t.get("success"):
                per_turn_prefill[t["turn_idx"]].append(t["t_server_prefill_ms"])
                per_turn_decode[t["turn_idx"]].append(t["t_decode_ms"])

    turns = sorted(per_turn_prefill.keys())
    if not turns:
        return

    prefill_means = [np.mean(per_turn_prefill[t]) for t in turns]
    decode_means = [np.mean(per_turn_decode[t]) for t in turns]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(turns))
    width = 0.35

    bars1 = ax.bar(x - width/2, prefill_means, width, label="Prefill (TTFT)",
                   color="#2563eb", alpha=0.8)
    bars2 = ax.bar(x + width/2, decode_means, width, label="Decode",
                   color="#dc2626", alpha=0.8)

    ax.set_xlabel("Turn Index", fontsize=12)
    ax.set_ylabel("Latency P50 (ms)", fontsize=12)
    ax.set_title(f"Prefill vs Decode per Turn ({best_tag})", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in turns])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    fname = outdir / "08e_prefill_decode_ratio.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {fname}")


# ──────────────────────────────────────────────────────────────────────
# 6. Throughput Scaling
# ──────────────────────────────────────────────────────────────────────
def plot_throughput_scaling():
    """Output token throughput vs concurrency."""
    # Load per-config data for throughput
    by_turns = defaultdict(list)
    for tag, data in configs_data.items():
        conc = data["config"]["concurrency"]
        turns = data["config"]["num_turns"]
        batches = data.get("batches", [])
        if not batches:
            continue
        throughputs = []
        for b in batches:
            s = b.get("summary", {})
            tp = s.get("output_token_throughput", 0)
            if tp > 0:
                throughputs.append(tp)
        if throughputs:
            by_turns[turns].append((conc, np.mean(throughputs)))

    if not by_turns:
        print("  [skip] No throughput data available")
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    for i, turns in enumerate(sorted(by_turns.keys())):
        entries = sorted(by_turns[turns], key=lambda x: x[0])
        concs = [e[0] for e in entries]
        tputs = [e[1] for e in entries]
        ax.plot(concs, tputs, marker="o", linewidth=2, markersize=6,
                color=COLORS[i % len(COLORS)], label=f"T={turns}")

    ax.set_xlabel("Concurrency (parallel rollouts)", fontsize=12)
    ax.set_ylabel("Output Token Throughput (tok/s)", fontsize=12)
    ax.set_title("Output Token Throughput Scaling", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

    fname = outdir / "08e_throughput_scaling.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {fname}")


# ──────────────────────────────────────────────────────────────────────
# Generate all plots
# ──────────────────────────────────────────────────────────────────────
print("=" * 60)
print(" Phase 2e Plot Generator")
print("=" * 60)

print("\n--- Plot 1: Turn Latency Scaling ---")
plot_turn_latency_scaling()

print("\n--- Plot 2: E2E Scaling ---")
plot_e2e_scaling()

print("\n--- Plot 3: TTFT Heatmap ---")
plot_turn_ttft_heatmap()

print("\n--- Plot 4: KV Cache Saturation ---")
plot_kv_cache_saturation()

print("\n--- Plot 5: Prefill vs Decode ---")
plot_prefill_decode_ratio()

print("\n--- Plot 6: Throughput Scaling ---")
plot_throughput_scaling()

print(f"\n[OK] All plots saved to {outdir}")
