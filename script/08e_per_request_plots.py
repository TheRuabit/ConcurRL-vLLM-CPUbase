#!/usr/bin/env python3
"""
Phase 2e Per-Request Scatter Plots
====================================
Generates per-request scatter plots from sweep data showing:
  1. Per-request E2E time distribution (scatter)
  2. Decode time breakdown by turn (stacked bar per request)
  3. Prefill vs Decode scatter per turn
  4. Per-request timing waterfall (Gantt-style)

Usage:
    python script/08e_per_request_plots.py
    python script/08e_per_request_plots.py --indir result/08_sweep --config C256_T8
    python script/08e_per_request_plots.py --indir result/08_sweep_varlen --config C256_T4

Output:
    result/plots/08e_per_request_<config>_scatter.png
    result/plots/08e_per_request_<config>_decode_breakdown.png
    result/plots/08e_per_request_<config>_waterfall.png
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

parser = argparse.ArgumentParser(description="Phase 2e per-request scatter plots")
parser.add_argument("--indir", default=None, help="Sweep results directory")
parser.add_argument("--config", default=None, help="Config tag to plot (e.g. C256_T8)")
parser.add_argument("--outdir", default=None, help="Output plot directory")
parser.add_argument("--batch", type=int, default=0, help="Batch index to plot")
args = parser.parse_args()

project_dir = Path(__file__).resolve().parents[1]
indir = Path(args.indir) if args.indir else project_dir / "result" / "08_sweep"
outdir = Path(args.outdir) if args.outdir else project_dir / "result" / "plots"
outdir.mkdir(parents=True, exist_ok=True)

# Find config to plot
if args.config:
    config_tag = args.config
else:
    # Pick highest concurrency config
    summary_path = indir / "08_sweep_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        ok = [r for r in summary["results"] if r.get("status") == "ok"]
        ok.sort(key=lambda r: r["concurrency"], reverse=True)
        config_tag = ok[0]["config_tag"] if ok else None
    else:
        config_tag = None

if not config_tag:
    print("ERROR: No config found")
    exit(1)

config_path = indir / config_tag / "08_tool_call_driver.json"
if not config_path.exists():
    print(f"ERROR: {config_path} not found")
    exit(1)

data = json.loads(config_path.read_text())
batch = data["batches"][args.batch]
rollouts = batch["rollout_results"]
config = data["config"]

print(f"Plotting per-request data for {config_tag} (batch {args.batch})")
print(f"  Concurrency: {config['concurrency']}, Turns: {config['num_turns']}")
print(f"  Rollouts: {len(rollouts)}")

# Collect all turn traces with rollout context
all_requests = []
for r in rollouts:
    for t in r.get("turn_traces", []):
        if t.get("success"):
            all_requests.append({
                "rollout_idx": r["rollout_idx"],
                "turn_idx": t["turn_idx"],
                "t_server_prefill_ms": t["t_server_prefill_ms"],
                "t_prefill_ms": t["t_prefill_ms"],
                "t_decode_ms": t["t_decode_ms"],
                "t_e2e_ms": t["t_e2e_ms"],
                "t_sem_wait_ms": t["t_sem_wait_ms"],
                "t_http_connect_ms": t["t_http_connect_ms"],
                "t_serialize_ms": t["t_serialize_ms"],
                "t_response_parse_ms": t["t_response_parse_ms"],
                "num_output_tokens": t["num_output_tokens"],
                "category": r.get("output_length_category", "fixed"),
            })

print(f"  Total requests: {len(all_requests)}")


# ──────────────────────────────────────────────────────────────────────
# 1. Per-Request E2E Scatter
# ──────────────────────────────────────────────────────────────────────
def plot_e2e_scatter():
    fig, ax = plt.subplots(figsize=(16, 8))

    num_turns = config["num_turns"]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, num_turns))

    for turn in range(num_turns):
        reqs = [r for r in all_requests if r["turn_idx"] == turn]
        if not reqs:
            continue
        x = [r["rollout_idx"] for r in reqs]
        y = [r["t_e2e_ms"] / 1000 for r in reqs]  # convert to seconds
        ax.scatter(x, y, s=12, alpha=0.5, color=colors[turn], label=f"Turn {turn}", zorder=3)

    ax.set_xlabel("Rollout Index", fontsize=12)
    ax.set_ylabel("E2E Latency (s)", fontsize=12)
    ax.set_title(f"Per-Request E2E Latency — {config_tag} (concurrency={config['concurrency']})",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, markerscale=2)
    ax.grid(True, alpha=0.3)

    fname = outdir / f"08e_per_request_{config_tag}_scatter.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {fname}")


# ──────────────────────────────────────────────────────────────────────
# 2. Decode Time Breakdown by Turn (per-request stacked bar)
# ──────────────────────────────────────────────────────────────────────
def plot_decode_breakdown():
    """For each rollout, show decode time per turn as stacked bars."""
    fig, ax = plt.subplots(figsize=(16, 8))

    num_turns = config["num_turns"]
    colors = plt.cm.tab10(np.linspace(0, 1, num_turns))

    # Sort rollouts by total E2E
    rollout_e2e = {}
    for r in rollouts:
        total = sum(t["t_decode_ms"] for t in r.get("turn_traces", []) if t.get("success"))
        rollout_e2e[r["rollout_idx"]] = total
    sorted_idx = sorted(rollout_e2e.keys(), key=lambda i: rollout_e2e[i])

    # Build stacked bars
    bottoms = np.zeros(len(sorted_idx))
    for turn in range(num_turns):
        values = []
        for ri in sorted_idx:
            req = next((r for r in rollouts if r["rollout_idx"] == ri), None)
            if req:
                turn_req = next((t for t in req.get("turn_traces", []) if t.get("turn_idx") == turn and t.get("success")), None)
                values.append(turn_req["t_decode_ms"] / 1000 if turn_req else 0)
            else:
                values.append(0)
        ax.bar(range(len(sorted_idx)), values, bottom=bottoms, width=1.0,
               color=colors[turn], label=f"Turn {turn} decode", alpha=0.85)
        bottoms += np.array(values)

    ax.set_xlabel("Rollout Index (sorted by total decode time)", fontsize=12)
    ax.set_ylabel("Decode Time (s)", fontsize=12)
    ax.set_title(f"Decode Time Breakdown per Rollout — {config_tag}",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")

    fname = outdir / f"08e_per_request_{config_tag}_decode_breakdown.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {fname}")


# ──────────────────────────────────────────────────────────────────────
# 3. Prefill vs Decode Scatter (per turn)
# ──────────────────────────────────────────────────────────────────────
def plot_prefill_vs_decode():
    """Scatter: x=prefill (TTFT), y=decode, color=turn."""
    fig, ax = plt.subplots(figsize=(12, 8))

    num_turns = config["num_turns"]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, num_turns))

    for turn in range(num_turns):
        reqs = [r for r in all_requests if r["turn_idx"] == turn]
        if not reqs:
            continue
        x = [r["t_server_prefill_ms"] / 1000 for r in reqs]
        y = [r["t_decode_ms"] / 1000 for r in reqs]
        ax.scatter(x, y, s=15, alpha=0.4, color=colors[turn], label=f"Turn {turn}", zorder=3)

    ax.set_xlabel("Prefill / TTFT (s)", fontsize=12)
    ax.set_ylabel("Decode Time (s)", fontsize=12)
    ax.set_title(f"Prefill vs Decode per Request — {config_tag}",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, markerscale=2)
    ax.grid(True, alpha=0.3)

    fname = outdir / f"08e_per_request_{config_tag}_prefill_vs_decode.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {fname}")


# ──────────────────────────────────────────────────────────────────────
# 4. Per-Request Waterfall (Gantt-style, top-N rollouts)
# ──────────────────────────────────────────────────────────────────────
def plot_waterfall():
    """Gantt chart showing prefill + decode for each turn of top-N rollouts."""
    top_n = min(30, len(rollouts))  # show top 30 rollouts

    # Sort by total E2E and pick top-N
    sorted_r = sorted(rollouts, key=lambda r: r["rollout_e2e_ms"], reverse=True)[:top_n]

    fig, ax = plt.subplots(figsize=(16, max(8, top_n * 0.4)))

    num_turns = config["num_turns"]
    prefill_color = "#2563eb"
    decode_color = "#dc2626"

    for i, r in enumerate(reversed(sorted_r)):
        y = i
        cum_offset = 0
        for turn in range(num_turns):
            t = next((t for t in r.get("turn_traces", []) if t.get("turn_idx") == turn and t.get("success")), None)
            if not t:
                continue
            prefill = t["t_server_prefill_ms"]
            decode = t["t_decode_ms"]

            # Prefill bar
            ax.barh(y, prefill / 1000, left=cum_offset / 1000, height=0.6,
                    color=prefill_color, alpha=0.7, edgecolor="none")
            cum_offset += prefill

            # Decode bar
            ax.barh(y, decode / 1000, left=cum_offset / 1000, height=0.6,
                    color=decode_color, alpha=0.7, edgecolor="none")
            cum_offset += decode

    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Rollout (sorted by E2E, top-N)", fontsize=12)
    ax.set_title(f"Per-Rollout Waterfall — {config_tag} (top {top_n})",
                 fontsize=14, fontweight="bold")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=prefill_color, alpha=0.7, label="Prefill (TTFT)"),
        Patch(facecolor=decode_color, alpha=0.7, label="Decode"),
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3, axis="x")

    fname = outdir / f"08e_per_request_{config_tag}_waterfall.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {fname}")


# ──────────────────────────────────────────────────────────────────────
# Generate all plots
# ──────────────────────────────────────────────────────────────────────
print(f"\n--- Plot 1: E2E Scatter ---")
plot_e2e_scatter()

print(f"\n--- Plot 2: Decode Breakdown ---")
plot_decode_breakdown()

print(f"\n--- Plot 3: Prefill vs Decode ---")
plot_prefill_vs_decode()

print(f"\n--- Plot 4: Waterfall ---")
plot_waterfall()

print(f"\n[OK] Per-request plots saved to {outdir}")
