#!/usr/bin/env python3
"""
Gauge Waveform Plotter
======================
Reads poll_samples from 03_concurrency_driver.json and generates
time-series waveform plots for num_requests_running / num_requests_waiting /
kv_cache_usage.

Usage:
    python script/03b_gauge_plot.py
    python script/03b_gauge_plot.py --input result/03_concurrency_driver.json
    python script/03b_gauge_plot.py --outdir result/plots

Output:
    result/plots/gauge_<metric>_conc<level>.png  (per-scenario)
    result/plots/gauge_<metric>_overview.png     (all scenarios stacked)
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── CLI ──
parser = argparse.ArgumentParser(description="Plot gauge waveforms from poll_samples")
parser.add_argument("--input", default=None,
                    help="Path to 03_concurrency_driver.json")
parser.add_argument("--outdir", default=None,
                    help="Output directory for plots")
args = parser.parse_args()

# Resolve paths
root = Path(__file__).resolve().parents[1]
in_path = Path(args.input) if args.input else root / "result" / "03_concurrency_driver.json"
out_dir = Path(args.outdir) if args.outdir else root / "result" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)

# ── Load data ──
with open(in_path) as f:
    data = json.load(f)

vm = data.get("vllm_metrics", {})
scenarios = sorted(vm.keys(), key=int)

# ── Metrics to plot ──
METRICS = [
    ("running",   "num_requests_running",   "#2563eb"),
    ("waiting",   "num_requests_waiting",   "#dc2626"),
    ("kv_cache",  "KV Cache Usage (%)",      "#16a34a"),
]

def get_samples(conc: str, key: str) -> tuple[list[float], list[float]]:
    """Extract (times_ms, values) for a given scenario and metric key."""
    samples = vm.get(conc, {}).get("poll_samples", [])
    if not samples:
        return [], []
    times = [s["t"] for s in samples]
    values = [s.get(key, 0) for s in samples]
    return times, values


# ── 1. Per-scenario individual plots ──
for conc in scenarios:
    samples = vm.get(conc, {}).get("poll_samples", [])
    if not samples:
        continue

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f"Concurrency = {conc}  ({len(samples)} samples, "
                 f"{samples[-1]['t']:.0f}ms duration)",
                 fontsize=14, fontweight="bold")

    for ax, (key, label, color) in zip(axes, METRICS):
        times, values = get_samples(conc, key)
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

        # Annotate max
        max_idx = np.argmax(v_arr)
        ax.annotate(f"max={v_arr[max_idx]:.0f}",
                     xy=(t_arr[max_idx], v_arr[max_idx]),
                     xytext=(10, 10), textcoords="offset points",
                     fontsize=9, color=color, fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color=color, lw=0.8))

    axes[-1].set_xlabel("Time (ms)", fontsize=10)
    plt.tight_layout()
    fname = out_dir / f"gauge_conc{conc}.png"
    fig.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  {fname}")


# ── 2. Overview: running across all scenarios ──
fig, axes = plt.subplots(len(scenarios), 1, figsize=(14, 2.5 * len(scenarios)),
                         sharex=False)
if len(scenarios) == 1:
    axes = [axes]

fig.suptitle("num_requests_running — All Concurrency Levels", fontsize=14, fontweight="bold")

for ax, conc in zip(axes, scenarios):
    times, values = get_samples(conc, "running")
    if not times:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="gray")
        ax.set_ylabel(f"Conc={conc}")
        continue

    t_arr = np.array(times)
    v_arr = np.array(values)

    # Normalize time to 0-100% for comparison
    t_norm = (t_arr - t_arr[0]) / (t_arr[-1] - t_arr[0]) * 100

    ax.fill_between(t_norm, v_arr, alpha=0.3, color="#2563eb")
    ax.plot(t_norm, v_arr, linewidth=0.5, color="#2563eb")
    ax.set_ylabel(f"Conc={conc}\n(max={max(values):.0f})", fontsize=9)
    ax.set_ylim(0, max(max(values) * 1.15, 10))
    ax.grid(True, alpha=0.3)

    # Add horizontal reference line at 100
    if max(values) > 90:
        ax.axhline(y=100, color="red", linestyle="--", alpha=0.4, linewidth=0.8)
        ax.text(1, 101, "cap=100", fontsize=7, color="red", alpha=0.6)

axes[-1].set_xlabel("Scenario Progress (%)", fontsize=10)
plt.tight_layout()
fname = out_dir / "gauge_running_overview.png"
fig.savefig(fname, dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"  {fname}")


# ── 3. Overview: waiting across all scenarios ──
fig, axes = plt.subplots(len(scenarios), 1, figsize=(14, 2 * len(scenarios)),
                         sharex=False)
if len(scenarios) == 1:
    axes = [axes]

fig.suptitle("num_requests_waiting — All Concurrency Levels", fontsize=14, fontweight="bold")

for ax, conc in zip(axes, scenarios):
    times, values = get_samples(conc, "waiting")
    if not times:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="gray")
        ax.set_ylabel(f"Conc={conc}")
        continue

    t_arr = np.array(times)
    v_arr = np.array(values)
    t_norm = (t_arr - t_arr[0]) / (t_arr[-1] - t_arr[0]) * 100

    ax.fill_between(t_norm, v_arr, alpha=0.3, color="#dc2626")
    ax.plot(t_norm, v_arr, linewidth=0.5, color="#dc2626")
    ax.set_ylabel(f"Conc={conc}", fontsize=9)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel("Scenario Progress (%)", fontsize=10)
plt.tight_layout()
fname = out_dir / "gauge_waiting_overview.png"
fig.savefig(fname, dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"  {fname}")


print(f"\n[03b_plot] Plots saved to {out_dir}/")
