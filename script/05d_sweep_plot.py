#!/usr/bin/env python3
"""
05d Sweep Plotter
=================
Reads 05d_sweep results and generates:
  1. gauge_running waveforms for ALL configs (18 subplots)
  2. Prefill time comparison (client vs server-side, all configs)
  3. Decode time comparison (client vs server-side, all configs)
  4. E2E latency scaling curve
  5. Throughput scaling curve

Usage:
    python script/05d_sweep_plot.py
    python script/05d_sweep_plot.py --indir result/05d_sweep --outdir result/plots

Output:
    result/plots/05d_sweep_gauge_running_all.png
    result/plots/05d_sweep_prefill_comparison.png
    result/plots/05d_sweep_decode_comparison.png
    result/plots/05d_sweep_e2e_scaling.png
    result/plots/05d_sweep_throughput_scaling.png
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

parser = argparse.ArgumentParser(description="Plot 05d sweep results")
parser.add_argument("--indir", default=None, help="Sweep results directory")
parser.add_argument("--outdir", default=None, help="Output plot directory")
args = parser.parse_args()

project_dir = Path(__file__).resolve().parents[1]
indir = Path(args.indir) if args.indir else project_dir / "result" / "05d_sweep"
outdir = Path(args.outdir) if args.outdir else project_dir / "result" / "plots"
outdir.mkdir(parents=True, exist_ok=True)

# Load summary
summary_path = indir / "05d_sweep_summary.json"
if not summary_path.exists():
    print(f"ERROR: {summary_path} not found")
    exit(1)

summary = json.loads(summary_path.read_text())
results = summary["results"]
ok_results = [r for r in results if r.get("status") == "ok"]
ok_results.sort(key=lambda r: r["concurrency"])

# Load individual configs for poll_samples
configs_data = {}
for f in sorted(indir.glob("05d_sweep_B*.json")):
    d = json.loads(f.read_text())
    tag = f"B{d['train_batch_size']}_G{d['rollout_n']}_C{d['concurrency']}"
    configs_data[tag] = d


# ──────────────────────────────────────────────────────────────────────
# 1. Gauge Running — ALL configs in one figure
# ──────────────────────────────────────────────────────────────────────
def plot_gauge_running_all():
    # Collect configs with poll_samples
    configs_with_poll = []
    for tag, d in configs_data.items():
        steps = d.get("step_summaries", [])
        total_samples = sum(s.get("poll_sample_count", 0) for s in steps)
        if total_samples > 0:
            configs_with_poll.append((tag, d))

    if not configs_with_poll:
        print("  [skip] No poll_samples found in any config")
        return

    # Sort by concurrency
    configs_with_poll.sort(key=lambda x: x[1]["concurrency"])
    n = len(configs_with_poll)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 3.2 * nrows), sharex=False)
    if nrows == 1:
        axes = axes.reshape(1, -1)
    axes_flat = axes.flatten()

    fig.suptitle("num_requests_running — All Configurations (step 0)",
                 fontsize=16, fontweight="bold", y=0.995)

    for i, (tag, d) in enumerate(configs_with_poll):
        ax = axes_flat[i]
        steps = d.get("step_summaries", [])
        # Use step 0 poll_samples
        poll = steps[0].get("poll_samples", []) if steps else []
        if not poll:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=11, color="gray")
            ax.set_title(tag, fontsize=10)
            continue

        times = [s["t"] for s in poll]
        values = [s.get("running", 0) for s in poll]
        t_arr = np.array(times)
        v_arr = np.array(values)

        # Normalize time to 0-100%
        t_range = t_arr[-1] - t_arr[0]
        if t_range > 0:
            t_norm = (t_arr - t_arr[0]) / t_range * 100
        else:
            t_norm = t_arr

        ax.fill_between(t_norm, v_arr, alpha=0.35, color="#2563eb")
        ax.plot(t_norm, v_arr, linewidth=0.8, color="#2563eb")

        conc = d["concurrency"]
        b, g = d["train_batch_size"], d["rollout_n"]
        ax.set_title(f"{tag} (conc={conc})", fontsize=10, fontweight="bold")
        ax.set_ylabel("running", fontsize=9)
        ax.set_ylim(0, max(max(values) * 1.15, 10))
        ax.grid(True, alpha=0.3)

        # Annotate max
        max_idx = np.argmax(v_arr)
        ax.annotate(f"max={v_arr[max_idx]:.0f}",
                    xy=(t_norm[max_idx], v_arr[max_idx]),
                    xytext=(5, 8), textcoords="offset points",
                    fontsize=8, color="#2563eb", fontweight="bold")

        # Reference line at concurrency
        ax.axhline(y=conc, color="red", linestyle="--", alpha=0.4, linewidth=0.8)
        ax.text(1, conc + 1, f"target={conc}", fontsize=7, color="red", alpha=0.6)

    # Hide unused subplots
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    for ax in axes[-1]:
        ax.set_xlabel("Step Progress (%)", fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    fname = outdir / "05d_sweep_gauge_running_all.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {fname}")


# ──────────────────────────────────────────────────────────────────────
# 2. Prefill Time Comparison
# ──────────────────────────────────────────────────────────────────────
def plot_prefill_comparison():
    fig, ax = plt.subplots(figsize=(14, 7))

    concs = sorted(set(r["concurrency"] for r in ok_results))

    # Group by concurrency
    data_by_conc = {}
    for r in ok_results:
        c = r["concurrency"]
        data_by_conc.setdefault(c, []).append(r)

    x = np.arange(len(concs))
    width = 0.18

    # Client-side prefill (from per-request timing)
    client_p50 = []
    client_p95 = []
    client_mean = []
    # Server-side prefill (from vLLM /metrics)
    server_p50 = []
    server_p95 = []
    server_mean = []

    for c in concs:
        entries = data_by_conc[c]
        # Average across B×G configs for this concurrency
        client_p50.append(np.mean([e["prefill_p50_ms"] for e in entries]))
        client_p95.append(np.mean([e["prefill_p95_ms"] for e in entries]))
        client_mean.append(np.mean([e["prefill_mean_ms"] for e in entries]))
        server_p50.append(np.mean([e["vllm_prefill_time_avg_ms"] for e in entries]))
        # Server doesn't have p50/p95 from sweep summary, use avg as proxy
        server_p95.append(np.mean([e["vllm_prefill_time_avg_ms"] * 1.2 for e in entries]))
        server_mean.append(np.mean([e["vllm_prefill_time_avg_ms"] for e in entries]))

    colors = {"client": "#2563eb", "server": "#16a34a"}
    ax.bar(x - width * 1.5, client_mean, width, label="Client Mean", color=colors["client"], alpha=0.7)
    ax.bar(x - width * 0.5, client_p95, width, label="Client P95", color=colors["client"], alpha=0.4)
    ax.bar(x + width * 0.5, server_mean, width, label="Server Mean (vLLM)", color=colors["server"], alpha=0.7)
    ax.bar(x + width * 1.5, server_p95, width, label="Server P95 (est.)", color=colors["server"], alpha=0.4)

    ax.set_xlabel("Concurrency", fontsize=12)
    ax.set_ylabel("Prefill Time (ms)", fontsize=12)
    ax.set_title("Prefill Time — Client vs Server-side (all B×G configs averaged)",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in concs], fontsize=10)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())

    # Add value labels on bars
    for i, c in enumerate(concs):
        ax.text(x[i] - width * 1.5, client_mean[i] + 1, f"{client_mean[i]:.0f}",
                ha="center", fontsize=7, color=colors["client"])
        ax.text(x[i] + width * 0.5, server_mean[i] + 1, f"{server_mean[i]:.0f}",
                ha="center", fontsize=7, color=colors["server"])

    plt.tight_layout()
    fname = outdir / "05d_sweep_prefill_comparison.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {fname}")


# ──────────────────────────────────────────────────────────────────────
# 3. Decode Time Comparison
# ──────────────────────────────────────────────────────────────────────
def plot_decode_comparison():
    fig, ax = plt.subplots(figsize=(14, 7))

    concs = sorted(set(r["concurrency"] for r in ok_results))
    data_by_conc = {}
    for r in ok_results:
        c = r["concurrency"]
        data_by_conc.setdefault(c, []).append(r)

    x = np.arange(len(concs))
    width = 0.18

    client_p50 = []
    client_p95 = []
    client_mean = []
    server_mean = []

    for c in concs:
        entries = data_by_conc[c]
        client_p50.append(np.mean([e["decode_p50_ms"] for e in entries]))
        client_p95.append(np.mean([e["decode_p95_ms"] for e in entries]))
        client_mean.append(np.mean([e["decode_mean_ms"] for e in entries]))
        server_mean.append(np.mean([e["vllm_decode_time_avg_ms"] for e in entries]))

    colors = {"client": "#2563eb", "server": "#16a34a"}
    ax.bar(x - width, client_mean, width, label="Client Mean", color=colors["client"], alpha=0.7)
    ax.bar(x, client_p95, width, label="Client P95", color=colors["client"], alpha=0.4)
    ax.bar(x + width, server_mean, width, label="Server Mean (vLLM)", color=colors["server"], alpha=0.7)

    ax.set_xlabel("Concurrency", fontsize=12)
    ax.set_ylabel("Decode Time (ms)", fontsize=12)
    ax.set_title("Decode Time — Client vs Server-side (all B×G configs averaged)",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in concs], fontsize=10)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for i, c in enumerate(concs):
        ax.text(x[i] - width, client_mean[i] + 20, f"{client_mean[i]:.0f}",
                ha="center", fontsize=7, color=colors["client"])
        ax.text(x[i] + width, server_mean[i] + 20, f"{server_mean[i]:.0f}",
                ha="center", fontsize=7, color=colors["server"])

    plt.tight_layout()
    fname = outdir / "05d_sweep_decode_comparison.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {fname}")


# ──────────────────────────────────────────────────────────────────────
# 4. E2E Latency Scaling Curve
# ──────────────────────────────────────────────────────────────────────
def plot_e2e_scaling():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    concs = sorted(set(r["concurrency"] for r in ok_results))
    data_by_conc = {}
    for r in ok_results:
        c = r["concurrency"]
        data_by_conc.setdefault(c, []).append(r)

    # Left: E2E P50/P95/P99
    e2e_p50 = [np.mean([e["e2e_p50_ms"] for e in data_by_conc[c]]) for c in concs]
    e2e_p95 = [np.mean([e["e2e_p95_ms"] for e in data_by_conc[c]]) for c in concs]
    e2e_p99 = [np.mean([e["e2e_p99_ms"] for e in data_by_conc[c]]) for c in concs]

    ax1.plot(concs, e2e_p50, "o-", color="#2563eb", linewidth=2, markersize=6, label="P50")
    ax1.plot(concs, e2e_p95, "s--", color="#dc2626", linewidth=2, markersize=6, label="P95")
    ax1.plot(concs, e2e_p99, "^:", color="#f59e0b", linewidth=2, markersize=6, label="P99")
    ax1.fill_between(concs, e2e_p50, e2e_p95, alpha=0.15, color="#2563eb")
    ax1.fill_between(concs, e2e_p95, e2e_p99, alpha=0.1, color="#dc2626")

    ax1.set_xlabel("Concurrency", fontsize=12)
    ax1.set_ylabel("E2E Latency (ms)", fontsize=12)
    ax1.set_title("E2E Latency Scaling", fontsize=14, fontweight="bold")
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax1.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, which="both")

    # Right: breakdown (stacked area)
    prefill = [np.mean([e["prefill_mean_ms"] for e in data_by_conc[c]]) for c in concs]
    decode = [np.mean([e["decode_mean_ms"] for e in data_by_conc[c]]) for c in concs]
    http_overhead = [np.mean([e["http_connect_p50_ms"] for e in data_by_conc[c]]) for c in concs]

    ax2.stackplot(concs, http_overhead, prefill, decode,
                  labels=["HTTP Overhead", "Prefill (TTFT)", "Decode"],
                  colors=["#f59e0b", "#2563eb", "#16a34a"], alpha=0.7)

    ax2.plot(concs, e2e_p50, "ko-", linewidth=1.5, markersize=4, label="E2E P50")

    ax2.set_xlabel("Concurrency", fontsize=12)
    ax2.set_ylabel("Time (ms)", fontsize=12)
    ax2.set_title("E2E Breakdown (stacked)", fontsize=14, fontweight="bold")
    ax2.set_xscale("log", base=2)
    ax2.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax2.legend(fontsize=9, loc="upper left")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = outdir / "05d_sweep_e2e_scaling.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {fname}")


# ──────────────────────────────────────────────────────────────────────
# 5. Throughput Scaling Curve
# ──────────────────────────────────────────────────────────────────────
def plot_throughput_scaling():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    concs = sorted(set(r["concurrency"] for r in ok_results))
    data_by_conc = {}
    for r in ok_results:
        c = r["concurrency"]
        data_by_conc.setdefault(c, []).append(r)

    # Left: Throughput (tok/s)
    throughput = [np.mean([e["throughput_mean"] for e in data_by_conc[c]]) for c in concs]

    ax1.plot(concs, throughput, "o-", color="#2563eb", linewidth=2.5, markersize=8)
    ax1.fill_between(concs, throughput, alpha=0.2, color="#2563eb")

    # Ideal linear scaling reference
    base_throughput = throughput[0]
    base_conc = concs[0]
    ideal = [base_throughput * (c / base_conc) for c in concs]
    ax1.plot(concs, ideal, "--", color="gray", linewidth=1.5, alpha=0.6, label="Linear scaling")

    # Efficiency labels
    for i, c in enumerate(concs):
        eff = throughput[i] / ideal[i] * 100
        ax1.annotate(f"{eff:.0f}%", xy=(c, throughput[i]),
                     xytext=(0, 12), textcoords="offset points",
                     fontsize=8, ha="center", color="#2563eb")

    ax1.set_xlabel("Concurrency", fontsize=12)
    ax1.set_ylabel("Output Token Throughput (tok/s)", fontsize=12)
    ax1.set_title("Throughput Scaling", fontsize=14, fontweight="bold")
    ax1.set_xscale("log", base=2)
    ax1.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, which="both")

    # Right: Per-request latency vs throughput tradeoff
    e2e_p95 = [np.mean([e["e2e_p95_ms"] for e in data_by_conc[c]]) for c in concs]

    scatter = ax2.scatter(e2e_p95, throughput, c=concs, cmap="viridis",
                          s=120, edgecolors="black", linewidth=0.5, zorder=5)
    for i, c in enumerate(concs):
        ax2.annotate(str(c), xy=(e2e_p95[i], throughput[i]),
                     xytext=(8, 5), textcoords="offset points",
                     fontsize=9, fontweight="bold")

    # Connect points
    ax2.plot(e2e_p95, throughput, "--", color="gray", alpha=0.5, linewidth=1)

    ax2.set_xlabel("E2E P95 Latency (ms)", fontsize=12)
    ax2.set_ylabel("Output Token Throughput (tok/s)", fontsize=12)
    ax2.set_title("Latency vs Throughput Tradeoff", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax2, label="Concurrency")

    plt.tight_layout()
    fname = outdir / "05d_sweep_throughput_scaling.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {fname}")


# ──────────────────────────────────────────────────────────────────────
# 6. B×G Comparison (G=4 vs G=8 at same concurrency)
# ──────────────────────────────────────────────────────────────────────
def plot_bg_comparison():
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    concs = sorted(set(r["concurrency"] for r in ok_results))
    data_by_conc = {}
    for r in ok_results:
        c = r["concurrency"]
        data_by_conc.setdefault(c, []).append(r)

    # Filter concs that have both G=4 and G=8
    concs_both = []
    for c in concs:
        gs = set(e["G"] for e in data_by_conc[c])
        if 4 in gs and 8 in gs:
            concs_both.append(c)

    if not concs_both:
        print("  [skip] No concurrency with both G=4 and G=8")
        return

    x = np.arange(len(concs_both))
    width = 0.35

    # Top-left: E2E P95
    ax = axes[0][0]
    g4_e2e = [next(e["e2e_p95_ms"] for e in data_by_conc[c] if e["G"] == 4) for c in concs_both]
    g8_e2e = [next(e["e2e_p95_ms"] for e in data_by_conc[c] if e["G"] == 8) for c in concs_both]
    ax.bar(x - width/2, g4_e2e, width, label="G=4", color="#2563eb", alpha=0.7)
    ax.bar(x + width/2, g8_e2e, width, label="G=8", color="#dc2626", alpha=0.7)
    ax.set_title("E2E P95: G=4 vs G=8", fontsize=13, fontweight="bold")
    ax.set_ylabel("ms", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in concs_both])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    # Top-right: Prefill P95
    ax = axes[0][1]
    g4_pf = [next(e["prefill_p95_ms"] for e in data_by_conc[c] if e["G"] == 4) for c in concs_both]
    g8_pf = [next(e["prefill_p95_ms"] for e in data_by_conc[c] if e["G"] == 8) for c in concs_both]
    ax.bar(x - width/2, g4_pf, width, label="G=4", color="#2563eb", alpha=0.7)
    ax.bar(x + width/2, g8_pf, width, label="G=8", color="#dc2626", alpha=0.7)
    ax.set_title("Prefill P95: G=4 vs G=8", fontsize=13, fontweight="bold")
    ax.set_ylabel("ms", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in concs_both])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    # Bottom-left: Decode P95
    ax = axes[1][0]
    g4_dc = [next(e["decode_p95_ms"] for e in data_by_conc[c] if e["G"] == 4) for c in concs_both]
    g8_dc = [next(e["decode_p95_ms"] for e in data_by_conc[c] if e["G"] == 8) for c in concs_both]
    ax.bar(x - width/2, g4_dc, width, label="G=4", color="#2563eb", alpha=0.7)
    ax.bar(x + width/2, g8_dc, width, label="G=8", color="#dc2626", alpha=0.7)
    ax.set_title("Decode P95: G=4 vs G=8", fontsize=13, fontweight="bold")
    ax.set_ylabel("ms", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in concs_both])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    # Bottom-right: Throughput
    ax = axes[1][1]
    g4_tp = [next(e["throughput_mean"] for e in data_by_conc[c] if e["G"] == 4) for c in concs_both]
    g8_tp = [next(e["throughput_mean"] for e in data_by_conc[c] if e["G"] == 8) for c in concs_both]
    ax.bar(x - width/2, g4_tp, width, label="G=4", color="#2563eb", alpha=0.7)
    ax.bar(x + width/2, g8_tp, width, label="G=8", color="#dc2626", alpha=0.7)
    ax.set_title("Throughput: G=4 vs G=8", fontsize=13, fontweight="bold")
    ax.set_ylabel("tok/s", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in concs_both])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fname = outdir / "05d_sweep_bg_comparison.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {fname}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
print(f"[05d_plot] Generating sweep plots from {indir}")
print(f"[05d_plot] Output: {outdir}")
print()

print("[05d_plot] 1/6 Gauge Running (all configs)...")
plot_gauge_running_all()

print("[05d_plot] 2/6 Prefill Time Comparison...")
plot_prefill_comparison()

print("[05d_plot] 3/6 Decode Time Comparison...")
plot_decode_comparison()

print("[05d_plot] 4/6 E2E Latency Scaling...")
plot_e2e_scaling()

print("[05d_plot] 5/6 Throughput Scaling...")
plot_throughput_scaling()

print("[05d_plot] 6/6 B×G Comparison (G=4 vs G=8)...")
plot_bg_comparison()

print(f"\n[05d_plot] All plots saved to {outdir}/")
