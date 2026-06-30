#!/usr/bin/env python3
"""
Phase 2 Metrics Compiler
=========================
Reads raw timing data from 05_grpo_train.json (and optionally 06_reward_timings.json),
computes per-stage statistics, derives CPU vs GPU breakdown, and generates:
  1. result/07_phase2_metrics.json — structured analytics
  2. PHASE_2_SUMMARY.md — human-readable report

Usage:
    python script/07_phase2_metrics.py
    python script/07_phase2_metrics.py --input result/05_grpo_train.json

Output:
    result/07_phase2_metrics.json
    PHASE_2_SUMMARY.md
"""

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Parse CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Phase 2 metrics compiler — GRPO RL loop timing analysis"
)
parser.add_argument("--input", default=None,
                    help="Path to 05_grpo_train.json")
parser.add_argument("--reward-timings", default=None,
                    help="Path to 06_reward_timings.json")
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
    in_path = project_dir / "result" / "05_grpo_train.json"

if args.reward_timings:
    reward_path = Path(args.reward_timings)
else:
    reward_path = project_dir / "result" / "06_reward_timings.json"

if args.output_json:
    out_json = Path(args.output_json)
else:
    out_json = project_dir / "result" / "07_phase2_metrics.json"

if args.output_md:
    out_md = Path(args.output_md)
else:
    out_md = project_dir / "PHASE_2_SUMMARY.md"

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
    print(f"[07_compiler] Phase 2 Metrics Compiler")
    print(f"[07_compiler] Input:  {in_path}")

    if not in_path.exists():
        print(f"[07_compiler] ERROR: Input file not found: {in_path}")
        print(f"[07_compiler] Run 05_grpo_train.py first")
        sys.exit(1)

    data = json.loads(in_path.read_text(encoding="utf-8"))
    step_timings = data.get("step_timings", [])

    if not step_timings:
        print(f"[07_compiler] ERROR: No step timings in input file")
        sys.exit(1)

    model = data.get("model", "unknown")
    rollout_n = data.get("rollout_n", 0)
    train_batch_size = data.get("train_batch_size", 0)
    num_epochs = data.get("num_epochs", 0)
    total_elapsed = data.get("total_elapsed_s", 0)

    print(f"[07_compiler] Model:  {model}")
    print(f"[07_compiler] Steps:  {len(step_timings)}")

    # -------------------------------------------------------------------
    # Compute per-stage statistics
    # -------------------------------------------------------------------
    # Collect all stage names across steps
    all_stages = set()
    for step in step_timings:
        all_stages.update(step.keys())

    stage_stats = {}
    for stage in sorted(all_stages):
        values = [step[stage] for step in step_timings if stage in step]
        if values:
            stage_stats[stage] = compute_full_stats(values)

    # -------------------------------------------------------------------
    # Derive CPU vs GPU breakdown
    # -------------------------------------------------------------------
    # CPU stages: log_prob computation, advantage (pure CPU work)
    cpu_stages = ["t_old_log_prob_s", "t_ref_log_prob_s", "t_adv_s"]
    # GPU stages: actor update (gradient computation + optimizer)
    gpu_stages = ["t_update_actor_s"]
    # Rollout stages: generation (vLLM GPU + CPU scheduling)
    rollout_stages = ["t_gen_s"]
    # Weight sync: parameter transfer
    sync_stages = ["t_update_weights_s"]

    def sum_stages(step: dict, stage_list: list[str]) -> float:
        return sum(step.get(s, 0) for s in stage_list)

    derived = []
    for step in step_timings:
        cpu_s = sum_stages(step, cpu_stages)
        gpu_s = sum_stages(step, gpu_stages)
        rollout_s = sum_stages(step, rollout_stages)
        sync_s = sum_stages(step, sync_stages)
        step_s = step.get("t_step_s", cpu_s + gpu_s + rollout_s + sync_s)

        derived.append({
            "cpu_overhead_s": round(cpu_s, 4),
            "gpu_train_s": round(gpu_s, 4),
            "rollout_s": round(rollout_s, 4),
            "weight_sync_s": round(sync_s, 4),
            "step_total_s": round(step_s, 4),
            "cpu_percent": round((cpu_s / step_s * 100) if step_s > 0 else 0, 2),
            "gpu_percent": round((gpu_s / step_s * 100) if step_s > 0 else 0, 2),
            "rollout_percent": round((rollout_s / step_s * 100) if step_s > 0 else 0, 2),
            "sync_percent": round((sync_s / step_s * 100) if step_s > 0 else 0, 2),
        })

    # Aggregate derived stats
    derived_stats = {}
    for key in ["cpu_overhead_s", "gpu_train_s", "rollout_s", "weight_sync_s",
                 "step_total_s", "cpu_percent", "gpu_percent", "rollout_percent", "sync_percent"]:
        values = [d[key] for d in derived]
        derived_stats[key] = compute_full_stats(values)

    # -------------------------------------------------------------------
    # Load reward timings (optional)
    # -------------------------------------------------------------------
    reward_stats = {}
    if reward_path.exists():
        reward_timings = json.loads(reward_path.read_text(encoding="utf-8"))
        if reward_timings:
            reward_stats = {
                "t_extract_ms": compute_full_stats([t["t_extract_ms"] for t in reward_timings]),
                "t_compare_ms": compute_full_stats([t["t_compare_ms"] for t in reward_timings]),
                "t_total_ms": compute_full_stats([t["t_total_ms"] for t in reward_timings]),
                "accuracy": round(
                    sum(1 for t in reward_timings if t.get("correct", False)) / len(reward_timings),
                    4,
                ),
                "num_evaluations": len(reward_timings),
            }
            print(f"[07_compiler] Reward timings: {len(reward_timings)} evaluations")

    # -------------------------------------------------------------------
    # Save JSON
    # -------------------------------------------------------------------
    output_json = {
        "benchmark": "phase2_grpo_metrics",
        "model": model,
        "rollout_n": rollout_n,
        "train_batch_size": train_batch_size,
        "num_epochs": num_epochs,
        "num_steps": len(step_timings),
        "total_elapsed_s": total_elapsed,
        "stage_stats": stage_stats,
        "derived_stats": derived_stats,
        "step_details": derived,
        "reward_stats": reward_stats,
    }

    out_json.write_text(json.dumps(output_json, indent=2, ensure_ascii=False))
    print(f"[07_compiler] JSON saved to {out_json}")

    # -------------------------------------------------------------------
    # Generate Markdown
    # -------------------------------------------------------------------
    lines = []
    lines.append("# Phase 2: GRPO RL Loop — Timing Breakdown\n")
    lines.append(f"**Model:** {model}  ")
    lines.append(f"**Rollout N:** {rollout_n}  ")
    lines.append(f"**Train Batch Size:** {train_batch_size}  ")
    lines.append(f"**Epochs:** {num_epochs}  ")
    lines.append(f"**Steps:** {len(step_timings)}  ")
    lines.append(f"**Total Time:** {total_elapsed:.1f}s\n")

    # Per-stage timing table
    lines.append("## Per-Stage Timing (seconds)\n")
    lines.append("| Stage | Mean | P50 | P95 | Min | Max |")
    lines.append("| --- | --- | --- | --- | --- | --- |")

    stage_labels = {
        "t_step_s": "Step E2E",
        "t_gen_s": "Rollout (gen)",
        "t_reward_s": "Reward",
        "t_old_log_prob_s": "Old Log Prob",
        "t_ref_log_prob_s": "Ref Log Prob",
        "t_adv_s": "Advantage",
        "t_update_actor_s": "Actor Update",
        "t_update_weights_s": "Weight Sync",
    }

    for stage_key, label in stage_labels.items():
        stats = stage_stats.get(stage_key)
        if not stats:
            continue
        lines.append(
            f"| {label} | "
            f"{stats['mean']:.3f} | {stats['p50']:.3f} | "
            f"{stats['p95']:.3f} | {stats['min']:.3f} | {stats['max']:.3f} |"
        )

    # CPU vs GPU breakdown
    lines.append("\n## CPU vs GPU Breakdown\n")
    lines.append("| Component | Mean (s) | Mean (%) | P95 (s) |")
    lines.append("| --- | --- | --- | --- |")

    breakdown_labels = {
        "rollout_s": "Rollout (vLLM GPU + CPU scheduling)",
        "cpu_overhead_s": "CPU Overhead (log_prob + advantage)",
        "gpu_train_s": "GPU Train (actor update)",
        "weight_sync_s": "Weight Sync (FSDP → vLLM)",
        "step_total_s": "Step Total",
    }

    for key, label in breakdown_labels.items():
        stats = derived_stats.get(key)
        if not stats:
            continue
        if key == "step_total_s":
            lines.append(
                f"| **{label}** | "
                f"**{stats['mean']:.3f}** | **100.0%** | "
                f"**{stats['p95']:.3f}** |"
            )
        else:
            pct_key = key.replace("_s", "_percent")
            pct_stats = derived_stats.get(pct_key, {})
            pct_mean = pct_stats.get("mean", 0)
            lines.append(
                f"| {label} | "
                f"{stats['mean']:.3f} | {pct_mean:.1f}% | "
                f"{stats['p95']:.3f} |"
            )

    # Reward statistics (if available)
    if reward_stats:
        lines.append("\n## Reward Function Statistics\n")
        lines.append("| Metric | Mean | P50 | P95 | Min | Max |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for key, label in [
            ("t_extract_ms", "Answer Extraction"),
            ("t_compare_ms", "Answer Comparison"),
            ("t_total_ms", "Total Reward Compute"),
        ]:
            st = reward_stats.get(key, {})
            if not st:
                continue
            lines.append(
                f"| {label} | "
                f"{st['mean']:.3f}ms | {st['p50']:.3f}ms | "
                f"{st['p95']:.3f}ms | {st['min']:.3f}ms | {st['max']:.3f}ms |"
            )
        lines.append(f"\n**Accuracy:** {reward_stats.get('accuracy', 0):.2%} "
                     f"({reward_stats.get('num_evaluations', 0)} evaluations)")

    # Phase 1 comparison note
    lines.append("\n## Phase 1 Correlation\n")
    lines.append("Compare rollout timing with Phase 1's concurrency sweep to validate "
                 "that the RL loop's rollout stage exhibits the same CPU scheduling bottleneck.\n")
    lines.append("| Metric | Phase 1 (conc=128) | Phase 2 (rollout) |")
    lines.append("| --- | --- | --- |")
    lines.append("| Rollout E2E | See PHASE_1_SUMMARY.md | See rollout_s above |")
    lines.append("| CPU % | 26.7% (conc=128) | See cpu_percent above |")

    md_content = "\n".join(lines) + "\n"
    out_md.write_text(md_content, encoding="utf-8")
    print(f"[07_compiler] Markdown saved to {out_md}")

    # -------------------------------------------------------------------
    # Console summary
    # -------------------------------------------------------------------
    print(f"\n{'='*80}")
    print(f"PHASE 2 — GRPO RL Loop Timing Summary")
    print(f"{'='*80}")
    print(f"  Steps:    {len(step_timings)}")
    print(f"  Total:    {total_elapsed:.1f}s")
    d = derived_stats
    if d.get("step_total_s", {}).get("mean"):
        print(f"  Avg step: {d['step_total_s']['mean']:.2f}s")
        print(f"  CPU%:     {d.get('cpu_percent', {}).get('mean', 0):.1f}%")
        print(f"  GPU%:     {d.get('gpu_percent', {}).get('mean', 0):.1f}%")
        print(f"  Rollout%: {d.get('rollout_percent', {}).get('mean', 0):.1f}%")
        print(f"  Sync%:    {d.get('sync_percent', {}).get('mean', 0):.1f}%")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
