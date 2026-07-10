#!/usr/bin/env python3
"""
Phase 2d Sweep: GRPO HTTP Driver Concurrency Sweep
=====================================================
Runs 05d_grpo_http_driver.py at multiple (B, G) configurations to cover
concurrency levels from 4 to 1024. Each concurrency level has at least
two B×G combinations for cross-validation.

Usage:
    python script/05d_sweep.py --url http://localhost:8000
    python script/05d_sweep.py --skip-low  # Skip concurrency 4/8
    python script/05d_sweep.py --num-steps 5

Output:
    result/05d_sweep/05d_sweep_B{B}_G{G}.json  (per-config)
    result/05d_sweep/05d_sweep_summary.json     (aggregated)
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser(
    description="GRPO HTTP driver concurrency sweep"
)
parser.add_argument("--url", default="http://localhost:8000",
                    help="vLLM server URL")
parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B",
                    help="Model name")
parser.add_argument("--num-steps", type=int, default=3,
                    help="GRPO steps per configuration")
parser.add_argument("--max-samples", type=int, default=200,
                    help="Max dataset samples")
parser.add_argument("--max-prompt-length", type=int, default=8192,
                    help="Max prompt token length")
parser.add_argument("--max-response-length", type=int, default=64,
                    help="Max response token length")
parser.add_argument("--temperature", type=float, default=0.7,
                    help="Sampling temperature")
parser.add_argument("--poll-interval", type=int, default=20,
                    help="Gauge poll interval in ms")
parser.add_argument("--scheduler-trace-path", default=None,
                    help="vLLM scheduler JSONL trace path to pass through to 05d")
parser.add_argument("--scheduler-trace-window-padding-ms", type=int, default=200,
                    help="Scheduler trace time-window padding passed through to 05d")
parser.add_argument("--skip-low", action="store_true", default=False,
                    help="Skip concurrency 4 and 8")
parser.add_argument("--skip-high", action="store_true", default=False,
                    help="Skip concurrency 512 and 1024")
parser.add_argument("--configs", nargs="*", default=None,
                    help="Override: specific BxG configs, e.g. '8x8' '16x4'")
parser.add_argument("--no-plots", action="store_true", default=False,
                    help="Skip plot generation")
parser.add_argument("--outdir", default=None,
                    help="Output directory (default: result/05d_sweep)")
args = parser.parse_args()

project_dir = Path(__file__).resolve().parents[1]
outdir = Path(args.outdir) if args.outdir else project_dir / "result" / "05d_sweep"
outdir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Sweep matrix: concurrency -> [(B, G), ...]
# ---------------------------------------------------------------------------
SWEEP_MATRIX = {
    4:   [(2, 2), (1, 4)],
    8:   [(4, 2), (2, 4)],
    16:  [(4, 4), (8, 2)],
    32:  [(8, 4), (16, 2)],
    64:  [(8, 8), (16, 4)],
    128: [(16, 8), (32, 4)],
    256: [(32, 8), (64, 4)],
    512: [(64, 8), (128, 4)],
    1024: [(128, 8), (256, 4)],
}

# Build list of configs to run
configs = []

if args.configs:
    # User-specified configs
    for spec in args.configs:
        parts = spec.lower().split("x")
        if len(parts) == 2:
            b, g = int(parts[0]), int(parts[1])
            configs.append((b, g))
        else:
            print(f"[sweep] WARNING: invalid config '{spec}', expected BxG format")
else:
    for conc in sorted(SWEEP_MATRIX.keys()):
        if args.skip_low and conc in (4, 8):
            continue
        if args.skip_high and conc in (512, 1024):
            continue
        for b, g in SWEEP_MATRIX[conc]:
            configs.append((b, g))

print(f"[sweep] GRPO HTTP Driver Concurrency Sweep")
print(f"[sweep] Server:          {args.url}")
print(f"[sweep] Model:           {args.model}")
print(f"[sweep] Steps/config:    {args.num_steps}")
print(f"[sweep] Configs:         {len(configs)}")
print(f"[sweep] Input tokens:    {args.max_prompt_length}")
print(f"[sweep] Output tokens:   {args.max_response_length}")
print(f"[sweep] Scheduler trace: {args.scheduler_trace_path or '<disabled>'}")
print(f"[sweep] Output dir:      {outdir}")
print()

# ---------------------------------------------------------------------------
# Run sweep
# ---------------------------------------------------------------------------
results = []
t_total_start = time.perf_counter()

for idx, (b, g) in enumerate(configs):
    conc = b * g
    tag = f"B{b}_G{g}_C{conc}"
    out_file = outdir / f"05d_sweep_{tag}.json"

    print(f"{'='*70}")
    print(f"  [{idx+1}/{len(configs)}] {tag}  (concurrency={conc})")
    print(f"{'='*70}")

    cmd = [
        sys.executable, str(project_dir / "script" / "05d_grpo_http_driver.py"),
        "--url", args.url,
        "--model", args.model,
        "--num-steps", str(args.num_steps),
        "--train-batch-size", str(b),
        "--rollout-n", str(g),
        "--max-samples", str(args.max_samples),
        "--max-prompt-length", str(args.max_prompt_length),
        "--max-response-length", str(args.max_response_length),
        "--temperature", str(args.temperature),
        "--poll-interval", str(args.poll_interval),
        "--output", str(out_file),
    ]
    if args.scheduler_trace_path:
        cmd.extend([
            "--scheduler-trace-path", args.scheduler_trace_path,
            "--scheduler-trace-window-padding-ms",
            str(args.scheduler_trace_window_padding_ms),
        ])
    if args.no_plots:
        cmd.append("--no-plots")

    t_start = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    t_elapsed = time.perf_counter() - t_start

    if proc.returncode != 0:
        print(f"  FAILED ({t_elapsed:.1f}s)")
        print(f"  stderr: {proc.stderr[-500:]}")
        results.append({
            "tag": tag, "B": b, "G": g, "concurrency": conc,
            "status": "failed", "elapsed_s": round(t_elapsed, 1),
            "error": proc.stderr[-500:],
        })
        continue

    # Load result to extract key metrics
    try:
        data = json.loads(out_file.read_text())
        agg = data.get("aggregate_request_stats", {})
        e2e = agg.get("t_e2e_ms", {})
        prefill = agg.get("t_server_prefill_ms", {})
        decode = agg.get("t_decode_ms", {})
        sem = agg.get("t_sem_wait_ms", {})
        http = agg.get("t_http_connect_ms", {})
        queue = agg.get("t_http_response_headers_wait_ms", {})

        step_summaries = data.get("step_summaries", [])
        step_rollout_times = [s.get("t_rollout_ms", 0) for s in step_summaries]
        step_throughputs = [s.get("output_token_throughput", 0) for s in step_summaries]
        rewards = [s.get("reward_mean", 0) for s in step_summaries]

        # vLLM server metrics from first step
        vllm = data.get("vllm_metrics", {})
        first_vllm = vllm.get("0", {})
        scheduler_summaries = [
            s.get("scheduler_trace_summary", {}) for s in step_summaries
            if s.get("scheduler_trace_available")
        ]
        scheduler_ticks = sum(s.get("num_schedule_ticks", 0)
                              for s in scheduler_summaries)
        scheduler_tokens = sum(s.get("scheduled_tokens_total", 0)
                               for s in scheduler_summaries)
        scheduler_alloc_failures = sum(s.get("kv_allocate_failure_count", 0)
                                      for s in scheduler_summaries)
        scheduler_preemptions = sum(s.get("preempted_req_count", 0)
                                    for s in scheduler_summaries)
        first_token_stage_summaries = [
            s.get("server_first_token_stage_summary", {})
            for s in step_summaries
            if s.get("server_first_token_stage_summary", {}).get(
                "matched_request_count", 0
            )
        ]
        first_stage = (
            first_token_stage_summaries[0]
            if first_token_stage_summaries else {}
        )

        entry = {
            "tag": tag, "B": b, "G": g, "concurrency": conc,
            "status": "ok", "elapsed_s": round(t_elapsed, 1),
            "num_requests": data.get("num_requests_total", 0),
            "num_successful": data.get("num_successful_total", 0),
            # Per-request E2E
            "e2e_p50_ms": e2e.get("p50", 0),
            "e2e_p95_ms": e2e.get("p95", 0),
            "e2e_p99_ms": e2e.get("p99", 0),
            "e2e_mean_ms": e2e.get("mean", 0),
            # Server prefill
            "prefill_p50_ms": prefill.get("p50", 0),
            "prefill_p95_ms": prefill.get("p95", 0),
            "prefill_mean_ms": prefill.get("mean", 0),
            # Decode
            "decode_p50_ms": decode.get("p50", 0),
            "decode_p95_ms": decode.get("p95", 0),
            "decode_mean_ms": decode.get("mean", 0),
            # Client-side breakdown
            "sem_wait_p50_ms": sem.get("p50", 0),
            "sem_wait_p95_ms": sem.get("p95", 0),
            "http_connect_p50_ms": http.get("p50", 0),
            "http_connect_p95_ms": http.get("p95", 0),
            "http_resp_headers_wait_p50_ms": queue.get("p50", 0),
            "http_resp_headers_wait_p95_ms": queue.get("p95", 0),
            # Rollout step
            "rollout_ms_mean": round(sum(step_rollout_times) / len(step_rollout_times), 1) if step_rollout_times else 0,
            "rollout_ms_p95": round(sorted(step_rollout_times)[int(len(step_rollout_times) * 0.95)], 1) if step_rollout_times else 0,
            "throughput_mean": round(sum(step_throughputs) / len(step_throughputs), 1) if step_throughputs else 0,
            "reward_mean": round(sum(rewards) / len(rewards), 4) if rewards else 0,
            # vLLM server-side (first step delta)
            "vllm_queue_time_avg_ms": first_vllm.get("vllm_queue_time_avg", 0) * 1000,
            "vllm_prefill_time_avg_ms": first_vllm.get("vllm_prefill_time_avg", 0) * 1000,
            "vllm_decode_time_avg_ms": first_vllm.get("vllm_decode_time_avg", 0) * 1000,
            "vllm_ttft_avg_ms": first_vllm.get("vllm_ttft_avg", 0) * 1000,
            "vllm_e2e_avg_ms": first_vllm.get("vllm_e2e_avg", 0) * 1000,
            "vllm_itl_avg_ms": first_vllm.get("vllm_itl_avg", 0) * 1000,
            "vllm_num_preemptions": first_vllm.get("vllm_num_preemptions", 0),
            "vllm_requests_running": first_vllm.get("vllm_requests_running", 0),
            "vllm_requests_waiting": first_vllm.get("vllm_requests_waiting", 0),
            "vllm_kv_cache_usage_pct": first_vllm.get("vllm_kv_cache_usage", 0) * 100,
            # vLLM scheduler deep trace, if enabled.
            "scheduler_trace_available": bool(scheduler_summaries),
            "scheduler_num_schedule_ticks": scheduler_ticks,
            "scheduler_scheduled_tokens_total": scheduler_tokens,
            "scheduler_kv_allocate_failures": scheduler_alloc_failures,
            "scheduler_preempted_req_count": scheduler_preemptions,
            "server_stage_matched_requests": first_stage.get(
                "matched_request_count", 0
            ),
            "server_scheduler_wait_p95_ms": first_stage.get(
                "server_scheduler_wait_ms", {}
            ).get("p95", 0),
            "server_first_schedule_to_output_p95_ms": first_stage.get(
                "server_first_schedule_to_output_ms", {}
            ).get("p95", 0),
            "server_stream_result_to_first_chunk_p95_ms": first_stage.get(
                "server_stream_result_to_first_chunk_ms", {}
            ).get("p95", 0),
        }
        results.append(entry)

        print(f"  OK ({t_elapsed:.1f}s) | E2E P95={e2e.get('p95',0):.0f}ms | "
              f"Prefill P95={prefill.get('p95',0):.0f}ms | "
              f"Decode P95={decode.get('p95',0):.0f}ms | "
              f"Tok/s={entry['throughput_mean']:.0f} | "
              f"SchedTicks={scheduler_ticks} AllocFail={scheduler_alloc_failures}")

    except Exception as e:
        print(f"  Parse error: {e}")
        results.append({
            "tag": tag, "B": b, "G": g, "concurrency": conc,
            "status": "parse_error", "elapsed_s": round(t_elapsed, 1),
            "error": str(e),
        })

t_total = time.perf_counter() - t_total_start

# ---------------------------------------------------------------------------
# Save summary
# ---------------------------------------------------------------------------
summary = {
    "benchmark": "05d_sweep",
    "server_url": args.url,
    "model": args.model,
    "num_steps_per_config": args.num_steps,
    "max_prompt_length": args.max_prompt_length,
    "max_response_length": args.max_response_length,
    "temperature": args.temperature,
    "scheduler_trace_path": args.scheduler_trace_path,
    "scheduler_trace_window_padding_ms": args.scheduler_trace_window_padding_ms,
    "total_elapsed_s": round(t_total, 1),
    "num_configs": len(configs),
    "num_ok": sum(1 for r in results if r.get("status") == "ok"),
    "results": results,
}

summary_path = outdir / "05d_sweep_summary.json"
summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
print(f"\n[sweep] Summary saved to {summary_path}")

# ---------------------------------------------------------------------------
# Console summary table
# ---------------------------------------------------------------------------
print(f"\n{'='*140}")
print(f"GRPO HTTP DRIVER SWEEP — Summary ({len(results)} configs, {t_total:.0f}s total)")
print(f"{'='*140}")
header = (f"{'Tag':>18s} {'Conc':>5s} {'B':>4s} {'G':>4s} "
          f"{'E2E_P50':>9s} {'E2E_P95':>9s} {'E2E_P99':>9s} "
          f"{'Prefill':>9s} {'Decode':>9s} {'SemWait':>9s} "
          f"{'HTTP':>9s} {'Tok/s':>8s} {'Reward':>8s} {'Status':>8s}")
print(header)
print("-" * 140)

for r in results:
    if r.get("status") != "ok":
        print(f"{r['tag']:>18s} {r['concurrency']:>5d} {r['B']:>4d} {r['G']:>4d} "
              f"{'FAIL':>9s} — {r.get('error', '')[:50]}")
        continue
    print(f"{r['tag']:>18s} {r['concurrency']:>5d} {r['B']:>4d} {r['G']:>4d} "
          f"{r['e2e_p50_ms']:7.0f}ms {r['e2e_p95_ms']:7.0f}ms {r['e2e_p99_ms']:7.0f}ms "
          f"{r['prefill_p95_ms']:7.0f}ms {r['decode_p95_ms']:7.0f}ms "
          f"{r['sem_wait_p95_ms']:7.0f}ms "
          f"{r['http_connect_p95_ms']:7.0f}ms "
          f"{r['throughput_mean']:6.0f}t/s "
          f"{r['reward_mean']:8.4f} "
          f"{'OK':>8s}")

# vLLM server-side table
print(f"\n{'='*120}")
print(f"vLLM SERVER-SIDE METRICS (per-config)")
print(f"{'='*120}")
header2 = (f"{'Tag':>18s} {'Conc':>5s} {'Queue':>9s} {'Prefill':>9s} "
           f"{'Decode':>9s} {'TTFT':>9s} {'E2E':>9s} {'ITL':>9s} "
           f"{'Preempt':>8s} {'Run':>5s} {'Wait':>5s} {'KV%':>5s}")
print(header2)
print("-" * 120)
for r in results:
    if r.get("status") != "ok":
        continue
    print(f"{r['tag']:>18s} {r['concurrency']:>5d} "
          f"{r['vllm_queue_time_avg_ms']:7.2f}ms "
          f"{r['vllm_prefill_time_avg_ms']:7.2f}ms "
          f"{r['vllm_decode_time_avg_ms']:7.2f}ms "
          f"{r['vllm_ttft_avg_ms']:7.2f}ms "
          f"{r['vllm_e2e_avg_ms']:7.2f}ms "
          f"{r['vllm_itl_avg_ms']:7.2f}ms "
          f"{r['vllm_num_preemptions']:8.0f} "
          f"{r['vllm_requests_running']:5.0f} "
          f"{r['vllm_requests_waiting']:5.0f} "
          f"{r['vllm_kv_cache_usage_pct']:5.1f}%")

print(f"{'='*120}")
print(f"\n[sweep] Done. {len(list(outdir.glob('05d_sweep_B*.json')))} result files in {outdir}/")
