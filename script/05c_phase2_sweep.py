#!/usr/bin/env python3
"""
Phase 2 Concurrency Sweep — Instrumented
==========================================
Runs 05_grpo_instrumented.py at multiple (rollout_n, train_batch_size) configs
to measure how rollout latency scales with B×G concurrency.

Collects per-request timing, vLLM /metrics, and optionally OTel traces
for each configuration.

Usage:
    python script/05c_phase2_sweep.py --model Qwen/Qwen3-30B-A3B
    python script/05c_phase2_sweep.py --data-path data/dapo-math-17k.parquet
    python script/05c_phase2_sweep.py --enable-otel --otlp-endpoint http://localhost:4317

Output:
    result/phase2_sweep/sweep_summary.json
    result/phase2_sweep/rollout_n={N}_batch={B}/05_grpo_instrumented.json
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

# Default sweep configs: (rollout_n, train_batch_size)
DEFAULT_CONFIGS = [
    (2, 4),    # B*G = 8
    (4, 4),    # B*G = 16
    (4, 8),    # B*G = 32
    (4, 16),   # B*G = 64
    (4, 32),   # B*G = 128
]

# ---------------------------------------------------------------------------
# Parse CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Phase 2 concurrency sweep — instrumented GRPO training"
)
parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B",
                    help="Model name or path")
parser.add_argument("--data-path", default=None,
                    help="Path to training data parquet")
parser.add_argument("--val-path", default=None,
                    help="Path to validation parquet")
parser.add_argument("--configs", default=None,
                    help="JSON list of [rollout_n, train_batch_size] pairs")
parser.add_argument("--jaeger-url", default="http://localhost:16686",
                    help="Jaeger query API URL")
parser.add_argument("--enable-otel", action="store_true", default=False,
                    help="Enable OTel tracing for vLLM")
parser.add_argument("--otlp-endpoint", default="http://localhost:4317",
                    help="OTLP collector endpoint")
parser.add_argument("--num-epochs", type=int, default=1,
                    help="Epochs per config (default 1 for sweep)")
parser.add_argument("--timeout", type=int, default=3600,
                    help="Timeout per config in seconds")
parser.add_argument("--max-prompt-length", type=int, default=512,
                    help="Max prompt token length")
parser.add_argument("--max-response-length", type=int, default=64,
                    help="Max response token length")
parser.add_argument("--ppo-mini-batch-size", type=int, default=1,
                    help="PPO mini-batch size")
parser.add_argument("--skip-venv", action="store_true", default=False,
                    help="Skip venv activation")
parser.add_argument("--rollout-only", action="store_true", default=False,
                    help="Run grouped veRL rollout only; skip all GRPO update stages")
parser.add_argument("--warmup-steps", type=int, default=1,
                    help="Warmup rollout steps excluded from metrics")
parser.add_argument("--measured-steps", type=int, default=2,
                    help="Measured rollout steps per configuration")
parser.add_argument("--disable-rollout-trace", action="store_true", default=False,
                    help="Disable request-correlated Python instrumentation")
args = parser.parse_args()

if args.data_path is None:
    args.data_path = str(PROJECT_DIR / "data" / "dapo-math-17k.parquet")
if args.val_path is None:
    args.val_path = str(PROJECT_DIR / "data" / "aime-2024.parquet")

if args.configs:
    configs = json.loads(args.configs)
else:
    configs = DEFAULT_CONFIGS

OUTDIR = PROJECT_DIR / "result" / "phase2_sweep"
OUTDIR.mkdir(parents=True, exist_ok=True)

NO_PROXY_VALUE = "localhost,127.0.0.1,::1,0.0.0.0"


def run_no_proxy(*cmd):
    """Run command with proxy disabled."""
    env = os.environ.copy()
    for var in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
                "all_proxy", "ALL_PROXY"]:
        env.pop(var, None)
    env["NO_PROXY"] = NO_PROXY_VALUE
    env["no_proxy"] = NO_PROXY_VALUE
    return env


def run_config(rollout_n: int, train_bs: int) -> dict:
    """Run one sweep configuration."""
    concurrency = rollout_n * train_bs
    config_dir = OUTDIR / f"rollout_n={rollout_n}_batch={train_bs}"
    config_dir.mkdir(parents=True, exist_ok=True)
    log_file = config_dir / "training.log"

    cmd = [
        PYTHON, str(PROJECT_DIR / "script" / "05_grpo_instrumented.py"),
        "--model", args.model,
        "--data-path", args.data_path,
        "--val-path", args.val_path,
        "--rollout-n", str(rollout_n),
        "--train-batch-size", str(train_bs),
        "--num-epochs", str(args.num_epochs),
        "--max-prompt-length", str(args.max_prompt_length),
        "--max-response-length", str(args.max_response_length),
        "--ppo-mini-batch-size", str(args.ppo_mini_batch_size),
        "--log-file", str(log_file),
        "--output", str(config_dir / "05_grpo_instrumented.json"),
    ]

    if args.enable_otel:
        cmd.extend(["--enable-otel", "--otlp-endpoint", args.otlp_endpoint])

    cmd.extend(["--jaeger-url", args.jaeger_url])
    if args.rollout_only:
        cmd.extend([
            "--rollout-only",
            "--warmup-steps", str(args.warmup_steps),
            "--measured-steps", str(args.measured_steps),
        ])
    if args.disable_rollout_trace:
        cmd.append("--disable-rollout-trace")

    print(f"\n{'='*70}")
    print(f"  Config: concurrency={concurrency}  rollout_n={rollout_n}  batch={train_bs}")
    print(f"{'='*70}")

    env = run_no_proxy()

    t_start = time.perf_counter()
    try:
        result = subprocess.run(
            cmd,
            env=env,
            cwd=str(PROJECT_DIR),
            timeout=args.timeout,
        )
        elapsed = time.perf_counter() - t_start

        if result.returncode != 0:
            print(f"  FAILED: exit code {result.returncode} after {elapsed:.0f}s")
            return {
                "status": "error",
                "exit_code": result.returncode,
                "elapsed_s": round(elapsed, 1),
            }

        # Read the output JSON for summary
        output_file = config_dir / "05_grpo_instrumented.json"
        summary = {}
        if output_file.exists():
            data = json.loads(output_file.read_text())
            rollout_only = data.get("rollout_only_metrics")
            if rollout_only:
                stages = rollout_only.get("request_stages", {})
                summary = {
                    "num_steps": args.measured_steps,
                    "group_ready_p95_ms": rollout_only.get(
                        "group_ready_latency", {}
                    ).get("p95_ms"),
                    "request_first_output_p95_ms": rollout_only.get(
                        "request_tail_latency", {}
                    ).get("p95_ms"),
                    "prompt_preparation_p95_ms": stages.get(
                        "prompt_preparation", {}
                    ).get("p95_ms"),
                    "request_admission_p95_ms": stages.get(
                        "request_admission", {}
                    ).get("p95_ms"),
                    "scheduler_wait_p95_ms": stages.get(
                        "scheduler_wait", {}
                    ).get("p95_ms"),
                    "first_scheduled_to_output_p95_ms": stages.get(
                        "first_scheduled_to_output", {}
                    ).get("p95_ms"),
                    "output_token_throughput": rollout_only.get(
                        "throughput", {}
                    ).get("output_tokens_per_s"),
                    "completed_group_throughput": rollout_only.get(
                        "throughput", {}
                    ).get("completed_groups_per_s"),
                    "kv_allocation_failures": rollout_only.get(
                        "kv_allocation", {}
                    ).get("failure_attempts", 0),
                    "preemption_events": rollout_only.get(
                        "preemption", {}
                    ).get("events", 0),
                    "terminal_request_failures": rollout_only.get(
                        "terminal_request_failures", 0
                    ),
                }
            rollout_steps = data.get("rollout_steps", [])
            if rollout_steps and not rollout_only:
                # Aggregate across steps
                all_p95_group = [s["group_completion_p95_ms"] for s in rollout_steps]
                all_throughput = [s["output_token_throughput"] for s in rollout_steps]
                all_p95_latency = [s["request_latency_p95_ms"] for s in rollout_steps]
                summary = {
                    "num_steps": len(rollout_steps),
                    "mean_rollout_ms": round(
                        sum(s["t_rollout_ms"] for s in rollout_steps) / len(rollout_steps), 1
                    ),
                    "mean_group_completion_p95_ms": round(sum(all_p95_group) / len(all_p95_group), 1),
                    "mean_output_token_throughput": round(sum(all_throughput) / len(all_throughput), 1),
                    "mean_request_latency_p95_ms": round(sum(all_p95_latency) / len(all_p95_latency), 1),
                    "mean_num_requests": round(
                        sum(s["num_requests"] for s in rollout_steps) / len(rollout_steps), 1
                    ),
                }

                # vLLM metrics from first step (most representative after warmup)
                first_step = rollout_steps[0]
                vllm_delta = first_step.get("vllm_metrics_delta", {})
                if vllm_delta:
                    summary["vllm_prefix_cache_hit_rate"] = vllm_delta.get(
                        "vllm_prefix_cache_hit_rate", None
                    )
                    for key in ["vllm_queue_time_p95", "vllm_prefill_time_p95",
                                "vllm_decode_time_p95", "vllm_ttft_p95", "vllm_itl_p95"]:
                        if key in vllm_delta:
                            summary[key] = vllm_delta[key]

                # Jaeger data from first step
                jaeger = first_step.get("jaeger", {})
                if jaeger:
                    for key, val in jaeger.items():
                        if key.startswith("otel_"):
                            summary[key] = val

        print(f"  SUCCESS: {elapsed:.0f}s")
        if summary:
            if args.rollout_only:
                print(f"    Steps: {summary.get('num_steps', '?')}, "
                      f"Group ready P95: {summary.get('group_ready_p95_ms')}ms, "
                      f"Tok/s: {summary.get('output_token_throughput')}")
            else:
                print(f"    Steps: {summary.get('num_steps', '?')}, "
                      f"Rollout: {summary.get('mean_rollout_ms', '?'):.0f}ms, "
                      f"P95 group: {summary.get('mean_group_completion_p95_ms', '?'):.0f}ms, "
                      f"Tok/s: {summary.get('mean_output_token_throughput', '?'):.0f}")

        return {
            "status": "success",
            "elapsed_s": round(elapsed, 1),
            **summary,
        }

    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - t_start
        print(f"  TIMEOUT after {args.timeout}s")
        return {"status": "timeout", "elapsed_s": round(elapsed, 1)}


def main():
    print(f"[05c_sweep] Phase 2 Concurrency Sweep (Instrumented)")
    print(f"[05c_sweep] Model:     {args.model}")
    print(f"[05c_sweep] Data:      {args.data_path}")
    print(f"[05c_sweep] Configs:   {[(r, b) for r, b in configs]}")
    print(f"[05c_sweep] Epochs:    {args.num_epochs}")
    print(f"[05c_sweep] OTel:      {'enabled' if args.enable_otel else 'disabled'}")
    print(f"[05c_sweep] Mode:      "
          f"{'verl_group_rollout_only' if args.rollout_only else 'grpo_training'}")
    print(f"[05c_sweep] Output:    {OUTDIR}")

    results = []
    for rollout_n, train_bs in configs:
        concurrency = rollout_n * train_bs
        result = run_config(rollout_n, train_bs)
        result["rollout_n"] = rollout_n
        result["train_batch_size"] = train_bs
        result["concurrency"] = concurrency
        results.append(result)

        # Save intermediate results
        (OUTDIR / "sweep_summary.json").write_text(
            json.dumps(results, indent=2, ensure_ascii=False)
        )

    # Print summary table
    print(f"\n{'='*100}")
    print(f"  SWEEP SUMMARY")
    print(f"{'='*100}")
    print(f"  {'Conc':>6s} {'N':>4s} {'B':>4s} {'Status':>8s} "
          f"{'Steps':>6s} {'P95_group':>12s} {'P95_req':>10s} {'Tok/s':>10s} "
          f"{'KVFail':>8s} {'Preempt':>8s}")
    print(f"  {'─'*6} {'─'*4} {'─'*4} {'─'*8} "
          f"{'─'*6} {'─'*10} {'─'*12} {'─'*10} "
          f"{'─'*10} {'─'*8}")
    for r in results:
        c = r["concurrency"]
        status = r["status"]
        if status == "success":
            print(f"  {c:6d} {r['rollout_n']:4d} {r['train_batch_size']:4d} {status:>8s} "
                  f"{r.get('num_steps', 0):6d} "
                  f"{(r.get('group_ready_p95_ms') or r.get('mean_group_completion_p95_ms', 0)):12.0f} "
                  f"{(r.get('request_first_output_p95_ms') or r.get('mean_request_latency_p95_ms', 0)):10.0f} "
                  f"{(r.get('output_token_throughput') or r.get('mean_output_token_throughput', 0)):10.0f} "
                  f"{r.get('kv_allocation_failures', 0):8d} "
                  f"{r.get('preemption_events', 0):8d}")
        else:
            print(f"  {c:6d} {r['rollout_n']:4d} {r['train_batch_size']:4d} {status:>8s}")

    print(f"{'='*100}")
    print(f"Results saved to: {OUTDIR / 'sweep_summary.json'}")


if __name__ == "__main__":
    main()
