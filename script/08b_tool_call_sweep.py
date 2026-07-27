#!/usr/bin/env python3
"""
Phase 2e Sweep: Multi-Turn Tool-Calling Concurrency Sweep
==========================================================
Runs 08_tool_call_driver.py across a matrix of (concurrency, num_turns)
configurations to measure how multi-turn tool-calling workloads stress
prefix caching, KV cache capacity, and scheduler prefill/decode interleaving.

Usage:
    python script/08b_tool_call_sweep.py --url http://localhost:8000
    python script/08b_tool_call_sweep.py --concurrencies 4 16 64 128 256
    python script/08b_tool_call_sweep.py --turns 2 4 8

Output:
    result/08_sweep/C{conc}_T{turns}/08_tool_call_driver.json  (per-config)
    result/08_sweep/08_sweep_summary.json                       (aggregated)
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser(
    description="Phase 2e: multi-turn tool-calling concurrency sweep"
)
parser.add_argument("--url", default="http://localhost:8000",
                    help="vLLM server URL")
parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B",
                    help="Model name")
parser.add_argument("--concurrencies", nargs="+", type=int,
                    default=[4, 16, 64, 128, 256],
                    help="Concurrency levels to test")
parser.add_argument("--turns", nargs="+", type=int,
                    default=[2, 4, 8],
                    help="Number of tool-calling turns per rollout")
parser.add_argument("--input-tokens", type=int, default=2000,
                    help="Approximate initial prompt size in tokens")
parser.add_argument("--max-output-tokens", type=int, default=512,
                    help="Max tokens per turn response (used when --output-length-dist is not set)")
parser.add_argument("--output-length-dist", default=None,
                    help=("Variable-length output distribution passed to driver. "
                          "Format: 'name1:tokens1:weight1,name2:tokens2:weight2,...'"))
parser.add_argument("--tool-latency-ms", type=int, default=100,
                    help="Simulated tool execution latency (ms)")
parser.add_argument("--tool-latency-jitter-ms", type=int, default=0,
                    help="Random jitter added to tool latency (uniform ±jitter)")
parser.add_argument("--tool-response-tokens", type=int, default=500,
                    help="Simulated tool result size in tokens")
parser.add_argument("--num-batches", type=int, default=3,
                    help="Measurement batches per configuration")
parser.add_argument("--warmup-batches", type=int, default=1,
                    help="Warmup batches")
parser.add_argument("--temperature", type=float, default=0.7,
                    help="Sampling temperature")
parser.add_argument("--request-timeout", type=int, default=600,
                    help="Per-request timeout in seconds")
parser.add_argument("--outdir", default=None,
                    help="Output directory (default: result/08_sweep)")
args = parser.parse_args()

project_dir = Path(__file__).resolve().parents[1]
outdir = Path(args.outdir) if args.outdir else project_dir / "result" / "08_sweep"
outdir.mkdir(parents=True, exist_ok=True)

# Build sweep matrix
configs = []
for conc in args.concurrencies:
    for turns in args.turns:
        configs.append((conc, turns))

print("=" * 60)
print(" Phase 2e Sweep: Multi-Turn Tool-Calling")
print("=" * 60)
print(f"  URL:              {args.url}")
print(f"  Model:            {args.model}")
print(f"  Concurrency:      {args.concurrencies}")
print(f"  Turns:            {args.turns}")
print(f"  Configs:          {len(configs)}")
print(f"  Batches/config:   {args.num_batches}")
print(f"  Tool latency:     {args.tool_latency_ms}ms")
print(f"  Tool resp tokens: {args.tool_response_tokens}")
print("=" * 60)

sweep_results = []
start_time = time.time()

for idx, (conc, turns) in enumerate(configs):
    config_tag = f"C{conc}_T{turns}"
    config_dir = outdir / config_tag
    config_dir.mkdir(parents=True, exist_ok=True)
    output_json = config_dir / "08_tool_call_driver.json"

    print(f"\n{'='*60}")
    print(f" [{idx+1}/{len(configs)}] {config_tag}: concurrency={conc}, turns={turns}")
    print(f"{'='*60}")

    cmd = [
        sys.executable, str(project_dir / "script" / "08_tool_call_driver.py"),
        "--url", args.url,
        "--model", args.model,
        "--concurrency", str(conc),
        "--num-turns", str(turns),
        "--input-tokens", str(args.input_tokens),
        "--max-output-tokens", str(args.max_output_tokens),
        "--tool-latency-ms", str(args.tool_latency_ms),
        "--tool-latency-jitter-ms", str(args.tool_latency_jitter_ms),
        "--tool-response-tokens", str(args.tool_response_tokens),
        "--num-batches", str(args.num_batches),
        "--warmup-batches", str(args.warmup_batches),
        "--temperature", str(args.temperature),
        "--request-timeout", str(args.request_timeout),
        "--output", str(output_json),
    ]
    if args.output_length_dist:
        cmd.extend(["--output-length-dist", args.output_length_dist])

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=args.request_timeout * (args.warmup_batches + args.num_batches) * conc + 600,
        )
        elapsed = time.time() - t0

        if result.returncode != 0:
            print(f"  [FAIL] exit code {result.returncode}")
            if result.stderr:
                print(f"  stderr: {result.stderr[-500:]}")
            sweep_results.append({
                "config_tag": config_tag,
                "concurrency": conc,
                "num_turns": turns,
                "status": "error",
                "error": result.stderr[-500:] if result.stderr else "non-zero exit",
                "elapsed_s": round(elapsed, 1),
            })
            continue

        # Load results
        if output_json.exists():
            data = json.loads(output_json.read_text())
            summary = data.get("summary", {})
            sweep_results.append({
                "config_tag": config_tag,
                "concurrency": conc,
                "num_turns": turns,
                "status": "ok",
                "elapsed_s": round(elapsed, 1),
                "rollout_e2e_p50": summary.get("rollout_e2e_mean_across_batches", {}).get("p50", 0),
                "rollout_e2e_p95": summary.get("rollout_e2e_mean_across_batches", {}).get("p95", 0),
                "throughput_p50": summary.get("output_token_throughput", {}).get("p50", 0),
                "group_completion_p95": summary.get("group_completion_p95", {}).get("p50", 0),
                "scheduled_to_first_output_p50": summary.get("scheduled_to_first_output_mean", {}).get("p50", 0),
                "first_output_to_completion_p50": summary.get("first_output_to_completion_mean", {}).get("p50", 0),
                "per_turn_ttft": summary.get("per_turn_ttft_mean_across_batches", {}),
                "per_turn_e2e": summary.get("per_turn_e2e_mean_across_batches", {}),
            })
            print(f"  [OK] {elapsed:.1f}s | "
                  f"rollout E2E P50={sweep_results[-1]['rollout_e2e_p50']:.0f}ms")
        else:
            sweep_results.append({
                "config_tag": config_tag,
                "concurrency": conc,
                "num_turns": turns,
                "status": "missing_output",
                "elapsed_s": round(elapsed, 1),
            })

    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        print(f"  [TIMEOUT] after {elapsed:.0f}s")
        sweep_results.append({
            "config_tag": config_tag,
            "concurrency": conc,
            "num_turns": turns,
            "status": "timeout",
            "elapsed_s": round(elapsed, 1),
        })
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  [ERROR] {e}")
        sweep_results.append({
            "config_tag": config_tag,
            "concurrency": conc,
            "num_turns": turns,
            "status": "error",
            "error": str(e)[:300],
            "elapsed_s": round(elapsed, 1),
        })

# Write summary
total_elapsed = time.time() - start_time
summary = {
    "config": {
        "url": args.url,
        "model": args.model,
        "concurrencies": args.concurrencies,
        "turns": args.turns,
        "input_tokens": args.input_tokens,
        "max_output_tokens": args.max_output_tokens,
        "tool_latency_ms": args.tool_latency_ms,
        "tool_response_tokens": args.tool_response_tokens,
        "num_batches": args.num_batches,
        "warmup_batches": args.warmup_batches,
    },
    "total_configs": len(configs),
    "total_elapsed_s": round(total_elapsed, 1),
    "results": sweep_results,
}

summary_path = outdir / "08_sweep_summary.json"
summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

print(f"\n{'='*60}")
print(f" Sweep complete! {total_elapsed:.0f}s total")
print(f" Results: {outdir}")
print(f" Summary: {summary_path}")
ok_count = sum(1 for r in sweep_results if r["status"] == "ok")
print(f" {ok_count}/{len(configs)} configs succeeded")
print(f"{'='*60}")
