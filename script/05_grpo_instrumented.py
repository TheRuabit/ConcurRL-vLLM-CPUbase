#!/usr/bin/env python3
"""
Phase 2: Instrumented GRPO Training
====================================
Runs veRL GRPO training with deep instrumentation:
  - Per-request latency breakdown (TTFT, decode, group completion)
  - vLLM /metrics Prometheus scraping (histograms + gauges)
  - Optional Jaeger OTel trace fetching

Replaces 05_grpo_train.py with full observability pipeline.

Usage:
    python script/05_grpo_instrumented.py --model Qwen/Qwen3-30B-A3B
    python script/05_grpo_instrumented.py --rollout-n 8 --train-batch-size 32
    python script/05_grpo_instrumented.py --enable-otel --otlp-endpoint http://localhost:4317

Output:
    result/05_grpo_instrumented.json
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Parse CLI BEFORE importing veRL (so --help is fast)
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Phase 2: Instrumented GRPO training via veRL"
)
parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B",
                    help="Model name or path")
parser.add_argument("--data-path", default=None,
                    help="Path to training data parquet")
parser.add_argument("--val-path", default=None,
                    help="Path to validation parquet")
parser.add_argument("--output", default=None,
                    help="Output JSON path")
parser.add_argument("--rollout-n", type=int, default=4,
                    help="Number of responses per prompt (GRPO group size)")
parser.add_argument("--train-batch-size", type=int, default=8,
                    help="Prompts per training step")
parser.add_argument("--ppo-mini-batch-size", type=int, default=2,
                    help="PPO mini-batch size")
parser.add_argument("--max-prompt-length", type=int, default=1024,
                    help="Max prompt token length")
parser.add_argument("--max-response-length", type=int, default=1024,
                    help="Max response token length")
parser.add_argument("--actor-lr", type=float, default=1e-6,
                    help="Actor learning rate")
parser.add_argument("--num-epochs", type=int, default=3,
                    help="Number of training epochs")
parser.add_argument("--rollout-tp", type=int, default=2,
                    help="Tensor parallel size for vLLM rollout")
parser.add_argument("--rollout-gpu-mem-util", type=float, default=0.50,
                    help="GPU memory utilization for vLLM rollout")
parser.add_argument("--enforce-eager", action="store_true", default=True,
                    help="Disable CUDA graphs in vLLM (default: True)")
parser.add_argument("--no-enforce-eager", action="store_false", dest="enforce_eager",
                    help="Allow CUDA graphs in vLLM")
parser.add_argument("--kl-loss-coef", type=float, default=0.001,
                    help="KL loss coefficient")
parser.add_argument("--reward-func", default=None,
                    help="Path to custom reward function")
parser.add_argument("--save-freq", type=int, default=50,
                    help="Checkpoint save frequency (steps)")
parser.add_argument("--test-freq", type=int, default=5,
                    help="Validation frequency (steps)")
parser.add_argument("--log-file", default=None,
                    help="Capture veRL stdout/stderr to this file")
parser.add_argument("--extra-overrides", nargs="*", default=[],
                    help="Additional Hydra overrides (key=value)")
# Instrumentation-specific args
parser.add_argument("--jaeger-url", default="http://localhost:16686",
                    help="Jaeger query API URL for OTel trace fetching")
parser.add_argument("--poll-interval", type=int, default=20,
                    help="Metrics poll interval in ms (0=disabled)")
parser.add_argument("--enable-otel", action="store_true", default=False,
                    help="Pass --otlp-traces-endpoint to vLLM engine")
parser.add_argument("--otlp-endpoint", default="http://localhost:4317",
                    help="OTLP collector endpoint for vLLM traces")
parser.add_argument("--dry-run", action="store_true", default=False,
                    help="Print overrides without running training")
parser.add_argument("--vllm-server-url", default=None,
                    help="External vLLM server URL (e.g. http://localhost:8000). "
                         "When set, skips launching vLLM and connects to existing server.")
parser.add_argument("--rollout-only", action="store_true", default=False,
                    help="Run veRL grouped rollout only; skip reward, advantage, and updates")
parser.add_argument("--warmup-steps", type=int, default=1,
                    help="Warmup rollout steps excluded from metrics")
parser.add_argument("--measured-steps", type=int, default=2,
                    help="Measured rollout steps")
parser.add_argument("--disable-rollout-trace", action="store_true", default=False,
                    help="Disable request-correlated Python event emission")
parser.add_argument("--trace-file", default=None,
                    help="Shared rollout event JSONL path")
args = parser.parse_args()

# Resolve paths
project_dir = Path(__file__).resolve().parents[1]

if args.output:
    out_path = Path(args.output)
else:
    out_path = project_dir / "result" / "05_grpo_instrumented.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
trace_path = (
    Path(args.trace_file)
    if args.trace_file
    else out_path.with_name(out_path.stem + "_events.jsonl")
)

if args.data_path is None:
    args.data_path = str(project_dir / "data" / "dapo-math-17k.parquet")
if args.val_path is None:
    args.val_path = str(project_dir / "data" / "aime-2024.parquet")
if args.reward_func is None:
    args.reward_func = str(project_dir / "script" / "06_math_reward.py")

# ---------------------------------------------------------------------------
# Import instrumentation (applies monkey-patches)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(project_dir))
import script.phase2_instrument as inst


# ---------------------------------------------------------------------------
# Build Hydra overrides
# ---------------------------------------------------------------------------
def build_hydra_overrides() -> list[str]:
    overrides = [
        # Data
        f"data.train_files=['{args.data_path}']",
        f"data.val_files=['{args.val_path}']",
        f"data.train_batch_size={args.train_batch_size}",
        f"data.max_prompt_length={args.max_prompt_length}",
        f"data.max_response_length={args.max_response_length}",
        "data.filter_overlong_prompts=False",
        "data.truncation=longest",

        # Model
        f"actor_rollout_ref.model.path={args.model}",
        "actor_rollout_ref.model.use_remove_padding=True",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        "+actor_rollout_ref.model.override_config.attn_implementation=sdpa",

        # Algorithm
        "algorithm.adv_estimator=grpo",
        "algorithm.use_kl_in_reward=False",

        # Actor (FSDP2 with offloading for 2-GPU setup)
        "actor_rollout_ref.actor.strategy=fsdp2",
        f"actor_rollout_ref.actor.optim.lr={args.actor_lr}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={args.ppo_mini_batch_size}",
        "actor_rollout_ref.actor.use_dynamic_bsz=True",
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096",
        "actor_rollout_ref.actor.use_kl_loss=True",
        f"actor_rollout_ref.actor.kl_loss_coef={args.kl_loss_coef}",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        "actor_rollout_ref.actor.entropy_coeff=0",
        "actor_rollout_ref.actor.fsdp_config.param_offload=True",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True",

        # Rollout (vLLM)
        "actor_rollout_ref.rollout.name=vllm",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={args.rollout_tp}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={args.rollout_gpu_mem_util}",
        f"actor_rollout_ref.rollout.enforce_eager={str(args.enforce_eager).lower()}",
        f"actor_rollout_ref.rollout.n={args.rollout_n}",
        f"actor_rollout_ref.rollout.max_model_len={args.max_prompt_length + args.max_response_length}",
        "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True",
        "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096",

        # Enable vLLM Prometheus metrics (critical for instrumentation)
        "actor_rollout_ref.rollout.disable_log_stats=False",
        "actor_rollout_ref.rollout.prometheus.enable=True",

        # Enable prefix caching (default True, explicit)
        "actor_rollout_ref.rollout.enable_prefix_caching=True",

        # Expert parallelism for MoE (Qwen3-30B-A3B has 128 experts, TP=4 needs EP)
        "actor_rollout_ref.rollout.expert_parallel_size=1",

        # Reference policy
        "actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True",
        "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=4096",
        "actor_rollout_ref.ref.fsdp_config.param_offload=True",

        # Reward
        f"reward.custom_reward_function.path={args.reward_func}",
        "reward.custom_reward_function.name=compute_score",

        # Trainer
        "trainer.balance_batch=True",
        "trainer.logger=['console']",
        "trainer.project_name=concurrl_phase2",
        "trainer.experiment_name=grpo_instrumented",
        "trainer.n_gpus_per_node=2",
        "trainer.nnodes=1",
        f"trainer.save_freq={args.save_freq}",
        f"trainer.test_freq={args.test_freq}",
        f"trainer.total_epochs={args.num_epochs}",
        "trainer.use_v1=False",
    ]

    if args.rollout_only:
        overrides.extend([
            "actor_rollout_ref.actor.use_kl_loss=False",
            "trainer.val_before_train=False",
            "trainer.save_freq=-1",
            "trainer.test_freq=-1",
            "+actor_rollout_ref.rollout.agent.agent_loop_manager_class="
            "script.verl_rollout_only.InstrumentedAgentLoopManager",
        ])

    # OTLP tracing for vLLM
    if args.enable_otel:
        overrides.extend([
            f"actor_rollout_ref.rollout.engine_kwargs.vllm.otlp_traces_endpoint={args.otlp_endpoint}",
            "actor_rollout_ref.rollout.engine_kwargs.vllm.collect_detailed_traces=all",
        ])

    # Extra overrides
    overrides.extend(args.extra_overrides)

    return overrides


# ---------------------------------------------------------------------------
# Timing log parser (for veRL's built-in timing_s/* metrics)
# ---------------------------------------------------------------------------
def parse_timing_from_log(log_text: str) -> list[dict]:
    timing_pattern = re.compile(r"timing_s/(\w+):\s+([\d.]+)")
    steps = []
    current_step = {}

    for line in log_text.split("\n"):
        if "step" in line.lower() and ("epoch" in line.lower() or "global_step" in line.lower()):
            if current_step:
                steps.append(current_step)
                current_step = {}

        match = timing_pattern.search(line)
        if match:
            stage_name = match.group(1)
            timing_s = float(match.group(2))
            current_step[f"t_{stage_name}_s"] = timing_s

    if current_step:
        steps.append(current_step)

    return steps


# ---------------------------------------------------------------------------
# Ray TaskRunner patching
# ---------------------------------------------------------------------------
def _make_task_runner():
    """Create a TaskRunner that applies instrumentation in the Ray worker.

    run_ppo() creates a Ray remote TaskRunner. The TaskRunner.run() method
    executes in a separate Ray worker process where the monkey-patches from
    phase2_instrument.py are NOT applied (they only exist in the main process).

    This function patches TaskRunner.run() to import and apply the
    instrumentation module before the original run() executes.
    """
    import ray
    from verl.trainer.main_ppo_v0 import BaseTaskRunner, TaskRunner

    original_runner_class = TaskRunner.__ray_metadata__.modified_class
    original_run = original_runner_class.run

    class InstrumentedTaskRunner(BaseTaskRunner):
        def run(self, config):
        # Import and apply patches in the Ray worker process
            project = str(project_dir)
            if project not in sys.path:
                sys.path.insert(0, project)
            try:
                if args.rollout_only:
                    from script.verl_rollout_only import install_driver_patches
                    install_driver_patches(
                        warmup_steps=args.warmup_steps,
                        measured_steps=args.measured_steps,
                    )
                    print(f"[05_instrumented] Rollout-only patches applied in TaskRunner "
                          f"(PID={os.getpid()})")
                else:
                    import script.phase2_instrument as worker_inst
                    if args.vllm_server_url:
                        worker_inst.set_external_vllm_url(args.vllm_server_url)
                    worker_inst.apply_patches()
                    print(f"[05_instrumented] Instrumentation patches applied in TaskRunner "
                          f"(PID={os.getpid()})")
            except Exception as e:
                print(f"[05_instrumented] WARNING: Failed to apply patches in TaskRunner: {e}")
                raise

            return original_run(self, config)

    return ray.remote(InstrumentedTaskRunner)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"[05_instrumented] Phase 2: Instrumented GRPO Training")
    print(f"[05_instrumented] Model:         {args.model}")
    print(f"[05_instrumented] Data:          {args.data_path}")
    print(f"[05_instrumented] Rollout N:     {args.rollout_n}")
    print(f"[05_instrumented] Batch size:    {args.train_batch_size}")
    print(f"[05_instrumented] B x G:         {args.train_batch_size * args.rollout_n}")
    print(f"[05_instrumented] Epochs:        {args.num_epochs}")
    print(f"[05_instrumented] OTel:          {'enabled' if args.enable_otel else 'disabled'}")
    print(f"[05_instrumented] Jaeger URL:    {args.jaeger_url}")
    print(f"[05_instrumented] Output:        {out_path}")
    print(f"[05_instrumented] Mode:          "
          f"{'verl_group_rollout_only' if args.rollout_only else 'grpo_training'}")
    if args.rollout_only:
        print(f"[05_instrumented] Steps:         "
              f"{args.warmup_steps} warmup + {args.measured_steps} measured")
        print(f"[05_instrumented] Trace:         "
              f"{'disabled' if args.disable_rollout_trace else trace_path}")
    print()

    overrides = build_hydra_overrides()

    if args.dry_run:
        print("[05_instrumented] Dry run — Hydra overrides:")
        for o in overrides:
            print(f"  {o}")
        return

    # Import veRL and apply patches
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    # Apply patches in main process (for TaskRunner.run wrapper)
    inst.reset()
    if args.rollout_only:
        if trace_path.exists():
            trace_path.unlink()
        os.environ["CONCURL_ROLLOUT_TRACE"] = (
            "0" if args.disable_rollout_trace else "1"
        )
        os.environ["CONCURL_ROLLOUT_TRACE_PATH"] = str(trace_path.resolve())
    else:
        if args.vllm_server_url:
            inst.set_external_vllm_url(args.vllm_server_url)
        inst.apply_patches()
    task_runner_class = _make_task_runner()

    # Ensure Ray workers can access the project directory
    import ray
    if not ray.is_initialized():
        ray.init(runtime_env={
            "working_dir": str(project_dir),
            "env_vars": {
                "CONCURL_ROLLOUT_TRACE": os.environ.get("CONCURL_ROLLOUT_TRACE", "0"),
                "CONCURL_ROLLOUT_TRACE_PATH": os.environ.get(
                    "CONCURL_ROLLOUT_TRACE_PATH", ""
                ),
            },
            "excludes": [
                ".venv/**", ".venv-*/**", ".deps/**", ".tools/**",
                "venv_18/**", "result/**",
                ".git/**", "__pycache__/**", "*.egg-info/**",
                "data/**", "vllm_src_018/**", "Qwen/**",
                ".mimocode/**", "models/**", "*.parquet",
                "*.log",
            ],
        })

    from verl.trainer.main_ppo import run_ppo

    print(f"[05_instrumented] Launching veRL GRPO trainer...")
    print(f"[05_instrumented] ({len(overrides)} Hydra overrides)")
    print()

    # Prepare log file
    log_handle = None
    if args.log_file:
        log_path = Path(args.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = open(log_path, "w")
        print(f"[05_instrumented] Logging to: {log_path}")

    t_start = time.perf_counter()
    t_elapsed = 0.0

    # --- Real-time JSON dump thread ---
    import threading

    def _dump_intermediate(tag="intermediate"):
        """Write current instrumentation state to JSON (called periodically)."""
        try:
            inst_data = inst.get_instrumentation_data()
            log_text = ""
            if args.log_file and Path(args.log_file).exists():
                log_text = Path(args.log_file).read_text(encoding="utf-8", errors="replace")
            step_timings = parse_timing_from_log(log_text) if log_text else []
            output = {
                "benchmark": "phase2_grpo_instrumented",
                "status": tag,
                "model": args.model,
                "data_path": args.data_path,
                "rollout_n": args.rollout_n,
                "train_batch_size": args.train_batch_size,
                "concurrency": args.train_batch_size * args.rollout_n,
                "ppo_mini_batch_size": args.ppo_mini_batch_size,
                "max_prompt_length": args.max_prompt_length,
                "max_response_length": args.max_response_length,
                "num_epochs": args.num_epochs,
                "rollout_tp": args.rollout_tp,
                "enable_otel": args.enable_otel,
                "jaeger_url": args.jaeger_url,
                "total_elapsed_s": round(time.perf_counter() - t_start, 2),
                "num_steps": len(inst_data["rollout_steps"]),
                "step_timings": step_timings,
                "rollout_steps": inst_data["rollout_steps"],
                "server_addresses": inst_data["server_addresses"],
                "num_requests_total": inst_data["num_requests"],
                "request_records": inst_data.get("request_records", []),
                "verl_overrides": overrides,
            }
            out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
            print(f"[05_instrumented] Intermediate dump ({tag}): {out_path}")
        except Exception as ex:
            print(f"[05_instrumented] WARNING: intermediate dump failed: {ex}")

    def _periodic_dumper():
        """Background thread that dumps JSON every 60s."""
        while not _stop_event.is_set():
            _stop_event.wait(60)
            if not _stop_event.is_set():
                _dump_intermediate("periodic")

    _stop_event = threading.Event()
    _dumper_thread = threading.Thread(target=_periodic_dumper, daemon=True)
    _dumper_thread.start()

    try:
        # Find veRL config directory
        import verl
        verl_dir = Path(verl.__file__).parent
        config_dir = str(verl_dir / "trainer" / "config")

        with initialize_config_dir(config_dir=config_dir, version_base=None):
            config = compose(config_name="ppo_trainer", overrides=overrides)

            if log_handle:
                import io
                old_stdout, old_stderr = sys.stdout, sys.stderr
                sys.stdout = sys.stderr = io.TextIOWrapper(
                    log_handle.buffer, encoding="utf-8", line_buffering=True
                )
                try:
                    run_ppo(config, task_runner_class=task_runner_class)
                finally:
                    sys.stdout, sys.stderr = old_stdout, old_stderr
            else:
                run_ppo(config, task_runner_class=task_runner_class)

        t_elapsed = time.perf_counter() - t_start
        print(f"[05_instrumented] Execution completed in {t_elapsed:.1f}s")

    except Exception as e:
        t_elapsed = time.perf_counter() - t_start
        print(f"[05_instrumented] ERROR after {t_elapsed:.1f}s: {e}")
        # Dump partial results on error
        _dump_intermediate("error")
        if log_handle:
            log_handle.close()
        raise
    finally:
        _stop_event.set()
        if log_handle:
            log_handle.close()

    # Final dump
    _dump_intermediate("final")

    # Collect instrumentation data
    inst_data = inst.get_instrumentation_data()

    # Parse timing from log file
    log_text = ""
    if args.log_file and Path(args.log_file).exists():
        log_text = Path(args.log_file).read_text(encoding="utf-8", errors="replace")
    step_timings = parse_timing_from_log(log_text) if log_text else []

    # Build final output
    rollout_only_metrics = None
    if args.rollout_only:
        from script.verl_rollout_only import aggregate_trace
        rollout_only_metrics = aggregate_trace(
            trace_path,
            expected_group_size=args.rollout_n,
        )

    output = {
        "benchmark": "phase2_grpo_instrumented",
        "status": "completed",
        "execution_mode": (
            "verl_group_rollout_only" if args.rollout_only else "grpo_training"
        ),
        "model": args.model,
        "data_path": args.data_path,
        "rollout_n": args.rollout_n,
        "train_batch_size": args.train_batch_size,
        "concurrency": args.train_batch_size * args.rollout_n,
        "ppo_mini_batch_size": args.ppo_mini_batch_size,
        "max_prompt_length": args.max_prompt_length,
        "max_response_length": args.max_response_length,
        "num_epochs": args.num_epochs,
        "rollout_tp": args.rollout_tp,
        "enable_otel": args.enable_otel,
        "jaeger_url": args.jaeger_url,
        "total_elapsed_s": round(t_elapsed, 2),
        "num_steps": len(inst_data["rollout_steps"]),
        "step_timings": step_timings,
        "rollout_steps": inst_data["rollout_steps"],
        "server_addresses": inst_data["server_addresses"],
        "num_requests_total": inst_data["num_requests"],
        "request_records": inst_data.get("request_records", []),
        "rollout_only_metrics": rollout_only_metrics,
        "rollout_trace_enabled": not args.disable_rollout_trace,
        "rollout_trace_path": str(trace_path) if args.rollout_only else None,
        "verl_overrides": overrides,
    }

    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\n[05_instrumented] Results saved to {out_path}")

    if rollout_only_metrics:
        stages = rollout_only_metrics["request_stages"]
        print("\nveRL GROUP ROLLOUT-ONLY")
        for name in [
            "rollout_first_output_wait",
            "prompt_preparation",
            "request_admission",
            "scheduler_wait",
            "first_scheduled_to_output",
        ]:
            metric = stages.get(name, {})
            print(f"  {name:32s} P95={metric.get('p95_ms')} ms")
        group_p95 = rollout_only_metrics["group_ready_latency"].get("p95_ms")
        throughput = rollout_only_metrics["throughput"]
        print(f"  {'group_ready':32s} P95={group_p95} ms")
        print(f"  output_tokens/s={throughput.get('output_tokens_per_s')}  "
              f"completed_groups/s={throughput.get('completed_groups_per_s')}")
        print(f"  KV alloc fails={rollout_only_metrics['kv_allocation']['failure_attempts']}  "
              f"preemptions={rollout_only_metrics['preemption']['events']}  "
              f"terminal failures={rollout_only_metrics['terminal_request_failures']}")

    # Print summary
    rollout_steps = inst_data["rollout_steps"]
    if rollout_steps:
        print(f"\n{'='*100}")
        print(f"INSTRUMENTED GRPO TRAINING — Rollout Timing Summary")
        print(f"{'='*100}")
        print(f"  {'Step':>4s} {'Rollout(ms)':>12s} {'Req':>5s} {'OK':>4s} "
              f"{'P50(ms)':>10s} {'P95(ms)':>10s} {'Tokens':>8s} {'Tok/s':>8s} "
              f"{'P50_lat':>10s} {'P95_lat':>10s}")
        print(f"  {'─'*4} {'─'*12} {'─'*5} {'─'*4} "
              f"{'─'*10} {'─'*10} {'─'*8} {'─'*8} "
              f"{'─'*10} {'─'*10}")
        for step in rollout_steps:
            print(f"  {step['step']:4d} "
                  f"{step['t_rollout_ms']:12.0f} "
                  f"{step['num_requests']:5d} "
                  f"{step['num_successful']:4d} "
                  f"{step['group_completion_p50_ms']:10.0f} "
                  f"{step['group_completion_p95_ms']:10.0f} "
                  f"{step['output_tokens_total']:8d} "
                  f"{step['output_token_throughput']:8.1f} "
                  f"{step['request_latency_p50_ms']:10.0f} "
                  f"{step['request_latency_p95_ms']:10.0f}")
        print(f"{'='*100}")


if __name__ == "__main__":
    main()
